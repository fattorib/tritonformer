"""Standard Transformer with Causal Attention."""

import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

import tritonformer.nn as tnn
from tritonformer import (
    cross_entropy_loss,
    fast_causal_attention,
    fast_causal_attention_with_bias,
)

# -----------------
# Utility Functions
# -----------------


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    num_attention_heads: int
    head_dim: int
    ffn_dim: int
    num_hidden_layers: int
    use_linear_bias: bool
    attn_bias: bool


def get_slopes(n: int) -> List[float]:
    """Create slopes for ALiBi attention bias."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def generate_alibi_mask(seq_len: int, num_heads: int) -> torch.Tensor:
    """Create ALiBi attention bias."""
    a = -torch.tril(
        torch.arange(seq_len).view(seq_len, 1).repeat(1, seq_len)
        + torch.arange(0, -seq_len, -1)
    )

    a = a.to("cuda:0")
    a = a.to(torch.float16)

    slopes = torch.tensor(get_slopes(num_heads), device=a.device, dtype=a.dtype)
    alibi_mask = a * slopes[..., None, None]

    mask = torch.tril(
        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda:0")
    )

    alibi_mask = alibi_mask.masked_fill(
        mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
    )
    return alibi_mask


# ---------------------
# Weight Initialization
# ---------------------


def _weights_init(m, num_layers):
    if isinstance(m, (tnn.Linear)):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(m, tnn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, (nn.Embedding)):
        m.weight.data.normal_(mean=0.0, std=0.02)

    elif isinstance(m, tnn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

    for name, p in m.named_parameters():
        # scale residuals by 1/sqrt(L) and init lm head to zeros
        if "out_proj" in name and "weight" in name:
            p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * num_layers)))

        if "dense_4h_to_h" in name and "weight" in name:
            p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * num_layers)))

        if "logits_out" in name and "weight" in name:
            p.data.zero_()


# ------------------
# Module Definitions
# ------------------


class CausalAttention(nn.Module):
    """Self Attention Module."""

    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = (
            config.hidden_size // config.num_attention_heads
        )

        self.q_proj = tnn.Linear(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
            bias=False,
        )
        self.k_proj = tnn.Linear(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
            bias=False,
        )
        self.v_proj = tnn.Linear(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
            bias=False,
        )
        self.out_proj = tnn.Linear(
            config.hidden_size,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
            bias=False,
        )

    def forward(
        self, hidden_states: torch.Tensor, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, q_seq_len, hidden_dim = hidden_states.shape

        query_layer = rearrange(
            self.q_proj(hidden_states),
            "b s (nh hd) -> b nh s hd",
            nh=self.num_attention_heads,
            hd=self.hidden_size_per_attention_head,
        )
        key_layer = rearrange(
            self.k_proj(hidden_states),
            "b s (nh hd) -> b nh s hd",
            nh=self.num_attention_heads,
            hd=self.hidden_size_per_attention_head,
        )
        value_layer = rearrange(
            self.v_proj(hidden_states),
            "b s (nh hd) -> b nh s hd",
            nh=self.num_attention_heads,
            hd=self.hidden_size_per_attention_head,
        )

        if attn_bias is None:
            attn = fast_causal_attention.apply(query_layer, key_layer, value_layer)
        else:
            attn = fast_causal_attention_with_bias.apply(
                query_layer, key_layer, value_layer, attn_bias
            )

        context_layer = (
            attn.transpose(1, 2).contiguous().view(batch_size, q_seq_len, hidden_dim)
        )

        output = self.out_proj(context_layer)

        return output


class MLPBlock(nn.Module):
    """MLPBlock Module."""

    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.dense_h_to_4h = tnn.Linear(
            config.hidden_size,
            config.ffn_dim,
            device=device,
            dtype=torch.float16,
            fuse_activation=True,
            bias=config.use_linear_bias,
        )
        self.dense_4h_to_h = tnn.Linear(
            config.ffn_dim,
            config.hidden_size,
            device=device,
            dtype=torch.float16,
            bias=config.use_linear_bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class TransformerLayer(nn.Module):
    """TransformerBlock"""

    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.input_layernorm = tnn.LayerNorm(
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.post_attention_layernorm = tnn.LayerNorm(
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.attention = CausalAttention(config, device=device)
        self.mlp = MLPBlock(config, device=device)

    def forward(
        self, hidden_states: torch.Tensor, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        ln_output = self.input_layernorm(hidden_states)

        hidden_states = self.attention(ln_output, attn_bias)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.embed_positions = None
        if not config.attn_bias:
            self.embed_positions = nn.Embedding(
                num_embeddings=config.max_position_embeddings,
                embedding_dim=config.hidden_size,
                device=device,
                dtype=torch.float16,
            )
        self.layer_list = nn.ModuleList([])
        for layer_i in range(config.num_hidden_layers):
            self.layer_list.append(TransformerLayer(config, device=device))

        self.final_layernorm = tnn.LayerNorm(
            config.hidden_size,
            device=device,
            dtype=torch.float16,
        )
        self.logits_out = tnn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.float16,
            fuse_activation=False,
        )

        init_function = partial(
            _weights_init, **{"num_layers": config.num_hidden_layers}
        )
        self.apply(init_function)

        if config.attn_bias:
            self.register_buffer(
                "attn_bias",
                generate_alibi_mask(
                    config.max_position_embeddings, config.num_attention_heads
                )[0, ...],
            )
        else:
            self.attn_bias = None

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        token_embeddings = self.embed_tokens(input_ids)

        if self.embed_positions is not None:
            positions = torch.arange(seq_len, device=token_embeddings.device)[
                None
            ].expand(
                batch_size,
                -1,
            )
            pos_embeddings = self.embed_positions(positions)
            hidden_states = token_embeddings + pos_embeddings
        else:
            hidden_states = token_embeddings

        for layer_i, layer in enumerate(self.layer_list):
            hidden_states = layer(hidden_states=hidden_states, attn_bias=self.attn_bias)

        hidden_states = self.final_layernorm(hidden_states)

        logits_lm = self.logits_out(hidden_states)

        if labels is not None:
            shift_logits = logits_lm[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = cross_entropy_loss.apply(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(
                    -1,
                ),
            )

            return logits_lm, loss.mean()

        else:
            return logits_lm
