from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_attn_bias(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_ptr,
    l_ptr,
    qkv_stride_b,
    qkv_stride_h,
    qkv_stride_sq,
    qkv_stride_hd,
    ml_stride_b,
    ml_stride_h,
    attn_bias_ptr,
    attn_bias_stride_h,
    attn_bias_stride_r,
    attn_bias_stride_c,
    BLOCK_HD: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    head_scale,
    num_head,
    context_sq,
):
    """Flash Attention with causal masking and attention bias mask."""

    q_chunk_pid = tl.program_id(axis=0)  # parallelize across sq chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = bh_pid // num_head
    off_h = bh_pid % num_head

    bh_offset = off_bs.to(tl.int64) * qkv_stride_b + off_h.to(tl.int64) * qkv_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        offsets=(0, 0),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    off_h = bh_pid % num_head
    attn_h_offset = off_h.to(tl.int64) * attn_bias_stride_h

    attn_bias_block_ptr = tl.make_block_ptr(
        attn_bias_ptr + attn_h_offset,
        shape=(context_sq, context_sq),
        block_shape=(BLOCK_SQ, BLOCK_SQ),
        strides=(attn_bias_stride_r, attn_bias_stride_c),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    out = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)
    m_i = tl.full([BLOCK_SQ], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # scale by 1/ln(2), 2^x much faster than e^x
    ln2_inv: tl.constexpr = 1.44269504

    head_scale *= ln2_inv
    head_scale = head_scale.to(tl.float16)

    q *= head_scale
    max_range = context_sq
    max_range = q_chunk_pid * BLOCK_SQ + 1

    for chunk in range(0, max_range - 1, BLOCK_SQ):
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)

        attn_bias = tl.load(attn_bias_block_ptr)
        attn_bias *= ln2_inv

        s_ij = tl.dot(q, k, allow_tf32=False)  # [BLOCK_SQ, BLOCK_SK]

        s_ij += attn_bias.to(tl.float32)

        m_ij = tl.max(s_ij, axis=1)  # [BLOCK_SQ, ]
        p_ij = tl.math.exp2(s_ij - m_ij[:, None])  # [BLOCK_SQ, BLOCK_SK]
        l_ij = tl.sum(p_ij, axis=1)  # [BLOCK_SQ, ]

        m_i_new = tl.maximum(m_i, m_ij)

        running_correction = tl.math.exp2(m_i - m_i_new)
        new_correction = tl.math.exp2(m_ij - m_i_new)

        l_i_new = (running_correction * l_i) + (new_correction * l_ij)

        out = (l_i * running_correction)[:, None] * out

        out += new_correction[:, None] * tl.dot(
            p_ij.to(tl.float16), v, allow_tf32=False
        )

        out /= (l_i_new)[:, None]

        m_i = m_i_new
        l_i = l_i_new

        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SQ))
        v_block_ptr = tl.advance(v_block_ptr, offsets=(BLOCK_SQ, 0))
        attn_bias_block_ptr = tl.advance(attn_bias_block_ptr, offsets=(0, BLOCK_SQ))

    # final block - we reuse code here to remove conditionals from for loop
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    attn_bias = tl.load(attn_bias_block_ptr)
    attn_bias *= ln2_inv

    s_ij = tl.dot(q, k, allow_tf32=False)  # [BLOCK_SQ, BLOCK_SK]
    s_ij += attn_bias.to(tl.float32)

    offs_k = tl.arange(0, BLOCK_SQ)
    offs_q = tl.arange(0, BLOCK_SQ)

    offs = max_range - 1
    s_ij = tl.where(
        q_chunk_pid * BLOCK_SQ + offs_k[:, None] >= (offs + offs_q[None, :]),
        s_ij,
        float("-inf"),
    )

    m_ij = tl.max(s_ij, axis=1)  # [BLOCK_SQ, ]
    p_ij = tl.math.exp2(s_ij - m_ij[:, None])  # [BLOCK_SQ, BLOCK_SK]
    l_ij = tl.sum(p_ij, axis=1)  # [BLOCK_SQ, ]
    m_i_new = tl.maximum(m_i, m_ij)
    running_correction = tl.math.exp2(m_i - m_i_new)
    new_correction = tl.math.exp2(m_ij - m_i_new)
    l_i_new = (running_correction * l_i) + (new_correction * l_ij)
    out = (l_i * running_correction)[:, None] * out
    out += new_correction[:, None] * tl.dot(p_ij.to(tl.float16), v, allow_tf32=False)
    out /= (l_i_new)[:, None]
    m_i = m_i_new
    l_i = l_i_new

    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    tl.store(out_block_ptr, value=out.to(tl.float16))

    bh_offset = off_bs.to(tl.int64) * ml_stride_b + off_h.to(tl.int64) * ml_stride_h

    # store m and l which are used in backward pass to recreate softmax activation
    m_ptr_start = m_ptr + (bh_offset) + (q_chunk_pid * BLOCK_SQ)
    l_ptr_start = l_ptr + (bh_offset) + (q_chunk_pid * BLOCK_SQ)

    tl.store(m_ptr_start + offs_q, m_i)
    tl.store(l_ptr_start + offs_q, l_i)


def flash_wrapper_fwd_attn_bias(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function wrapping flash attention forward kernel with custom attention bias mask."""

    batch, nh, sq, hd = q.shape

    BLOCK_HD = triton.next_power_of_2(hd)
    BLOCK_SQ = 64 if BLOCK_HD < 128 else 32
    num_warps = 4 if BLOCK_HD <= 128 else 8

    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}"
    assert (
        attn_bias.shape[0] == nh
    ), f"Expected attention bias to have leading dimension equal to number of heads ({nh})"
    assert (
        attn_bias.shape[1] == attn_bias.shape[2]
    ), f"Expected attention bias to be a square but got nr = {attn_bias.shape[1]} and nc = {attn_bias.shape[2]}"

    out = torch.empty_like(q)

    def grid(META):
        return (triton.cdiv(sq, META["BLOCK_SQ"]), batch * nh)

    m = torch.empty((batch, nh, sq), device=q.device, dtype=torch.float16)
    l = torch.empty_like(m)

    head_scale = 1.0 / (q.shape[-1] ** 0.5)

    flash_attn_fwd_attn_bias[grid](
        q,
        k,
        v,
        out,
        m,
        l,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        m.stride(0),
        m.stride(1),
        attn_bias,
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        BLOCK_HD=BLOCK_HD,
        BLOCK_SQ=BLOCK_SQ,
        num_warps=num_warps,
        num_stages=2,
        head_scale=head_scale,
        context_sq=sq,
        num_head=nh,
    )

    return out, m, l


@triton.jit
def flash_attn_bwd_attn_bias(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_ptr,
    l_ptr,
    dO_ptr,
    dV_ptr,
    dK_ptr,
    dQ_ptr,
    qkv_stride_b,
    qkv_stride_h,
    qkv_stride_sq,
    qkv_stride_hd,
    ml_stride_b,
    ml_stride_h,
    attn_bias_ptr,
    attn_bias_stride_h,
    attn_bias_stride_r,
    attn_bias_stride_c,
    head_scale,
    BLOCK_HD: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    context_sq,
    num_head,
):
    """Flash Attention backward pass with causal masking and attention bias mask"""

    kv_chunk_pid = tl.program_id(axis=0)  # parallelize across kv chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = (bh_pid // num_head,)
    off_h = (bh_pid % num_head,)

    bh_offset = off_bs.to(tl.int64) * qkv_stride_b + off_h.to(tl.int64) * qkv_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    off_h = bh_pid % num_head
    attn_h_offset = off_h.to(tl.int64) * attn_bias_stride_h

    attn_bias_block_ptr = tl.make_block_ptr(
        attn_bias_ptr + attn_h_offset,
        shape=(context_sq, context_sq),
        block_shape=(BLOCK_SQ, BLOCK_SQ),
        strides=(attn_bias_stride_r, attn_bias_stride_c),
        order=(1, 0),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    dV = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)
    dK = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    k_trans = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)

    # constants so tl.math.exp2 can be used instead of tl.exp
    ln2_inv: tl.constexpr = 1.44269504
    ln2: tl.constexpr = 0.6931471824645996

    head_scale *= ln2_inv
    head_scale = head_scale.to(tl.float16)
    max_range = context_sq
    min_range = kv_chunk_pid * BLOCK_SQ

    offs_k = tl.arange(0, BLOCK_SQ)
    offs_q = (kv_chunk_pid * BLOCK_SQ) + tl.arange(0, BLOCK_SQ)

    ml_bh_offset = off_bs.to(tl.int64) * ml_stride_b + off_h.to(tl.int64) * ml_stride_h

    m_ptr_start = m_ptr + ml_bh_offset
    l_ptr_start = l_ptr + ml_bh_offset

    # loop is split into pre/post masking to remove conditional use
    for q_chunk in range(0, min_range + 1, BLOCK_SQ):
        q = tl.load(q_block_ptr)
        dout = tl.load(dout_block_ptr)
        out = tl.load(out_block_ptr)
        q *= head_scale

        attn_bias = tl.load(attn_bias_block_ptr)
        attn_bias *= ln2_inv

        m_i = tl.load(m_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]
        l_i = tl.load(l_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]

        s_ij = tl.dot(q, k_trans, allow_tf32=False)
        s_ij += attn_bias.to(tl.float32)

        s_ij = tl.where(
            (q_chunk + offs_k[:, None]) >= (offs_q[None, :]),
            s_ij,
            float("-inf"),
        )

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dV += tl.dot(tl.trans(P_ij.to(tl.float16)), dout, allow_tf32=False)

        dP_ij = tl.dot(dout, v, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]
        dS_ij = P_ij * (dP_ij - D_i)

        dK += tl.dot(tl.trans(dS_ij.to(tl.float16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SQ, 0))
        out_block_ptr = tl.advance(out_block_ptr, offsets=(BLOCK_SQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SQ, 0))
        attn_bias_block_ptr = tl.advance(attn_bias_block_ptr, offsets=(BLOCK_SQ, 0))

    min_range_offset = min_range + BLOCK_SQ

    for q_chunk in range(min_range_offset, max_range, BLOCK_SQ):
        q = tl.load(q_block_ptr)
        dout = tl.load(dout_block_ptr)
        out = tl.load(out_block_ptr)
        q *= head_scale

        attn_bias = tl.load(attn_bias_block_ptr)
        attn_bias *= ln2_inv

        m_i = tl.load(m_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]
        l_i = tl.load(l_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]
        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        s_ij += attn_bias.to(tl.float32)

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dV += tl.dot(tl.trans(P_ij.to(tl.float16)), dout, allow_tf32=False)

        dP_ij = tl.dot(dout, v, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]
        dS_ij = P_ij * (dP_ij - D_i)

        dK += tl.dot(tl.trans(dS_ij.to(tl.float16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SQ, 0))
        out_block_ptr = tl.advance(out_block_ptr, offsets=(BLOCK_SQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SQ, 0))
        attn_bias_block_ptr = tl.advance(attn_bias_block_ptr, offsets=(BLOCK_SQ, 0))

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    tl.store(dV_block_ptr, value=dV.to(tl.float16))
    tl.store(dK_block_ptr, value=(ln2 * dK).to(tl.float16))

    # ----------
    # compute dQ
    # ----------

    # reset block pointers
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    off_h = bh_pid % num_head
    attn_h_offset = off_h.to(tl.int64) * attn_bias_stride_h

    attn_bias_block_ptr = tl.make_block_ptr(
        attn_bias_ptr + attn_h_offset,
        shape=(context_sq, context_sq),
        block_shape=(BLOCK_SQ, BLOCK_SQ),
        strides=(attn_bias_stride_r, attn_bias_stride_c),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    q = tl.load(q_block_ptr)
    q *= head_scale

    dQ = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SQ)

    max_range = kv_chunk_pid * BLOCK_SQ + 1
    final = max_range - 1

    m_ptr_start = m_ptr + (ml_bh_offset) + (kv_chunk_pid * BLOCK_SQ)
    l_ptr_start = l_ptr + (ml_bh_offset) + (kv_chunk_pid * BLOCK_SQ)

    m_i = tl.load(m_ptr_start + offs_k, mask=offs_k < context_sq)[:, None]
    l_i = tl.load(l_ptr_start + offs_k, mask=offs_k < context_sq)[:, None]

    dout = tl.load(dout_block_ptr)
    out = tl.load(out_block_ptr)

    for q_chunk in range(0, final, BLOCK_SQ):
        v_trans = tl.load(v_block_ptr)
        k_trans = tl.load(k_block_ptr)

        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        attn_bias = tl.load(attn_bias_block_ptr)
        attn_bias *= ln2_inv

        s_ij += attn_bias.to(tl.float32)

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dP_ij = tl.dot(dout, v_trans, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]

        dS_ij = P_ij * (dP_ij - D_i)

        dQ += tl.dot(dS_ij.to(tl.float16), tl.trans(k_trans), allow_tf32=False)

        v_block_ptr = tl.advance(v_block_ptr, offsets=(0, BLOCK_SQ))
        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SQ))
        attn_bias_block_ptr = tl.advance(attn_bias_block_ptr, offsets=(0, BLOCK_SQ))

    v_trans = tl.load(v_block_ptr)
    k_trans = tl.load(k_block_ptr)
    attn_bias = tl.load(attn_bias_block_ptr)
    attn_bias *= ln2_inv

    s_ij = tl.dot(q, k_trans, allow_tf32=False)
    s_ij += attn_bias.to(tl.float32)

    # causal masking on final block
    s_ij = tl.where(
        kv_chunk_pid * BLOCK_SQ + offs_k[:, None] >= (final + offs_k[None, :]),
        s_ij,
        float("-inf"),
    )

    P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

    dP_ij = tl.dot(dout, v_trans, allow_tf32=False)
    D_i = tl.sum(dout * out, axis=1)[:, None]

    dS_ij = P_ij * (dP_ij - D_i)
    dQ += tl.dot(dS_ij.to(tl.float16), tl.trans(k_trans), allow_tf32=False)

    tl.store(
        dQ_block_ptr,
        (ln2 * head_scale * dQ).to(tl.float16),
    )


def flash_wrapper_bwd_attn_bias(
    grad_output: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    attn_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function wrapping flash attention backward kernel with custom attention bias mask."""
    batch, nh, sq, hd = q.shape

    BLOCK_HD = triton.next_power_of_2(hd)
    BLOCK_SQ = 64 if BLOCK_HD < 128 else 32

    num_warps = 4 if BLOCK_HD < 128 else 8

    assert hd in [32, 64, 128], "Only head_dims of [32,64,128] are supported."
    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}"

    dQ = torch.zeros_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)

    def grid(META):
        return (triton.cdiv(sq, META["BLOCK_SQ"]), batch * nh)

    head_scale = (1.0) / (q.shape[-1] ** 0.5)

    flash_attn_bwd_attn_bias[grid](
        q,
        k,
        v,
        out,
        m,
        l,
        grad_output,
        dV,
        dK,
        dQ,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        m.stride(0),
        m.stride(1),
        attn_bias,
        attn_bias.stride(0),
        attn_bias.stride(1),
        attn_bias.stride(2),
        head_scale=head_scale,
        BLOCK_HD=BLOCK_HD,
        BLOCK_SQ=BLOCK_SQ,
        context_sq=sq,
        num_warps=num_warps,
        num_stages=3,
        num_head=nh,
    )

    return dQ, dK, dV
