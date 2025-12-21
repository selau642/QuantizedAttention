import torch
import helion
import helion.language as hl
from torch.nn.attention.flex_attention import flex_attention

import math
from typing import Tuple
from typing import cast

from helion._testing import DEVICE
from helion._testing import run_example

from torch.autograd import Function
HELION_INTERPRET=1

class FlashAttention_2_BF16_autograd_function(Function):
    """

    forward in bf16
    backward in fp32

    """
    @staticmethod
    def forward(
        ctx, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        causal: bool
    ):
        atten_bf16, bwd_normalizing_blk = helion_atten_bf16_fwd_training(q,k,v, causal) 

        ctx.args = (causal,)
        ctx.save_for_backward(q, k, v, atten_bf16, bwd_normalizing_blk)

    @staticmethod
    def backward(
        ctx,
        d_atten
    ):
        q, k, v, atten_bf16, bwd_normalizing_blk = ctx.saved_tensors
        causal = ctx.args
        dq, dk, dv = helion_atten_bwd(
            q,
            k,
            v,   
            atten_bf16,
            bwd_normalizing_blk,
            d_atten, 
            causal
        )

        return dq, dk, dv

@helion.kernel(
    static_shapes=True,
    # autotune_max_generations=8 
    # nvidia RTX3080 best config
    config=helion.Config(
        block_sizes=[4, 32, 16], 
        indexing=['pointer', 'pointer', 'block_ptr', 'block_ptr', 'pointer'], 
        l2_groupings=[64], 
        load_eviction_policies=['first', 'first', 'last'], 
        loop_orders=[[1, 0]], 
        num_stages=3, 
        num_warps=4, 
        pid_type='flat', 
        range_flattens=[None, None], 
        range_multi_buffers=[None, True], 
        range_num_stages=[0, 0], 
        range_unroll_factors=[0, 0], 
        range_warp_specializes=[]), 
)
def atten_fwd_training(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes forward for Attention
        Attention = Softmax(Q * K_T/SQRT(d_k)) * V

    Args:
        q: Query shape [batch, head, q_tokens, head_dim]
        k: Key shape [batch, head, k_tokens, head_dim]
        v: Value shape [batch, head, v_tokens, head_dim]

    Returns:
        Attention: shape [batch, head, q_tokens, head_dim] 
        bwd_normalizing_const: shape [batch*head, q_tokens] used for backward
    """

    batch, head, q_tokens, q_head_dim = q.shape
    k_batch, k_head, k_tokens, k_head_dim = k.shape
    v_batch, v_head, v_tokens, v_head_dim = k.shape

    # q_tokens = q.size(-2)
    # k_tokens = k.size(-2)
    # v_tokens = v.size(-2)

    assert k_tokens == v_tokens, "input k_tokens must match v_tokens"
    assert q_head_dim == k.size(-1) == v.size(-1), "all head dimensions must match for q, k, v tensors"

    # head_dim = q.size(-1)

    head_dim = q_head_dim
    q_bh = q.reshape([-1, q_tokens, head_dim])
    k_bh = k.reshape([-1, k_tokens, head_dim]).transpose(1, 2)
    v_bh = v.reshape([-1, v_tokens, head_dim])
    bwd_normalizing_blk = torch.zeros([batch*head, q_tokens], device=q_bh.device)
    atten_out = torch.empty_like(q_bh)
    sm_scale =  1.0 / math.sqrt(head_dim) 

    qk_scale = sm_scale * 1.44269504089 
    # 1.44269504089 = 1/ln(2), 
    # where ln = log base euler number(2.7182)

    for bh_tile , q_tile in hl.tile([batch*head, q_tokens]):

        # make buffers
        qk_max_blk = hl.full([bh_tile, q_tile], float("-inf"), dtype=torch.float32)
        exp_qk_sum_blk = torch.full_like(qk_max_blk, 1.0) 
        exp_qk_v_blk = hl.zeros([bh_tile, q_tile, head_dim], dtype=torch.float32)

        # load q
        q_blk = q_bh[bh_tile, q_tile, :]

        for k_tile in hl.tile(k_tokens):
            # load k
            k_blk = k_bh[bh_tile, :, k_tile]
            
            # q dot k
            qk_blk= torch.bmm(q_blk, k_blk) # batch matrix multiply

            if causal and q_tile.begin < k_tile.end - 1:
                mask_blk = torch.triu(
                    torch.full([q_tile, k_tile], 1.0, dtype=torch.bool), 
                    diagonal=q_tile.begin - k_tile.begin + 1
                    ).to(q_bh.device)

                qk_blk.masked_fill_(mask_blk, float("-inf"))

            # recalculate max tile
            next_qk_max_blk = torch.max(qk_max_blk, torch.amax(qk_blk, -1) * qk_scale)
            qk_blk = qk_blk * qk_scale - next_qk_max_blk[:, :, None]

            exp_qk_blk = torch.exp2(qk_blk) 
            # torch.exp2 is 2 power qk_tile, 
            # not e power qk_tile, 
            # qk_scale 1/ln2 with convert to 2 power qk_tile to e power qk_tile

            next_exp_qk_sum_blk = torch.sum(exp_qk_blk, -1)

            rescale = torch.exp2(qk_max_blk - next_qk_max_blk)
            exp_qk_sum_blk = exp_qk_sum_blk * rescale + next_exp_qk_sum_blk
            exp_qk_v_blk = exp_qk_v_blk * rescale[:, :, None]

            v_tile = v_bh[bh_tile, k_tile, :]

            exp_qk = exp_qk.to(v.dtype)
            exp_qk_v_blk = torch.baddbmm(exp_qk_v_blk, exp_qk, v_tile) 
            # batch add accumulator to batch matrix multiply exp_qk, v_tile
            qk_max_blk = next_qk_max_blk

        # if training: 
        #    qk_max_tile += torch.log2(exp_qk_sum_tile)


        # if training: 
        # store qk_max_tile + log2(exp_qk_sum_tile) to bwd_normalizing_const for backprop
        bwd_normalizing_blk[bh_tile, q_tile] = qk_max_blk + torch.log2(exp_qk_sum_blk)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile

        atten_blk = exp_qk_v_blk/ exp_qk_sum_blk[:, :, None]
        atten_out[bh_tile, q_tile, :] = atten_blk.to(atten_out.dtype)
    
    return atten_out.view([batch, head, q_tokens, head_dim]), bwd_normalizing_blk

@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_bf16_fwd_training(
    q_fp16: torch.Tensor,
    k_fp16: torch.Tensor,
    v_bf16: torch.Tensor,
    causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes forward for Attention in BrainFloat16
        Attention = Softmax(Q * K_T/SQRT(d_k)) * V

    1. bf16 has larger range but less mantissa(less range around 0) 
    2. fp16 has better decimal less range

    => prob logits in bf16, actual probs in fp16/fp32

    sum = mantissa fp16/fp32
    division = mantissa fp16/fp32
    multiply probs < 1 = like division = mantissa fp16/fp32

    multiply values > 1 = range bf16
    exp = range bf16
    dot_product = multiply and sum (information in cosine similarity which is angles = mantissa more important)

    Args:
        q: Query shape [batch, head, q_tokens, head_dim]
        k: Key shape [batch, head, k_tokens, head_dim]
        v: Value shape [batch, head, v_tokens, head_dim]

    Returns:
        Attention: shape [batch, head, q_tokens, head_dim] 
        bwd_normalizing_blk : shape [batch*head, q_tokens] used for backward
    """
    BETA = 2.0 
    # at BETA=8.0, exp(QK) / Sum(exp(QK)) = overflows to NaN

    batch, head, q_tokens, q_head_dim = q_fp16.shape
    k_batch, k_head, k_tokens, k_head_dim = k_fp16.shape
    v_batch, v_head, v_tokens, v_head_dim = v_bf16.shape
    cuda_device=q_fp16.device

    assert k_tokens == v_tokens, "input k_tokens must match v_tokens"
    assert q_head_dim == k_fp16.size(-1) == v_bf16.size(-1), "all head dimensions must match for q, k, v tensors"

    head_dim = q_head_dim
    q_bh_fp16 = q_fp16.reshape([-1, q_tokens, head_dim])
    k_bh_fp16 = k_fp16.reshape([-1, k_tokens, head_dim]).transpose(1, 2)
    v_bh_bf16 = v_bf16.reshape([-1, v_tokens, head_dim])

    bwd_normalizing_blk_fp32 = torch.zeros(
        [batch*head, q_tokens], 
        dtype=torch.float32, 
        device=cuda_device
        )
    atten_out_fp32 = torch.empty_like(
        q_bh_fp16, 
        dtype=torch.float32, 
        device=cuda_device
        )
    sm_scale =  1.0 / math.sqrt(head_dim) 

    qk_scale = sm_scale * 1.44269504 
    # 1.44269504 = 1/ln(2), 
    # where ln = log base euler number(2.7182)

    # min_bfloat16 = torch.finfo(torch.bfloat16).min
    for bh_tile, q_tile in hl.tile([batch*head, q_tokens]):

        # make buffers
        qk_max_blk_bf16 = hl.full([bh_tile, q_tile, 1], float("-inf"), dtype=torch.bfloat16)
        exp_qk_sum_blk_fp32 = hl.full([bh_tile, q_tile, 1], 1.0, dtype=torch.float32) 
        exp_qk_v_blk_fp32 = hl.zeros([bh_tile, q_tile, head_dim], dtype=torch.float32)

        # load q
        q_blk_fp16 = q_bh_fp16[bh_tile, q_tile, :]

        for k_tile in hl.tile(k_tokens):
            # load k
            k_blk_fp16 = k_bh_fp16[bh_tile, :, k_tile]
            
            # q dot k
            qk_blk_fp16 = torch.bmm(q_blk_fp16, k_blk_fp16)

            
            qk_blk_bf16 = qk_blk_fp16.to(torch.bfloat16) 
            # batch matrix multiply
            # fp16 inputs will still accumulate in fp32
            # qk_tile.shape = [bh_idx_list, q_idx_list, k_idx_list]

            # if causal and begin_q < end_k - 1:

            #     mask_blk = torch.triu(
            #         torch.full([q_tile, k_tile], 1.0, dtype=torch.bool, device=cuda_device), 
            #         diagonal=begin_q - begin_k + 1
            #         )

                # qk_blk_bf16.masked_fill_(mask_blk, float("-inf"))

            # recalculate max tile
            qk_max_blk_bf16 = torch.max(qk_max_blk_bf16, torch.amax(qk_blk_bf16, -1, keepdim=True) * qk_scale).to(torch.bfloat16)
            """


            bf16 adjustment to avoid floating point rounding errors


            """
            approx_max = (qk_blk_bf16 >= (qk_max_blk_bf16 - 1e-3))
            # qk_tile.shape = [bh_idx_list, q_idx_list, k_idx_list]
            num_approx_max = torch.sum(approx_max, dim=-1, keepdims=True)
            # num_approx_max.shape = [bh_idx_list, q_idx_list, 1]
            more_than_one_approx_max = (num_approx_max > 1)

            qk_max_blk_bf16 = torch.where(
                more_than_one_approx_max & (qk_max_blk_bf16 > 0), 
                BETA * qk_max_blk_bf16, 
                qk_max_blk_bf16
            )

            qk_max_blk_bf16 = torch.where(
                more_than_one_approx_max & (qk_max_blk_bf16 < 0), 
                0.0*qk_max_blk_bf16, # bug where qk_max_blk_bf16 assign fp32 in loop but init in fp32
                qk_max_blk_bf16
            )

            # rescaling qk_tile
            qk_blk_bf16 = qk_blk_bf16 * qk_scale - qk_max_blk_bf16

            exp_qk_blk_bf16 = torch.exp2(qk_blk_bf16.to(torch.float32)).to(torch.bfloat16)
            # torch.exp2 is 2 power qk_tile, 
            # not e power qk_tile, 
            # qk_scale 1/ln2 with convert to 2 power qk_tile to e power qk_tile

            next_exp_qk_sum_blk_fp32 = torch.sum(exp_qk_blk_bf16.to(torch.float32), -1, keepdim=True)
            # qk_max_blk_bf16 = qk_max_blk_bf16.squeeze(-1)

            rescale_bf16 = torch.exp2((qk_max_blk_bf16 - qk_max_blk_bf16).to(torch.float32)).to(torch.bfloat16)
            exp_qk_sum_blk_fp32 = exp_qk_sum_blk_fp32 * rescale_bf16.to(torch.float32) + next_exp_qk_sum_blk_fp32
            exp_qk_v_blk_fp32= exp_qk_v_blk_fp32* rescale_bf16

            v_blk_bf16 = v_bh_bf16[bh_tile, k_tile, :]

            #exp_qk_blk = exp_qk_blk_bf16.to(v.dtype)

            exp_qk_v_blk_fp32 = torch.baddbmm(exp_qk_v_blk_fp32, exp_qk_blk_bf16, v_blk_bf16) 
            # batch add accumulator to batch matrix multiply exp_qk, v_tile

        bwd_normalizing_blk_fp32[bh_tile, q_tile] = qk_max_blk_bf16.squeeze(-1) + torch.log2(exp_qk_sum_blk_fp32).squeeze(-1)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile

        atten_blk_fp32 = exp_qk_v_blk_fp32/ exp_qk_sum_blk_fp32
        atten_out_fp32[bh_tile, q_tile, :] = atten_blk_fp32 #.to(torch.float32)
            
    return atten_out_fp32.view([batch, head, q_tokens, head_dim]) #, bwd_normalizing_blk_fp32
   

helion.kernel(
    static_shapes=True
)
def helion_flash_atten_2_algo_4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,   
    atten: torch.Tensor,
    bwd_normalizing_const: torch.Tensor,
    d_atten: torch.Tensor,
    causal: bool
):
    """
    Computes backward for Attention in Float32 
    Using FlashAtten 2 Algo 4
        Attention = Softmax(Q * K_T/SQRT(d_k)) * V

    Args:
        q: Query shape [batch, head, q_tokens, head_dim]
        k: Key shape [batch, head, k_tokens, head_dim]
        v: Value shape [batch, head, v_tokens, head_dim]
        atten: Atten shape [batch, head, v_tokens, head_dim]
        bwd_normalizing_const: logit of (exp - max_row) / sum_exp, shape [batch*head, q_tokens]

        d_atten: Atten shape [batch, head, q_tokens, head_dim]

    Returns:
    """

    batch, head, q_tokens, q_head_dim = q.shape
    k_batch, k_head, k_tokens, k_head_dim = k.shape
    v_batch, v_head, v_tokens, v_head_dim = k.shape

    head_dim = q_head_dim
    q_bh = q.reshape([-1, q_tokens, head_dim])
    k_bh = k.reshape([-1, k_tokens, head_dim]).transpose(1, 2)
    v_bh = v.reshape([-1, v_tokens, head_dim])
    q_device = q.device
    atten_bh = atten.reshape([-1, q_tokens, head_dim])
    d_atten_bh = d_atten.reshape([-1, q_tokens, head_dim])


    sm_scale =  1.0 / math.sqrt(head_dim) 
    qk_scale = sm_scale * 1.44269504 

    dq_bh = torch.zeros_like(q_bh)
    dk_bh = torch.zeros_like(k_bh)
    dv_bh = torch.zeros_like(v_bh)

    for bh_tile, k_tile in hl.tile([batch*head, k_tokens]):

        k_blk = k_bh[bh_tile, :, k_tile]
        # q_tile.shape = [bh, q, head_dim]

        v_blk = v_bh[bh_tile, k_tile, :]
        # v_tile.shape = [bh, q, head_dim]

        dk_blk = hl.zeros((bh_tile, k_tile, head_dim), device=q_device)
        dv_blk = hl.zeros((bh_tile, k_tile, head_dim), device=q_device)

        for q_tile in hl.tile(q_tokens):

            bwd_normalizing_const_tile = bwd_normalizing_const[bh_tile, q_tile]
            q_blk = q_bh[bh_tile, q_tile, :]

            atten_blk = atten_bh[bh_tile, q_tile, k_tile]
            # q dot k
            qk_blk = torch.bmm(q_blk, k_blk) * qk_scale # batch matrix multiply
            if causal and q_tile.begin < k_tile.end - 1:

                mask_blk = torch.triu(
                    torch.full([q_tile, k_tile], 1.0, dtype=torch.bool), 
                    diagonal=q_tile.begin - k_tile.begin + 1
                    ).to(q_bh.device)

                qk_blk.masked_fill_(mask_blk, float("-inf"))

            exp_qk_blk = torch.exp2(qk_blk) * bwd_normalizing_const_tile 
            # exp_qk.shape = [bh, q, k]

            d_atten_blk = d_atten_bh[bh_tile, q_tile, :]
            # d_atten_blk.shape = [bh, q, head_dim]

            dv_blk += torch.bmm(exp_qk_blk.transpose(-1, -2), d_atten_blk)
            # [bh, k, q] @ [bh, q, head_dim]
            # dv_blk.shape = [bh, k, head_dim]

            dp_blk = torch.bmm(d_atten_blk, v_blk.transpose(-1, -2))
            # [bh, q, head_dim] @ [bh, head_dim, k]
            # dp_blk.shape = [bh, q, k]

            """
            where overflow occurs dO * O sum in bf16
            """
            D_blk = torch.sum(d_atten_blk * atten_blk, dim=-1, keepdims=True)
            # [bh, q, head_dim] elem_wise [bh, q, head_dim]
            # sum on head_dim [bh, q, head_dim]
            # [bh, q, 1]

            d_softmax_blk = qk_blk * (dp_blk - D_blk)
            # [bh, q, k] @ ([bh, q, k] - [bh, q, 1])
            # [bh, q, k]

            dq_bh[bh_tile, q_tile,:] += qk_scale * torch.bmm(d_softmax_blk, k_tile)
            # [bh, q, k] @ [bh, k, head_dim]
            # dq_bh.shape = [bh, q, head_dim]

            dk_blk += qk_scale * torch.bmm(d_softmax_blk.transpose(-1, -2), q_tile)
            # [bh, k, q] @ [bh, q, head]
            # dk_tile.shape = [bh, k, head_dim]

        dk_bh[bh_tile, k_tile, :] = dk_blk
        dv_bh[bh_tile, k_tile, :] = dv_blk

    return dq_bh.view([batch, head, q_tokens, head_dim]), \
        dk_bh.view([batch, head, k_tokens, head_dim]), \
        dv_bh.view([batch, head, v_tokens, head_dim])

helion.kernel(
    static_shapes=True
)
def helion_atten_bwd_bf16_paper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,   
    atten: torch.Tensor,
    bwd_normalizing_const: torch.Tensor,
    d_atten: torch.Tensor,
    causal: bool
):
    """
    Computes backward for Attention in Float32
        Attention = Softmax(Q * K_T/SQRT(d_k)) * V

    Args:
        q: Query shape [batch, head, q_tokens, head_dim]
        k: Key shape [batch, head, k_tokens, head_dim]
        v: Value shape [batch, head, v_tokens, head_dim]
        atten: Atten shape [batch, head, v_tokens, head_dim]
        bwd_normalizing_const: logit of (exp - max_row) / sum_exp, shape [batch*head, q_tokens]

        d_atten: Atten shape [batch, head, q_tokens, head_dim]

    Returns:
    """

    batch, head, q_tokens, q_head_dim = q.shape
    k_batch, k_head, k_tokens, k_head_dim = k.shape
    v_batch, v_head, v_tokens, v_head_dim = k.shape

    head_dim = q_head_dim
    q_bh = q.reshape([-1, q_tokens, head_dim])
    k_bh = k.reshape([-1, k_tokens, head_dim]).transpose(1, 2)
    v_bh = v.reshape([-1, v_tokens, head_dim])

    atten_bh = atten.reshape([-1, q_tokens, head_dim])
    d_atten_bh = d_atten.reshape([-1, q_tokens, head_dim])

    sm_scale =  1.0 / math.sqrt(head_dim) 
    qk_scale = sm_scale * 1.44269504 

    dq_bh = torch.zeros_like(q_bh)
    dk_bh = torch.zeros_like(k_bh)
    dv_bh = torch.zeros_like(v_bh)

    for bh_tile, q_tile in hl.tile([batch*head, q_tokens]):

        q_blk = q_bh[bh_tile, q_tile, :]
        # q_tile.shape = [bh, q, head_dim]

        v_blk = v_bh[bh_tile, q_tile, :]
        # v_tile.shape = [bh, q, head_dim]

        bwd_normalizing_const_tile = bwd_normalizing_const[bh_tile, q_tile]

        for k_tile in hl.tile(k_tokens):
            k_blk = k_bh[bh_tile, :, k_tile]
            atten_blk = atten_bh[bh_tile, q_tile, k_tile]
            # q dot k
            qk_blk = torch.bmm(q_blk, k_blk) * qk_scale # batch matrix multiply

            if causal and q_tile.begin < k_tile.end - 1:

                mask_blk = torch.triu(
                    torch.full([q_tile, k_tile], 1.0, dtype=torch.bool), 
                    diagonal=q_tile.begin - k_tile.begin + 1
                    ).to(q_bh.device)

                qk_blk.masked_fill_(mask_blk, float("-inf"))

            exp_qk_blk = torch.exp2(qk_blk) * bwd_normalizing_const_tile 
            # exp_qk.shape = [bh, q, k]

            d_atten_blk = d_atten_bh[bh_tile, q_tile, :]
            # d_atten_tile.shape = [bh, q, head_dim]

            dv_blk = torch.bmm(exp_qk_blk.transpose(-1, -2), d_atten_blk)
            # [bh, k, q] @ [bh, q, head_dim]
            # dv_tile.shape = [bh, k, head_dim]

            dp_blk = torch.bmm(d_atten_blk, v_blk.transpose(-1, -2))
            # [bh, q, head_dim] @ [bh, head_dim, k]
            # dp_tile.shape = [bh, q, k]

            atten_jvp = torch.sum(d_atten_blk * atten_blk, dim=-1, keepdims=True)
            # [bh, q, head_dim] elem_wise [bh, q, head_dim]
            # sum on head_dim [bh, q, head_dim]
            # [bh, q, 1]

            d_softmax_blk = qk_blk * qk_scale * (dp_blk - atten_jvp)
            # [bh, q, k] @ ([bh, q, k] - [bh, q, 1])
            # [bh, q, k]

            dq_blk = torch.bmm(d_softmax_blk, k_tile)
            # [bh, q, k] @ [bh, k, head_dim]
            # dq_tile.shape = [bh, q, head_dim]

            dk_blk = torch.bmm(d_softmax_blk.transpose(-1, -2), q_tile)
            # [bh, k, q] @ [bh, q, head]
            # dk_tile.shape = [bh, k, head_dim]

            dq_bh[bh_tile, q_tile, :] = dq_blk
            dk_bh[bh_tile, k_tile, :] = dk_blk
            dv_bh[bh_tile, k_tile, :] = dv_blk

    return dq_bh.view([batch, head, q_tokens, head_dim]), \
        dk_bh.view([batch, head, k_tokens, head_dim]), \
        dv_bh.view([batch, head, v_tokens, head_dim])


def test_forward(
    z: int,
    h: int,
    n_ctx: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> None:
    """
    Test the attention kernel implementation against PyTorch's native attention functions.

    Args:
        z: Batch size
        h: Number of attention heads
        n_ctx: Sequence length (context size)
        head_dim: Dimension of each attention head
        dtype: Data type for the tensors
        device: Device to run the test on
    """
    q, k, v = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    def ref_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Reference manual attention implementation"""
        p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        return torch.matmul(p, v)

    flex_compiled = cast(
        "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    )
    baselines = {
        "torch": torch.nn.functional.scaled_dot_product_attention,
        "flex": flex_compiled,
        "ref": ref_attention,
    }

    run_example(atten_fwd_training, baselines, (q, k, v))

    helion_t = atten_fwd_training(q, k, v)
    pytorch_t = ref_attention(q, k, v)
    torch.testing.assert_close(pytorch_t, helion_t, atol=1e-2, rtol=0)

def test_forward_bf16(
    z: int,
    h: int,
    n_ctx: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> None:
    """
    Test the attention kernel implementation against PyTorch's native attention functions.

    Args:
        z: Batch size
        h: Number of attention heads
        n_ctx: Sequence length (context size)
        head_dim: Dimension of each attention head
        dtype: Data type for the tensors
        device: Device to run the test on
    """
    q, k, v = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    def ref_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Reference manual attention implementation"""
        p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        return torch.matmul(p, v)

    # flex_compiled = cast(
    #     "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    # )
    # baselines = {
    #     "torch": torch.nn.functional.scaled_dot_product_attention,
    #     "flex": flex_compiled,
    #     "ref": ref_attention,
    # }

    # run_example(atten_fwd_training, baselines, (q, k, v))

    # helion_t = atten_fwd_training(q, k, v)
    q_fp16 = q.to(torch.float16)
    k_fp16 = q.to(torch.float16)
    v_bf16 = q.to(torch.bfloat16)

    helion_fp32_t = helion_atten_bf16_fwd_training(q_fp16, k_fp16, v_bf16, causal=False)
    pytorch_t = ref_attention(q, k, v)
    mse_diff = torch.nn.functional.mse_loss(pytorch_t, helion_fp32_t)
    print(mse_diff)
    # torch.testing.assert_close(pytorch_t, helion_fp32_t, atol=1e-2, rtol=0)

if __name__ == "__main__":

    # Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    # test_forward(2, 32, 1024, 64, torch.float32, device=DEVICE)

    """
    # nvidia RTX3080 best config
    config=helion.Config(
        block_sizes=[4, 32, 16], 
        indexing=['pointer', 'pointer', 'block_ptr', 'block_ptr', 'pointer'], 
        l2_groupings=[64], 
        load_eviction_policies=['first', 'first', 'last'], 
        loop_orders=[[1, 0]], 
        num_stages=3, 
        num_warps=4, 
        pid_type='flat', 
        range_flattens=[None, None], 
        range_multi_buffers=[None, True], 
        range_num_stages=[0, 0], 
        range_unroll_factors=[0, 0], 
        range_warp_specializes=[]), 

        =================================================================
        Benchmark Results
        =================================================================
        Implementation       Time (ms)    Speedup
        -----------------------------------------------------------------
        helion               0.7240       0.90x
        torch                0.6528       1.00x (ref) # F.scaled_dot_product
        flex                 0.6595       0.99x
        ref                  5.4569       0.12x
        =================================================================

    """


    test_forward_bf16(8, 35, 1024, 64, torch.float32, device=DEVICE)