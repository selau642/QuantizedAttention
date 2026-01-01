"""

Sage Attention 3 paper
int8 forward and backward
for fast fine tuning of transformers on Blackwell

"""

import torch
import helion
import helion.language as hl

from torch.autograd import Function
from typing import Tuple
import math
from helion.autotuner import PowerOfTwoFragment

class SageAttention3_Int8_autograd_function(Function):
    @staticmethod
    def foward(q, k, v):
        k_mean = k.mean(-1)
        k = k - k_mean # k-smoothing
        O, l_bh, \
        sq_bh, sk_bh, sv_bh, \
        Bq, Bkv = helion_atten_int8_linked_fwd(
            q,
            k,
            v
        )

        return O, l_bh, \
            k_mean, \
            sq_bh, sk_bh, sv_bh, \
            Bq, Bkv 

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v = inputs
        O, l_bh, k_mean, sq_bh, sk_bh, sv_bh, Bq, Bkv = output

        ctx.mark_non_differentiable(l_bh, k_mean, sq_bh, sk_bh, sv_bh)
        ctx.save_for_backward(q, k, v, O, l_bh, k_mean, sq_bh, sk_bh, sv_bh)
        ctx.args = (Bq, Bkv)


    @staticmethod
    def backward(ctx, dO, _lse, _sq, _sk, _sv, _Bq, _Bkv):

        q, k, v, O, l_bh, k_mean, sq_bh, sk_bh, sv_bh = ctx.saved_tensors
        Bq, Bkv = ctx.args

        dq, dk, dv = helion_atten_int8_linked_bwd(
            q, sq_bh,
            k, k_mean, sk_bh,
            v, sv_bh,
            O,
            l_bh,
            Bq, Bkv
        )
        return dq, dk, dv

@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_fwd(
    q_int8_input: torch.Tensor,
    sq: torch.Tensor,

    k_int8_input: torch.Tensor,
    sk: torch.Tensor,

    v_int8_input: torch.Tensor,
    sv: torch.Tensor, 

) -> Tuple[torch.Tensor, torch.Tensor]:
 

    batch_head, q_chunk, q_chunk_size, q_head_dim = q_int8_input.shape
    batch_head, k_chunk, k_chunk_size, k_head_dim = k_int8_input.shape
    batch_head, v_chunk, v_chunk_size, v_head_dim = v_int8_input.shape

    assert k_chunk == v_chunk
    assert k_chunk_size == v_chunk_size

    batch_head, q_chunk = sq.shape
    batch_head, k_chunk = sk.shape
    batch_head, v_chunk = sv.shape

    k_int8_input = k_int8_input.transpose(-1, -2, -3)

    q_bh = q_int8_input
    k_bh = k_int8_input
    v_bh = v_int8_input

    cuda_device = q_int8_input.device

    l_bh = torch.zeros(
        [batch_head, q_chunk, q_chunk_size], 
        dtype=torch.float32, 
        device=cuda_device
        )
    # lse = rowsum p

    O_bh = torch.empty_like(
        q_bh, 
        dtype=torch.float32, 
        device=cuda_device
        )

    sm_scale =  1.0 / math.sqrt(q_head_dim) 

    qk_scale = sm_scale * 1.44269504 
 
    for bh_tile, q_chunk_tile in hl.tile([batch_head, q_chunk]):
        O = hl.zeros([bh_tile, q_chunk_tile, q_chunk_size, q_head_dim])
        l = hl.zeros([bh_tile, q_chunk_tile.block_size * q_chunk_size])
        m = hl.zeros([bh_tile, q_chunk_tile.block_size * q_chunk_size])

        q_int8 = q_bh[bh_tile, q_chunk_tile, :, :].reshape(
            -1, 
            q_chunk_tile.block_size * q_chunk_size, 
            q_head_dim
        )    
        # = [bh, q_tile, q_head_dim] 

        for k_chunk_tile in hl.tile([k_chunk]):
            k_int8 = k_bh[bh_tile, :, :, k_chunk_tile].reshape(
                -1, 
                -1, 
                k_chunk_tile.block_size * k_chunk_size
            )     
            # = [bh, k_head_dim, k_tile ] 

            S = torch.bmm(q, k)
            # = [bh, q_tile, k_tile]
            # = [bh, q_chunk*q_chunk_size, k_chunk*k_chunk_size]

            next_m = torch.max(
                m, 
                torch.amax(S, -1, keepdim=True) 
            )

@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_linked_fwd(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    int, int
    ]:
 
    """

    Forward tuning determines the block size Bq and Bkv for backward according to original SageAttention3 paper

    """
    batch, head, q_tokens, q_head_dim = q_input.shape
    batch, head, k_tokens, k_head_dim = k_input.shape
    batch, head, v_tokens, v_head_dim = v_input.shape

    assert k_tokens == v_tokens, "k and v tokens are different"
    assert k_head_dim == v_head_dim, "k head_dim and v head_dim are different"

    q_bh = q_input.reshape([-1, q_tokens, q_head_dim])
    k_bh = k_input.reshape([-1, k_tokens, k_head_dim]).transpose(1, 2)
    v_bh = v_input.reshape([-1, v_tokens, v_head_dim])

    cuda_device = q_input.device

    l_bh = torch.zeros(
        [batch*head, q_tokens], 
        dtype=torch.float32, 
        device=cuda_device
        )
    # lse = rowsum p

    O_bh = torch.empty_like(
        q_bh, 
        dtype=torch.float32, 
        device=cuda_device
        )

    sm_scale =  1.0 / math.sqrt(q_head_dim) 

    qk_scale = sm_scale * 1.44269504 


    split_q = hl.register_tunable("split_q", PowerOfTwoFragment(8, 256, 16))
    Bq = helion.next_power_of_2(helion.cdiv(q_tokens, split_q))
    sq_bh = torch.zeros((batch*head, split_q), device=cuda_device)

    split_kv = hl.register_tunable("split_kv", PowerOfTwoFragment(8, 256, 16))
    Bkv = helion.next_power_of_2(helion.cdiv(k_tokens, split_kv))
    sk_bh = torch.zeros((batch*head, split_kv), device=cuda_device)
    sv_bh = torch.zeros((batch*head, split_kv), device=cuda_device)

    for bh_tile, q_tile in hl.tile([batch*head, q_tokens], block_size=[None, Bq]):

        O = hl.zeros([bh_tile, q_tile, q_head_dim])
        l = hl.zeros([bh_tile, q_tile])
        m = hl.full([bh_tile, q_tile, 1], float("-inf"), dtype=torch.float16)

        q = q_bh[bh_tile, q_tile, :]        

        sq = torch.max(q.abs()) / 127
        q_int8 = q / sq
        q_int8 = q_int8.to(torch.int8)

        sq_bh[bh_tile, q_tile.id] = sq 

        for k_tile in hl.tile([k_tokens], block_size=[Bkv]):
            k = k_bh[bh_tile, :, k_tile]        

            sk = torch.max(k.abs()) / 127
            sk_bh[bh_tile, k_tile.id] = sk
            k_int8 = k / sk
            k_int8 = k_int8.to(torch.int8)

            S = torch.bmm(q_int8, k_int8) * sq * sk * qk_scale             

            row_sum = torch.amax(S, -1, keepdim=True)
            next_m = torch.max(
                m, 
                row_sum
            ).to(torch.float16)

            P = torch.exp2(S - next_m)

            """

            Special sp

            """
            next_l = torch.sum(P, -1, keepdim=True)

            rescale = torch.exp2(m - next_m)
            m = next_m

            l = l * rescale + next_l

            O = O* rescale

            sp = torch.exp2(row_sum - m) / 127
            P_int8 = P / sp
            P_int8 = P.to(torch.int8)

            v = v_bh[bh_tile, k_tile, :]
            sv = torch.max(v.abs()) / 127
            sv_bh[bh_tile, k_tile.id] = sv

            v_int8 = v / sv
            v_int8 = v_int8.to(torch.int8)

            O = torch.baddbmm(O, P_int8, v_int8) * sp * sv

        l_bh[bh_tile, q_tile] = m.squeeze(-1) + torch.log2(l).squeeze(-1)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile
        # usage qk = qk - bwd_normalizing_blk_fp32
        # exp_qk = torch.exp2(qk)

        O_final = O / l
        O_bh[bh_tile, q_tile, :] = O_final

    return O_bh.view([batch, head, q_tokens, q_head_dim]), l_bh, \
        sq_bh, sk_bh, sv_bh, \
        Bq, Bkv


@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_linked_bwd(
    q_input: torch.Tensor,
    sq_bh: torch.Tensor, 

    k_input: torch.Tensor,
    k_mean_input: torch.Tensor,
    sk_bh: torch.Tensor,

    v_input: torch.Tensor,
    sv_bh: torch.Tensor,

    O_input: torch.Tensor,
    lse_input: torch.Tensor,

    Bq: int,
    Bkv: int,

    dO_input: torch.Tensor, 
) -> Tuple[torch.Tensor, torch.Tensor]:

    """

    Online Quantization(different weights for training and backward)

    """

    batch, head, q_tokens, q_head_dim = q_input.shape
    k_batch, k_head, k_tokens, k_head_dim = k_input.shape
    k_batch, k_head, k_tokens = k_mean_input.shape
    v_batch, v_head, v_tokens, v_head_dim = v_input.shape

    head_dim = q_head_dim

    q_bh = q_input.reshape([-1, q_tokens, head_dim])
    k_bh = k_input.reshape([-1, k_tokens, head_dim]).transpose(-1, -2)
    k_mean_bh = k_mean_input.reshape([-1, k_tokens])

    v_bh = v_input.reshape([-1, v_tokens, head_dim])
    O_bh = O_input.reshape([-1, q_tokens, head_dim])
    dO_bh = dO_input.reshape([-1, q_tokens, head_dim])

    sm_scale =  1.0 / math.sqrt(head_dim) 
    qk_scale = sm_scale * 1.44269504 

    cuda_device = q_input.device

    dq_bh = torch.zeros_like(q_bh)
    dk_bh = torch.zeros([batch*head, k_tokens, head_dim], device=cuda_device) # not transposed
    dv_bh = torch.zeros_like(v_bh)

    for bh_tile, k_tile in hl.tile([batch*head, k_tokens], block_size=[None, Bkv]):

        k = k_bh[bh_tile, :, k_tile]
        v = v_bh[bh_tile, k_tile, :]

        sk = sk_bh[bh_tile, k_tile.id]
        k_int8 = k / sk 
        k_int8 = k_int8.to(torch.int8)

        sv = sv_bh[bh_tile, k_tile.id]
        v_int8 = v / sv 
        v_int8 = v_int8.to(torch.int8)

        dk = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float32)
        dv = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float32)

        for q_tile in hl.tile(q_tokens, block_size=[Bq]):

            q = q_bh[bh_tile, q_tile, :]
            sq = sq_bh[bh_tile, q_tile.id]
            q_int8 = q / sq
            q_int8 = q_int8.to(torch.int8)

            S = torch.bmm(q_int8, k_int8) * sq * sv * qk_scale  # batch matrix multiply

            l = lse_input[bh_tile, q_tile]
            P = torch.exp2(S - l[:, :, None])

            # P.shape = [bh, q, k]
            sP = torch.max(P.abs()) / 127
            P_int8 = P / sP
            P_int8 = P_int8.to(torch.int8)

            P_int8_T = torch.transpose(P_int8, 1, 2)

            dO = dO_bh[bh_tile, q_tile, :]
            # dO.shape = [bh, q, head_dim]
            s_dO = torch.max(dO.abs()) / 127
            dO_int8 = dO / s_dO
            dO_int8 = dO_int8.to(torch.int8)

            dv = torch.baddbmm(dv, P_int8_T, dO_int8) * sP * s_dO

            # dv.shape 
            # [bh, k, head_dim] = [bh, k, q] @ [bh, q, head_dim]


            v_T = torch.transpose(v, 1, 2)
            dP_fp16 = torch.bmm(dO, v_T) 
            # [bh, q, head_dim] @ [bh, head_dim, k]
            # dP.shape = [bh, q, k]

            O = O_bh[bh_tile, q_tile, :]

            """

            D is where overflow occurs dO * O sum in bf16 

            """
            D = torch.sum(dO * O, dim=-1, keepdims=True)
            # [bh, q, head_dim] elem_wise [bh, q, head_dim]
            # sum on head_dim [bh, q, head_dim]
            # [bh, q, 1]

            dS = S * (dP_fp16 - D)

            s_dS = torch.max(dS.abs()) / 127
            dS_int8 = dS / s_dS
            dS_int8 = dS_int8.to(torch.int8)
            # [bh, q, k] * [bh, q, k] 

            k_int8_T = k_int8.transpose(-1,-2)

            dS_dot_k = torch.bmm(dS_int8, k_int8_T) * qk_scale * s_dS * sk
            k_mean = k_mean_bh[bh_tile, k_tile]
            dS_k_mean = torch.sum(dS, dim=0) * k_mean

            dq_bh[bh_tile, q_tile, :]  = dq_bh[bh_tile, q_tile, :]  + dS_dot_k + dS_k_mean

            # dq_bh.shape 
            # [bh, q, head_dim] = [bh, q, k] @ [bh, k, head_dim]
            dS_int8_T = dS_int8.transpose(-1, -2)
            
            dk = dk +  torch.bmm(dS_int8_T, q_int8) * s_dS * sq * qk_scale

        dk_bh[bh_tile, k_tile, :] = dk
        dv_bh[bh_tile, k_tile, :] = dv

    return dq_bh.view([batch, head, q_tokens, head_dim]), \
        dk_bh.view([batch, head, k_tokens, head_dim]), \
        dv_bh.view([batch, head, v_tokens, head_dim])

@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_unlinked_fwd(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
 
    """

    Online quantization and discard of quantized values

    """
    batch, head, q_tokens, q_head_dim = q_input.shape
    batch, head, k_tokens, k_head_dim = k_input.shape
    batch, head, v_tokens, v_head_dim = v_input.shape

    assert k_tokens == v_tokens, "k and v tokens are different"
    assert k_head_dim == v_head_dim, "k head_dim and v head_dim are different"

    q_bh = q_input.reshape([-1, q_tokens, q_head_dim])
    k_bh = k_input.reshape([-1, k_tokens, k_head_dim]).transpose(1, 2)
    v_bh = v_input.reshape([-1, v_tokens, v_head_dim])

    cuda_device = q_input.device

    l_bh = torch.zeros(
        [batch*head, q_tokens], 
        dtype=torch.float32, 
        device=cuda_device
        )
    # lse = rowsum p

    O_bh = torch.empty_like(
        q_bh, 
        dtype=torch.float32, 
        device=cuda_device
        )

    sm_scale =  1.0 / math.sqrt(q_head_dim) 

    qk_scale = sm_scale * 1.44269504 
 
    for bh_tile, q_tile in hl.tile([batch*head, q_tokens]):

        O = hl.zeros([bh_tile, q_tile, q_head_dim])
        l = hl.zeros([bh_tile, q_tile])
        m = hl.full([bh_tile, q_tile, 1], float("-inf"), dtype=torch.float16)

        q = q_bh[bh_tile, q_tile, :]        

        sq = torch.max(q.abs()) / 127
        q_int8 = q / sq
        q_int8 = q_int8.to(torch.int8)

        for k_tile in hl.tile([k_tokens]):
            k = k_bh[bh_tile, :, k_tile]        

            sk = torch.max(k.abs()) / 127
            k_int8 = k / sk
            k_int8 = k_int8.to(torch.int8)

            S = torch.bmm(q_int8, k_int8) * sq * sk * qk_scale             

            row_sum = torch.amax(S, -1, keepdim=True)
            next_m = torch.max(
                m, 
                row_sum
            ).to(torch.float16)

            P = torch.exp2(S - next_m)

            """

            Special sp

            """
            next_l = torch.sum(P, -1, keepdim=True)

            rescale = torch.exp2(m - next_m)
            m = next_m

            l = l * rescale + next_l

            O = O* rescale

            sp = torch.exp2(row_sum - m) / 127
            P_int8 = P / sp
            P_int8 = P.to(torch.int8)

            v = v_bh[bh_tile, k_tile, :]
            sv = torch.max(v.abs()) / 127
            v_int8 = v / sv
            v_int8 = v_int8.to(torch.int8)

            O = torch.baddbmm(O, P_int8, v_int8) * sp * sv

        l_bh[bh_tile, q_tile] = m.squeeze(-1) + torch.log2(l).squeeze(-1)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile
        # usage qk = qk - bwd_normalizing_blk_fp32
        # exp_qk = torch.exp2(qk)

        O_final = O / l
        O_bh[bh_tile, q_tile, :] = O_final

    return O_bh.view([batch, head, q_tokens, q_head_dim]), \
        l_bh


@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_unlinked_bwd(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    k_mean_input: torch.Tensor,

    v_input: torch.Tensor,
    O_input: torch.Tensor,
    lse_input: torch.Tensor,

    dO_input: torch.Tensor, 
) -> Tuple[torch.Tensor, torch.Tensor]:

    """

    Online Quantization(different weights for training and backward)

    """

    batch, head, q_tokens, q_head_dim = q_input.shape
    k_batch, k_head, k_tokens, k_head_dim = k_input.shape
    k_batch, k_head, k_tokens = k_mean_input.shape
    v_batch, v_head, v_tokens, v_head_dim = v_input.shape

    head_dim = q_head_dim

    q_bh = q_input.reshape([-1, q_tokens, head_dim])
    k_bh = k_input.reshape([-1, k_tokens, head_dim]).transpose(-1, -2)
    k_mean_bh = k_mean_input.reshape([-1, k_tokens])

    v_bh = v_input.reshape([-1, v_tokens, head_dim])
    O_bh = O_input.reshape([-1, q_tokens, head_dim])
    dO_bh = dO_input.reshape([-1, q_tokens, head_dim])

    sm_scale =  1.0 / math.sqrt(head_dim) 
    qk_scale = sm_scale * 1.44269504 

    cuda_device = q_input.device

    dq_bh = torch.zeros_like(q_bh)
    dk_bh = torch.zeros([batch*head, k_tokens, head_dim], device=cuda_device) # not transposed
    dv_bh = torch.zeros_like(v_bh)

    for bh_tile, k_tile in hl.tile([batch*head, k_tokens]):

        k = k_bh[bh_tile, :, k_tile]
        v = v_bh[bh_tile, k_tile, :]

        sk = torch.max(k.abs()) / 127
        k_int8 = k / sk
        k_int8 = k_int8.to(torch.int8)

        sv = torch.max(v.abs()) / 127
        v_int8 = v / sv
        v_int8 = v_int8.to(torch.int8)

        dk = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float32)
        dv = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float32)

        for q_tile in hl.tile(q_tokens):

            q = q_bh[bh_tile, q_tile, :]
            sq = torch.max(q.abs()) / 127
            q_int8 = q / sq
            q_int8 = q_int8.to(torch.int8)

            S = torch.bmm(q_int8, k_int8) * sq * sv * qk_scale  # batch matrix multiply

            l = lse_input[bh_tile, q_tile]
            P = torch.exp2(S - l[:, :, None])

            # P.shape = [bh, q, k]
            sP = torch.max(P.abs()) / 127
            P_int8 = P / sP
            P_int8 = P_int8.to(torch.int8)

            P_int8_T = torch.transpose(P_int8, 1, 2)

            dO = dO_bh[bh_tile, q_tile, :]
            # dO.shape = [bh, q, head_dim]
            s_dO = torch.max(dO.abs()) / 127
            dO_int8 = dO / s_dO
            dO_int8 = dO_int8.to(torch.int8)

            dv = torch.baddbmm(dv, P_int8_T, dO_int8) * sP * s_dO

            # dv.shape 
            # [bh, k, head_dim] = [bh, k, q] @ [bh, q, head_dim]


            v_T = torch.transpose(v, 1, 2)
            dP_fp16 = torch.bmm(dO, v_T) 
            # [bh, q, head_dim] @ [bh, head_dim, k]
            # dP.shape = [bh, q, k]

            O = O_bh[bh_tile, q_tile, :]

            """

            D is where overflow occurs dO * O sum in bf16 

            """
            D = torch.sum(dO * O, dim=-1, keepdims=True)
            # [bh, q, head_dim] elem_wise [bh, q, head_dim]
            # sum on head_dim [bh, q, head_dim]
            # [bh, q, 1]

            dS = S * (dP_fp16 - D)

            s_dS = torch.max(dS.abs()) / 127
            dS_int8 = dS / s_dS
            dS_int8 = dS_int8.to(torch.int8)
            # [bh, q, k] * [bh, q, k] 

            k_int8_T = k_int8.transpose(-1,-2)

            dS_dot_k = torch.bmm(dS_int8, k_int8_T) * qk_scale * s_dS * sk
            k_mean = k_mean_bh[bh_tile, k_tile]
            dS_k_mean = torch.sum(dS, dim=0) * k_mean

            dq_bh[bh_tile, q_tile, :]  = dq_bh[bh_tile, q_tile, :]  + dS_dot_k + dS_k_mean

            # dq_bh.shape 
            # [bh, q, head_dim] = [bh, q, k] @ [bh, k, head_dim]
            dS_int8_T = dS_int8.transpose(-1, -2)
            
            dk = dk +  torch.bmm(dS_int8_T, q_int8) * s_dS * sq * qk_scale

        dk_bh[bh_tile, k_tile, :] = dk
        dv_bh[bh_tile, k_tile, :] = dv

    return dq_bh.view([batch, head, q_tokens, head_dim]), \
        dk_bh.view([batch, head, k_tokens, head_dim]), \
        dv_bh.view([batch, head, v_tokens, head_dim])


    



    


