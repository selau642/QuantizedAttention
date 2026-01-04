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
from torch.nn.functional import mse_loss

class SageAttention3_Int8_autograd_function(Function):
    @staticmethod
    def forward(q_fp16, k_fp16, v_fp16):
        k_mean = k_fp16.mean(-1)
        k_fp16_minus_mean = k_fp16 - k_mean[:, :, :, None] # k-smoothing

        O, l_bh, \
        q_bh_int8, k_bh_int8, v_bh_int8, \
        sq_bh, sk_bh, sv_bh, \
        Bq, Bkv = helion_atten_int8_hl_dot_flat_fwd(
            q_fp16,
            k_fp16_minus_mean,
            v_fp16
        )

        return O, l_bh, \
            k_mean, \
            q_bh_int8, k_bh_int8, v_bh_int8, \
            sq_bh, sk_bh, sv_bh, \
            Bq, Bkv 

    @staticmethod
    def setup_context(ctx, inputs, output):
        q_fp16, k_fp16, v_fp16 = inputs

        O_fp16, l_bh, \
        k_mean, \
        q_bh_int8, k_bh_int8, v_bh_int8, \
        sq_bh, sk_bh, sv_bh, \
        Bq, Bkv, Bd = output

        ctx.mark_non_differentiable(l_bh, k_mean, sq_bh, sk_bh, sv_bh)
        ctx.save_for_backward(
            O_fp16, l_bh,
            k_mean,

            q_bh_int8, k_bh_int8, v_bh_int8, 
            sq_bh, sk_bh, sv_bh
            )
        ctx.args = (Bq, Bkv, Bd)


    @staticmethod
    def backward(ctx, 
        dO, _lse, 
        _k_mean, 
        _q_bh_int8, _k_bh_int8, _v_bh_int8,
        _sq, _sk, _sv, 
        _Bq, _Bkv, _Bd):

        O_fp16, l_bh, \
        k_mean, \
        q_bh_int8, k_bh_int8, v_bh_int8, \
        sq_bh, sk_bh, sv_bh = ctx.saved_tensors

        Bq, Bkv, Bd = ctx.args

        dq, dk, dv = helion_atten_int8_hl_dot_bwd(

            q_bh_int8, sq_bh,
            k_bh_int8, k_mean, sk_bh,
            v_bh_int8, sv_bh,

            O_fp16,
            l_bh,

            Bq, Bkv, Bd,

            dO
        )

        return dq, dk, dv

@helion.kernel(
    autotune_effort="none",
    static_shapes=True,
)
def helion_atten_int8_hl_dot_flat_fwd(
    q_fp16_input: torch.Tensor,
    k_fp16_input: torch.Tensor,
    v_fp16_input: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    int, int
    ]:
 

    """


    Forward tuning determines the block size Bq and Bkv 
    for backward according to original SageAttention3 paper


    """

    batch, head, q_tokens, q_head_dim = q_fp16_input.shape
    batch, head, k_tokens, k_head_dim = k_fp16_input.shape
    batch, head, v_tokens, v_head_dim = v_fp16_input.shape

    assert k_tokens == v_tokens, "k and v tokens are different"
    assert k_head_dim == v_head_dim, "k head_dim and v head_dim are different"

    q_bh = q_fp16_input.reshape([-1, q_head_dim])
    k_bh = k_fp16_input.reshape([-1, k_head_dim]).transpose(0, 1)
    v_bh = v_fp16_input.reshape([-1, v_head_dim])

    cuda_device = q_fp16_input.device
    total_q_tokens = batch * head * q_tokens
    total_kv_tokens = batch * head * k_tokens
    
    l_bh_fp16 = torch.zeros(
        [total_q_tokens], 
        dtype=torch.float16, 
        device=cuda_device
        )

    # lse = rowsum p
    
    O_bh_fp16 = torch.empty_like(
        v_bh, 
        dtype=torch.float16, 
        device=cuda_device
        )

    sm_scale = 1.0 / math.sqrt(q_head_dim) 

    qk_scale = sm_scale * 1.44269504 

    Bq = hl.register_tunable("Bq", PowerOfTwoFragment(32, 256, 32))
    split_q = helion.cdiv(total_q_tokens, Bq)
    sq_bh = torch.zeros((split_q), device=cuda_device)

    q_int8_bh = torch.zeros((total_q_tokens, q_head_dim), device=cuda_device)

    Bkv = hl.register_tunable("Bkv", PowerOfTwoFragment(32, 256, 32))
    split_kv = helion.cdiv(total_kv_tokens, Bkv)

    sk_bh = torch.zeros((split_kv), device=cuda_device)
    k_int8_bh_T = torch.zeros((k_head_dim, total_kv_tokens), device=cuda_device)

    sv_bh = torch.zeros((split_kv), device=cuda_device)
    v_int8_bh = torch.zeros((total_kv_tokens, v_head_dim), device=cuda_device)

    needle = torch.zeros(
        (1), 
        device=cuda_device
        )

    catch_q_dot_k = torch.zeros(
        (Bq, Bkv),
        device=cuda_device
    )
    
    catch_sq = torch.zeros(
        (1),
        device=cuda_device
    )
    catch_sk = torch.zeros(
        (1),
        device=cuda_device
    )

    catch_S = torch.zeros(
        (Bq, Bkv),
        device=cuda_device
    )
    catch_next_m = torch.zeros(
        (Bq, 1),
        device=cuda_device
    )

    catch_row_max_S = torch.zeros(
        (Bq, 1),
        device=cuda_device
    )


    for q_tile in hl.tile(total_q_tokens, block_size=Bq):

        O_fp32 = hl.zeros((q_tile, v_head_dim), dtype=torch.float32, device=cuda_device)
        l_fp32 = hl.full((q_tile, 1), 1.0, dtype=torch.float32, device=cuda_device)
        m_fp16 = hl.full((q_tile, 1), float("-inf"), dtype=torch.float16, device=cuda_device)

        for k_tile in hl.tile(total_kv_tokens, block_size=Bkv):

            q = q_bh[q_tile,:]        

            max_q_tokens = torch.amax(q.abs(), dim=0, keepdim=False) 
            sq = torch.amax(max_q_tokens, dim=0, keepdim=False) / 127

            q_int8 = q / sq
            q_int8 = q_int8.to(torch.int8)

            sq_bh[q_tile.id] = sq 
            q_int8_bh[q_tile, :] = q_int8

            k = k_bh[:, k_tile]        

            max_k_tokens = torch.amax(k.abs(), dim=0, keepdim=False) 
            sk = torch.amax(max_k_tokens, dim=0, keepdim=False) / 127
            sk_bh[k_tile.id] = sk

            k_int8 = k / sk
            k_int8 = k_int8.to(torch.int8)
            k_int8_bh_T[:, k_tile] = k_int8
            q_dot_k = hl.dot(q_int8, k_int8).to(torch.float32) * sq * sk * qk_scale
            S_fp16 = q_dot_k.to(torch.float16)

            row_max_S = torch.amax(S_fp16, -1, keepdim=True)
            next_m_fp16 = torch.max(
                m_fp16, 
                row_max_S 
            )
            # next_m_fp16 = [q_tile, 1]
            c1 = torch.sum(torch.isinf(row_max_S.to(torch.float32)), dim=0)
            c2 = torch.sum(c1, dim=0)
            if  c2 > 0:
                needle[:] = 1
                catch_q_dot_k[:, :] = q_dot_k
                catch_sk[:] = sk
                catch_sq[:] = sq
                catch_S[:, :] = S_fp16
                catch_next_m[:, :] = next_m_fp16
                catch_row_max_S[:, :] = row_max_S

            P_fp32 = torch.exp2(
                (S_fp16  - next_m_fp16).to(torch.float32)
            ).to(torch.float32)

            next_l_fp32 = torch.sum(P_fp32, -1, keepdim=True)
            # next_l_fp16.shape = [q_tile, 1] 
            rescale_fp32 = torch.exp2(
                (m_fp16 - next_m_fp16).to(torch.float32)
            ).to(torch.float32)

            m_fp16 = next_m_fp16

            l_fp32 = l_fp32 * rescale_fp32  + next_l_fp32

            O_fp32 = O_fp32 * rescale_fp32

            """

            Per q Token Quantization of P

            """
            sp = torch.exp2(
                (row_max_S - m_fp16).to(torch.float32)
                ) / 127
            # sp.shape = [q_tile, 1]
            P_int8 = P_fp32 / sp
            P_int8 = P_int8.to(torch.int8)
            # [q_tile, k_tile]
            # O_fp16 = P_int8 @ v_int8
            # [q_tile, head_dim] = [q_tile, k_tile] @ [k_tile, head_dim]
            v = v_bh[k_tile, :]
            max_v_tokens = torch.amax(v.abs(), dim=0, keepdim=False) 
            sv = torch.amax(max_v_tokens, dim=0, keepdim=False) / 127
            sv_bh[k_tile.id] = sv

            v_int8 = v / sv
            v_int8 = v_int8.to(torch.int8)
            v_int8_bh[k_tile, :] = v_int8

            P_dot_v_int32 = hl.dot(P_int8, v_int8) * sp * sv
            O_fp32 += P_dot_v_int32.to(torch.float32)

        l_bh_fp16[q_tile] = m_fp16.squeeze(-1) + torch.log2(l_fp32).squeeze(-1).to(torch.float16)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile
        # usage qk = qk - bwd_normalizing_blk_fp32
        # exp_qk = torch.exp2(qk)
        O_final = O_fp32 / l_fp32
        O_bh_fp16[q_tile, :] = O_final.to(torch.float16)

    return O_bh_fp16.view([batch, head, q_tokens, q_head_dim]), l_bh_fp16, \
        q_int8_bh, k_int8_bh_T, v_int8_bh, \
        sq_bh, sk_bh, sv_bh, \
        Bq, Bkv, \
        needle, catch_S, catch_q_dot_k, \
        catch_next_m, catch_row_max_S, \
        catch_sq, catch_sk, 

@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_hl_dot_fwd(
    q_fp16_input: torch.Tensor,
    k_fp16_input: torch.Tensor,
    v_fp16_input: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    int, int
    ]:
 

    """


    Forward tuning determines the block size Bq and Bkv 
    for backward according to original SageAttention3 paper

    split head_dim into (split_d) dims 
    to do small tile by tile mma in int8 x int8
    using hl.dot()

    NotImplemented multi level reductions


    """

    batch, head, q_tokens, q_head_dim = q_fp16_input.shape
    batch, head, k_tokens, k_head_dim = k_fp16_input.shape
    batch, head, v_tokens, v_head_dim = v_fp16_input.shape

    assert k_tokens == v_tokens, "k and v tokens are different"
    assert k_head_dim == v_head_dim, "k head_dim and v head_dim are different"
    q_head_dim = hl.specialize(q_head_dim)
    v_head_dim = hl.specialize(v_head_dim)

    q_bh = q_fp16_input.reshape([-1, q_head_dim])
    k_bh = k_fp16_input.reshape([-1, k_head_dim]).transpose(0, 1)
    v_bh = v_fp16_input.reshape([-1, v_head_dim])

    cuda_device = q_fp16_input.device
    total_q_tokens = batch * head * q_tokens

    l_bh_fp16 = torch.zeros(
        [total_q_tokens], 
        dtype=torch.float16, 
        device=cuda_device
        )
    # lse = rowsum p

    O_bh_fp16 = torch.empty_like(
        v_bh, 
        dtype=torch.float16, 
        device=cuda_device
        )

    sm_scale = 1.0 / math.sqrt(q_head_dim) 

    qk_scale = sm_scale * 1.44269504 

    B_qd = hl.register_tunable("B_qd", PowerOfTwoFragment(8, 128, 16))
    split_q_d = helion.cdiv(q_head_dim, B_qd)

    B_vd = hl.register_tunable("B_vd", PowerOfTwoFragment(8, 128, 16))
    split_v_d = helion.cdiv(v_head_dim, B_vd)

    Bq = hl.register_tunable("Bq", PowerOfTwoFragment(8, 256, 16))
    split_q = helion.cdiv(total_q_tokens, Bq)

    sq_bh = torch.zeros((split_q, split_q_d), device=cuda_device)
    q_int8_bh = torch.zeros((total_q_tokens, q_head_dim))
    total_kv_tokens = batch * head * k_tokens

    Bkv = hl.register_tunable("Bkv", PowerOfTwoFragment(8, 256, 16))
    split_kv = helion.cdiv(total_kv_tokens, Bkv)

    sk_bh = torch.zeros((split_kv, split_q_d), device=cuda_device)
    k_int8_bh_T = torch.zeros((k_head_dim, total_kv_tokens))

    sv_bh = torch.zeros((split_kv, split_v_d), device=cuda_device)
    v_int8_bh = torch.zeros((total_kv_tokens, v_head_dim))


    for q_tile in hl.tile(total_q_tokens, block_size=Bq):

        O_fp16 = hl.zeros((q_tile, v_head_dim), dtype=torch.float16, device=cuda_device)
        l_fp16 = hl.zeros((q_tile, 1), dtype=torch.float16, device=cuda_device)
        m_fp16 = hl.full((q_tile, 1), float("-inf"), dtype=torch.float16, device=cuda_device)

        for k_tile in hl.tile(total_kv_tokens, block_size=Bkv):

            S_fp16 = hl.zeros((q_tile, k_tile), dtype=torch.float16, device=cuda_device)

            for q_d_tile in hl.tile(q_head_dim, block_size=B_qd):

                q = q_bh[q_tile, q_d_tile]        
                sq = torch.max(q.abs()) / 127
                q_int8 = q / sq
                q_int8 = q_int8.to(torch.int8)

                sq_bh[q_tile.id, q_d_tile.id] = sq 
                q_int8_bh[q_tile, q_d_tile] = q_int8

                k = k_bh[q_d_tile, k_tile]        

                sk = torch.max(k.abs()) / 127
                sk_bh[k_tile.id, q_d_tile.id] = sk

                k_int8 = k / sk
                k_int8 = k_int8.to(torch.int8)
                k_int8_bh_T[q_d_tile, k_tile] = k_int8

                q_dot_k = hl.dot(q_int8, k_int8)
                q_dot_k_fp16 = q_dot_k.to(torch.float16) * sq * sk * qk_scale
                S_fp16 += q_dot_k_fp16           

            # S_fp16 = S_int32.to(torch.float16)

            row_max_S_fp16 = torch.amax(S_fp16, -1, keepdim=True)
            next_m_fp16 = torch.max(
                m_fp16, 
                row_max_S_fp16 
            )
            # next_m_fp16 = [q_tile, 1]

            P_fp16 = torch.exp2(S_fp16 - next_m_fp16)

            next_l_fp16 = torch.sum(P_fp16, -1, keepdim=True)
            # next_l_fp16.shape = [q_tile, 1] 
            rescale_fp16 = torch.exp2(m_fp16 - next_m_fp16)

            m_fp16 = next_m_fp16

            l_fp16 = l_fp16 * rescale_fp16 + next_l_fp16

            O_fp16 = O_fp16 * rescale_fp16

            """

            Per q Token Quantization of P

            """
            sp = torch.exp2(row_max_S_fp16 - m_fp16) / 127
            # sp.shape = [q_tile, 1]
            P_int8 = P_fp16 / sp
            P_int8 = P_int8.to(torch.int8)
            # [q_tile, k_tile]
            # O_fp16 = P_int8 @ v_int8

            # [q_tile, head_dim] = [q_tile, k_tile] @ [k_tile, head_dim]
            for v_d_tile in hl.tile(v_head_dim, block_size=B_vd):
                v = v_bh[k_tile, v_d_tile]
                sv = torch.max(v.abs()) / 127
                sv_bh[k_tile.id, v_d_tile.id] = sv

                v_int8 = v / sv
                v_int8 = v_int8.to(torch.int8)
                v_int8_bh[k_tile, v_d_tile] = v_int8

                P_dot_v_int32 = hl.dot(P_int8, v_int8) * sp * sv
                O_fp16[q_tile, v_d_tile] =  P_dot_v_int32.to(torch.float16)

        l_bh_fp16[q_tile] = m_fp16.squeeze(-1) + torch.log2(l_fp16).squeeze(-1)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile
        # usage qk = qk - bwd_normalizing_blk_fp32
        # exp_qk = torch.exp2(qk)

        O_final = O_fp16 / l_fp16
        O_bh_fp16[q_tile, :] = O_final

    return O_bh_fp16.view([batch, head, q_tokens, q_head_dim]), l_bh_fp16, \
        q_int8_bh, k_int8_bh_T, v_int8_bh, \
        sq_bh, sk_bh, sv_bh, \
        Bq, Bkv, B_qd, B_vd

@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_hl_dot_bwd(
    q_bh_int8: torch.Tensor,
    sq_bh: torch.Tensor, 

    k_bh_int8_T: torch.Tensor,
    k_mean_bh: torch.Tensor,
    sk_bh: torch.Tensor,

    v_bh_int8: torch.Tensor,
    sv_bh: torch.Tensor,

    O_fp16_input: torch.Tensor,
    lse_input: torch.Tensor,

    Bq: int,
    Bkv: int,
    Bd: int,

    dO_input: torch.Tensor, 
) -> Tuple[torch.Tensor, torch.Tensor]:

    """

    Online Quantization(different weights for training and backward)

    """

    batch_head_q_tokens, q_head_dim = q_bh_int8.shape
    k_head_dim , batch_head_k_tokens= k_bh_int8_T.shape
    batch_head_k_tokens = k_mean_bh.shape
    batch_head_v_tokens, v_head_dim = v_bh_int8.shape
    batch, head, q_tokens, v_head_dim =  O_fp16_input.shape

    head_dim = q_head_dim

    O_bh = O_fp16_input.reshape([-1, head_dim])
    dO_bh = dO_input.reshape([-1, head_dim])

    sm_scale =  1.0 / math.sqrt(head_dim) 
    qk_scale = sm_scale * 1.44269504 

    cuda_device = q_bh_int8.device

    dq_bh = torch.zeros_like(
        [batch_head_q_tokens, head_dim], 
        dtype=torch.float16
        )

    dk_bh = torch.zeros(
        [batch_head_k_tokens, head_dim], 
        device=cuda_device, 
        dtype=torch.float16) # not transposed

    dv_bh = torch.zeros_like(
        [batch_head_v_tokens, head_dim], 
        dtype=torch.float16
    )

    for k_tile in hl.tile([batch_head_k_tokens], block_size=[Bkv]):

        dk = hl.zeros((k_tile, head_dim), device=cuda_device, dtype=torch.float16)
        dv = hl.zeros((k_tile, head_dim), device=cuda_device, dtype=torch.float16)

        for q_tile in hl.tile([batch_head_k_tokens], block_size=[Bq]):

            S_int32 = hl.zeros((q_tile, k_tile), device=cuda_device, dtype=torch.int32)


            """

            head_dim loop for:
                S = Q @ K_t * qk_scale
                [q_tile, k_tile] = [q_tile, head_dim] @ [head_dim, k_tile]

            """
            for d_tile in hl.tile([q_head_dim], block_size=[Bd]):
                k_int8_T = k_bh_int8_T[k_tile, d_tile]
                sk = sk_bh[k_tile.id, d_tile.id]

                q_int8 = q_bh_int8[q_tile, d_tile]
                sq = sq_bh[q_tile.id, d_tile.id]

                S_int32[q_tile, k_tile] += hl.dot(q_int8, k_int8_T) * sq * sk  # batch matrix multiply

            S_fp16 = S_int32.to(torch.float16) * qk_scale
            # S_fp16.shape = [q_tile, k_tile]



            l = lse_input[q_tile]
            # l.shape = [q_tile] per row normalization const            
            P_fp16 = torch.exp2(S_fp16 - l[:, None])
            # P_fp16.shape = [bh, q, k]

            sP = torch.max(P_fp16.abs()) / 127
            P_int8 = P_fp16 / sP
            P_int8 = P_int8.to(torch.int8)

            P_int8_T = torch.transpose(P_int8, 1, 2)

            dO = dO_bh[q_tile, :]
            # dO.shape = [bh_q, head_dim]

            s_dO = torch.max(dO.abs()) / 127
            dO_int8 = dO / s_dO
            dO_int8 = dO_int8.to(torch.int8)

            """
            
            head_dim loop for:
                dv = P @ dO
                [bh_k, head_dim] = [bh_k, bh_q] @ [bh_q, head_dim]

            """
            for d_tile in hl.tile([q_head_dim], block_size=[Bd]):
                dO_int8_sub_d = dO_int8[q_tile, d_tile]
                dv[:, d_tile] = hl.dot(P_int8_T, dO_int8_sub_d) * sP * s_dO

            """
            
            head_dim loop for:
                dP = dO @ v_T
                [bh_q, bh_k] = [bh_q, head_dim] @ [head_dim, bh_k]

            """
            dP_fp16 = hl.zeros([q_tile, k_tile], dtype=torch.float16, device=cuda_device)
            for d_tile in hl.tile([q_head_dim], block_size=[Bd]): 
                dO_int8_sub_d = dO_int8[:, d_tile]
                v_int8_sub_d_T = v_bh_int8[k_tile, d_tile].transpose(-1, -2)
                sv = sv_bh[k_tile.id, d_tile.id]
                dP_fp16 += hl.dot(dO_int8_sub_d, v_int8_sub_d_T, out_dtype=torch.float16) * s_dO * sv   

            """

            D is where overflow occurs dO * O sum in bf16 
                D = rowsum(dO * O) 
                = [bh_q, head_dim] * [bh_q, head_dim]
                = sum over head_dim [bh, q, head_dim]
                = [bh, q, 1]

            """
            O_fp16 = O_bh[q_tile, :]
            D_fp16 = torch.sum(dO * O_fp16, dim=-1, keepdims=True)
            
            dS_fp16 = S_fp16 * (dP_fp16 - D_fp16)

            s_dS = torch.max(dS_fp16.abs()) / 127
            dS_int8 = dS_fp16 / s_dS
            dS_int8 = dS_int8.to(torch.int8)
            # [q_tile, k_tile]

            k_mean = k_mean_bh[k_tile]
            dS_k_mean = torch.sum(dS_fp16, dim=-1) * k_mean
            # dS_k_mean = sum over k [q_tile, k_tile] = [q_tile] * [1]

            """ 

            head_dim loop 
                dq = dq + dS @ k * qk_scale + dS_row_sum * k_mean 

                dq [bh_q, head_dim] = dq [bh_q, head_dim] 
                    + (dS @ k) [bh_q, bh_k] @ [bh_k, head_dim] 
                    + (dS_row_sum * k_mean) [bh_q] * k_mean


            """
            # dq_bh[q_tile, :]  = dq_bh[q_tile, :]  + dS_dot_k_fp16 + dS_k_mean
            for d_tile in hl.tile([head_dim], block_size=[Bd]):
                k_int8 = k_bh_int8_T[d_tile, k_tile].transpose(-1, -2)
                sk = sk_bh[k_tile, d_tile]

                # torch.bmm(dS_int8, k_int8_T) * qk_scale * s_dS * sk
                dS_dot_k_int32 = hl.dot(dS_int8, k_int8) * s_dS * sk
                dS_dot_k_fp16 = dS_dot_k_int32.to(torch.float16) * qk_scale

                dq_bh[q_tile, d_tile] += dS_dot_k_fp16 + dS_k_mean[:, None]                

            """

            head_dim loop
                dk = dS_T @ q * qk_scale
                [k_tile, head_dim] = [k_tile, q_tile] @ [q_tile, head_dim]

            """
            dS_int8_T = dS_int8.transpose(-1, -2)
            for d_tile in hl.tile([head_dim], block_size=[Bd]):
                q_int8_sub_d = q_int8[q_tile, d_tile]
                sq = sq_bh[q_tile.id, d_tile.id]
                dk[:, d_tile] = hl.dot(dS_int8_T, q_int8_sub_d)  * s_dS * sq * qk_scale

            dk_bh[k_tile, :] = dk
            dv_bh[q_tile, :] = dv

    return dq_bh.view([batch, head, q_tokens, head_dim]), \
        dk_bh.view([batch, head, q_tokens, head_dim]), \
        dv_bh.view([batch, head, q_tokens, head_dim])


@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_linked_fwd(
    q_fp16_input: torch.Tensor,
    k_fp16_input: torch.Tensor,
    v_fp16_input: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    int, int
    ]:
 
    """

    Forward tuning determines the block size Bq and Bkv for backward according to original SageAttention3 paper

    """
    batch, head, q_tokens, q_head_dim = q_fp16_input.shape
    batch, head, k_tokens, k_head_dim = k_fp16_input.shape
    batch, head, v_tokens, v_head_dim = v_fp16_input.shape

    assert k_tokens == v_tokens, "k and v tokens are different"
    assert k_head_dim == v_head_dim, "k head_dim and v head_dim are different"

    q_bh = q_fp16_input.reshape([-1, q_tokens, q_head_dim])
    k_bh = k_fp16_input.reshape([-1, k_tokens, k_head_dim]).transpose(1, 2)
    v_bh = v_fp16_input.reshape([-1, v_tokens, v_head_dim])

    cuda_device = q_fp16_input.device

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
    q_int8_bh = torch.zeros((batch*head, q_tokens, q_head_dim))

    split_kv = hl.register_tunable("split_kv", PowerOfTwoFragment(8, 256, 16))
    Bkv = helion.next_power_of_2(helion.cdiv(k_tokens, split_kv))

    sk_bh = torch.zeros((batch*head, split_kv), device=cuda_device)
    k_int8_bh_T = torch.zeros((batch*head, k_head_dim, k_tokens))

    sv_bh = torch.zeros((batch*head, split_kv), device=cuda_device)
    v_int8_bh = torch.zeros((batch*head, v_tokens, v_head_dim))

    for bh_tile, q_tile in hl.tile([batch*head, q_tokens], block_size=[None, Bq]):

        O = hl.zeros([bh_tile, q_tile, q_head_dim])
        l = hl.zeros([bh_tile, q_tile])
        m = hl.full([bh_tile, q_tile, 1], float("-inf"), dtype=torch.float16)

        q = q_bh[bh_tile, q_tile, :]        

        sq = torch.max(q.abs()) / 127
        q_int8 = q / sq
        q_int8 = q_int8.to(torch.int8)

        sq_bh[bh_tile, q_tile.id] = sq 
        q_int8_bh[bh_tile, q_tile, :] = q_int8

        for k_tile in hl.tile([k_tokens], block_size=[Bkv]):

            k = k_bh[bh_tile, :, k_tile]        

            sk = torch.max(k.abs()) / 127
            sk_bh[bh_tile, k_tile.id] = sk

            k_int8 = k / sk
            k_int8 = k_int8.to(torch.int8)
            k_int8_bh_T[bh_tile, :, k_tile] = k_int8
            # TODO replace bmm with hl.dot int8 x int8 -> int32
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

            v_int8_bh[bh_tile, k_tile, :] = v_int8

            O = torch.baddbmm(O, P_int8, v_int8) * sp * sv

        l_bh[bh_tile, q_tile] = m.squeeze(-1) + torch.log2(l).squeeze(-1)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile
        # usage qk = qk - bwd_normalizing_blk_fp32
        # exp_qk = torch.exp2(qk)

        O_final = O / l
        O_bh[bh_tile, q_tile, :] = O_final

    return O_bh.view([batch, head, q_tokens, q_head_dim]), l_bh, \
        q_int8_bh, k_int8_bh_T, v_int8_bh, \
        sq_bh, sk_bh, sv_bh, \
        Bq, Bkv


@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_int8_linked_bwd(
    batch, head,
    q_bh_int8: torch.Tensor,
    sq_bh: torch.Tensor, 

    k_bh_int8_T: torch.Tensor,
    k_mean_bh: torch.Tensor,
    sk_bh: torch.Tensor,

    v_bh_int8: torch.Tensor,
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

    batch_head, q_tokens, q_head_dim = q_bh_int8.shape
    batch_head, k_head_dim , k_tokens= k_bh_int8_T.shape
    batch_head, k_tokens = k_mean_bh.shape
    batch_head, v_tokens, v_head_dim = v_bh_int8.shape

    head_dim = q_head_dim

    O_bh = O_input.reshape([-1, q_tokens, head_dim])
    dO_bh = dO_input.reshape([-1, q_tokens, head_dim])

    sm_scale =  1.0 / math.sqrt(head_dim) 
    qk_scale = sm_scale * 1.44269504 

    cuda_device = q_bh_int8.device

    dq_bh = torch.zeros_like(q_bh_int8, dtype=torch.float16)
    dk_bh = torch.zeros([batch_head, k_tokens, head_dim], 
                        device=cuda_device, dtype=torch.float16) # not transposed
    dv_bh = torch.zeros_like(v_bh_int8, dtype=torch.float16)

    for bh_tile, k_tile in hl.tile([batch_head, k_tokens], block_size=[None, Bkv]):

        k_int8_T = k_bh_int8_T[bh_tile, :, k_tile]
        v_int8 = v_bh_int8[bh_tile, k_tile, :]

        sk = sk_bh[bh_tile, k_tile.id]
        sv = sv_bh[bh_tile, k_tile.id]

        dk = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float16)
        dv = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float16)

        for q_tile in hl.tile(q_tokens, block_size=[Bq]):

            q_int8 = q_bh_int8[bh_tile, q_tile, :]
            sq = sq_bh[bh_tile, q_tile.id]

            S = torch.bmm(q_int8, k_int8_T) * sq * sv * qk_scale  # batch matrix multiply

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

            # TODO fix: torch.baddnmm doesn't support int8
            dv = torch.baddbmm(dv, P_int8_T, dO_int8) * sP * s_dO

            # dv.shape 
            # [bh, k, head_dim] = [bh, k, q] @ [bh, q, head_dim]


            v_int8_T = torch.transpose(v_int8, 1, 2)
            dP_fp16 = torch.bmm(dO, v_int8_T) * sv 
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


def sage_attention_3_int8(q_fp16, k_fp16, v_fp16):
    """
    Sage Attention 3 backward and forward in Int8

    Args:
        q: query fp16 [batch, head, token, q_head_dim]
        k: key fp16 [batch, head, token, k_head_dim]
        v: value bf16 [batch, head, token, v_head_dim]

    returns:
        o: output = softmax(qk)/sqrt(d) * v

    """
    sage_output = SageAttention3_Int8_autograd_function.apply(
        q_fp16, k_fp16, v_fp16
    )

    return sage_output[0]

def baseline_pytorch_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    head_dim: int,
    causal: bool
) -> torch.Tensor:

    """Attention in Pytorch"""
    p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

    if causal:
        b, h, q_token, k_token = p.shape
        q_range_t = torch.arange(0, q_token).to(q.device)
        k_range_t = torch.arange(0, k_token).to(q.device)
        mask = q_range_t[:, None] - k_range_t[None, :] 
        mask = mask[None, None, :, :]
        p = torch.where(
            mask > 0, 
            p, 
            -128 * torch.log(
                torch.tensor([2],device=q.device)
            )
        ) 

    # p = torch._safe_softmax(p.float(), dim=-1).to(torch.float32)

    p = torch.softmax(p.to(torch.float32), dim=-1).to(torch.float32)
    return torch.matmul(p, v)

def test_forward_and_backward():

    z = 8
    h = 35
    n_ctx = 1024
    head_dim = 64
    dtype = torch.float32
    device = "cuda"
    causal = True

    q, k, v = [
        torch.randn(
            (z, h, n_ctx, head_dim),
            dtype=dtype, 
            device=device, 
            requires_grad=True
        )
        for _ in range(3)
    ]

    q_fp16 = q.clone().detach().to(torch.float16)
    k_fp16 = k.clone().detach().to(torch.float16)
    v_bf16 = v.clone().detach().to(torch.bfloat16)

    q_fp16.requires_grad = True
    k_fp16.requires_grad = True
    v_bf16.requires_grad = True

    ground_truth = torch.randn(
            (z, h, n_ctx, head_dim),
            dtype=dtype, 
            device=device, 
    )

    helion_sage_atten_fp16_t = sage_attention_3_int8(
        q_fp16, 
        k_fp16, 
        v_bf16, 
    )
    pytorch_t = baseline_pytorch_attention(q, k, v, head_dim=head_dim, causal=causal)

    helion_mse_diff = mse_loss(helion_sage_atten_fp16_t, ground_truth)
    pytorch_mse_diff = mse_loss(pytorch_t, ground_truth)

    helion_mse_diff.backward()
    pytorch_mse_diff.backward()

    abs_tol = 1e-2
    forward_close_bool_t = torch.isclose(
        helion_sage_atten_fp16_t.to(torch.float32),
        pytorch_t,
        atol=abs_tol,
        rtol=0
    )

    correct = torch.sum(forward_close_bool_t)
    total = helion_sage_atten_fp16_t.numel()
    error = total - correct
    mse_error = mse_loss(helion_sage_atten_fp16_t.to(torch.float32), pytorch_t)

    print("Helion Sage Attention Int8 Forward Output: ")
    print("at absolute tolerance: ", abs_tol)
    print("elements error: ", error)
    print("total elemetns: ", total)
    print("mse_error :", mse_error)

    print("")
    print("------------------------------")
    print("")

    print("Helion Sage Attention Int8 Backward Output: ")
    
    backward_close_bool_t = torch.isclose(
        q_fp16.grad.to(torch.float32),
        q.grad,
        atol=abs_tol,
        rtol=0
    )

    correct = torch.sum(backward_close_bool_t)
    total = q.numel()
    error = total - correct
    mse_error = mse_loss(q_fp16.grad.to(torch.float32), q.grad)


    print("")
    print("q.grad:")
    print("at absolute tolerance: ", abs_tol)
    print("elements error: ", error)
    print("total elemetns: ", total)
    print("mse_error :", mse_error)

    backward_close_bool_t = torch.isclose(
        k_fp16.grad.to(torch.float32),
        k.grad,
        atol=abs_tol,
        rtol=0
    )

    correct = torch.sum(backward_close_bool_t)
    total = k.numel()
    error = total - correct
    mse_error = mse_loss(k_fp16.grad.to(torch.float32), k.grad)


    print("")
    print("k.grad:")
    print("at absolute tolerance: ", abs_tol)
    print("elements error: ", error)
    print("total elemetns: ", total)
    print("mse_error :", mse_error)

    backward_close_bool_t = torch.isclose(
        v_bf16.grad.to(torch.float32),
        v.grad,
        atol=abs_tol,
        rtol=0
    )

    correct = torch.sum(backward_close_bool_t)
    total = v.numel()
    error = total - correct
    mse_error = mse_loss(v_bf16.grad.to(torch.float32), v.grad)


    print("")
    print("v.grad:")
    print("at absolute tolerance: ", abs_tol)
    print("elements error: ", error)
    print("total elemetns: ", total)
    print("mse_error :", mse_error)

    # torch.testing.assert_close(
    #     helion_bf16_t.to(torch.float32),
    #     pytorch_t,
    #     atol=1e-2,
    #     rtol=0
    # )

    # torch.testing.assert_close(
    #     q.grad, 
    #     q_fp16.grad.to(torch.float32),
    #     atol=1e-2,
    #     rtol=0
    # )

    # torch.testing.assert_close(
    #     k.grad, 
    #     k_fp16.grad.to(torch.float32),
    #     atol=1e-2,
    #     rtol=0
    # )

    # v_fp16 grad 2080 of 18350080 not close
    # torch.testing.assert_close(
    #     v.grad, 
    #     v_bf16.grad.to(torch.float32),
    #     atol=1e-2,
    #     rtol=0
    # )

def test_forward_int8() -> None:
    """

    Test the attention kernel implementation against PyTorch's native attention functions.

    """
    z = 8
    h = 35
    n_ctx = 1024
    head_dim = 64
    dtype = torch.float32
    device = "cuda"
    causal = False

    q, k, v = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    q_fp16 = q.to(torch.float16)
    k_fp16 = k.to(torch.float16)
    v_bf16 = v.to(torch.float16)

    out = helion_atten_int8_hl_dot_flat_fwd(q_fp16, k_fp16, v_bf16)
    helion_fp16_t = out[0]
    pytorch_t = baseline_pytorch_attention(q, k, v, head_dim=head_dim, causal=causal)
    mse_diff = torch.nn.functional.mse_loss(pytorch_t, helion_fp16_t.to(torch.float16))
    print(mse_diff)


if __name__ == "__main__":
    test_forward_int8()
    # test_forward_and_backward()

    


