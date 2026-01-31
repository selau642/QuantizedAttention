"""
Kernel to compute Attention with Forward Differentiation

Large Scale Diffusion Distillation 
via Score-Regularized Continuous-Time Consistency
Kaiwen Zheng, Yuji Wang, Qianli Ma
By Nvidia

https://research.nvidia.com/labs/dir/rcm

"""

import torch
import helion
import helion.language as hl

from torch.autograd import Function
from typing import Tuple
import math
from helion._testing import DEVICE
from helion.autotuner import PowerOfTwoFragment
from torch.func import jvp

@helion.kernel(
    autotune_effort="none",
    static_shapes=True,
    config=helion.Config(
        block_sizes=[2, 16, 16] 
        # block_size=[16, 16, 16] -> not enoughed shared memory for RTX3080
        # x=4, y=16, z=16, if y or z < 16 there is some funny join then permute bug
    )
)
def helion_attention_jvp_forward_fp32(
    q_fp32_input,
    k_fp32_input,
    v_fp32_input,

    tan_q_fp32_input,
    tan_k_fp32_input,
    tan_v_fp32_input
):

    """

    Computes forward for Attention with Forward Differentiation
        Attention = Softmax(Q * K_T/SQRT(d_k)) * V

        tS = tQ dot K_T + Q dot tK_T
        tP = P hadamard_prod tS - P hadamard_prod ((P hadamard_prod tS ) dot 1_N dot 1_N_T)
        tO = tP dot V + P dot tV
        
        let H = P hadamard_prod tS

        tO = P dot tV + HV - PV diag(rowsum(H))
            = P dot tV + HV - O diag(rowsum(H))

    Args:
        q_input: Query shape [batch, head, q_tokens, head_dim]
        tan_q_input: Query shape [batch, head, q_tokens, head_dim]

        k_input: Key shape [batch, head, k_tokens, head_dim]
        tan_k_input: Key shape [batch, head, k_tokens, head_dim]

        v_input: Value shape [batch, head, v_tokens, head_dim]
        tan_v_input: Value shape [batch, head, v_tokens, head_dim]

    Returns:
        o: output = softmax(qk)/sqrt(d) * v shape [batch, head, q_tokens, head_dim] 
        lse: normalizing const shape [batch*head, q_tokens] used for backward

    """

    batch, head, q_tokens, q_head_dim = q_fp32_input.shape
    k_batch, k_head, k_tokens, k_head_dim = k_fp32_input.shape
    v_batch, v_head, v_tokens, v_head_dim = v_fp32_input.shape
    cuda_device = q_fp32_input.device

    assert k_tokens == v_tokens, "input k_tokens must match v_tokens"
    assert q_head_dim == k_fp32_input.size(-1) == v_fp32_input.size(-1), "all head dimensions must match for q, k, v tensors"

    head_dim = q_head_dim
    q_bh_fp32 = q_fp32_input.reshape([-1, q_tokens, head_dim])
    k_bh_fp32 = k_fp32_input.reshape([-1, k_tokens, head_dim]).transpose(1, 2)
    v_bh_fp32 = v_fp32_input.reshape([-1, v_tokens, head_dim])

    tan_q_bh_fp32 = tan_q_fp32_input.reshape([-1, q_tokens, head_dim])
    tan_k_bh_fp32 = tan_k_fp32_input.reshape([-1, k_tokens, head_dim]).transpose(1, 2)
    tan_v_bh_fp32 = tan_v_fp32_input.reshape([-1, v_tokens, head_dim])

    """
    variable definitions

    S = QK -> similarity
    m = rowmax(S) -> max value in a row
    l = rowsum(S) -> running sum of a row for online softmax
    P = exp(S-M) -> probability
    O = PV -> expected value

    scale = 1 / sqrt(head_dim)

    """
    l_bh_fp32 = torch.zeros(
        [batch*head, q_tokens], 
        dtype=torch.float32, 
        device=cuda_device
        )
    # lse = rowsum p

    O_bh_fp32 = torch.empty_like(
        q_bh_fp32, 
        dtype=torch.float32, 
        device=cuda_device
        )

    tO_bh_fp32 = torch.empty_like(
        q_bh_fp32, 
        dtype=torch.float32, 
        device=cuda_device
        )



    sm_scale =  1.0 / math.sqrt(head_dim) 
    
    qk_scale = sm_scale * 1.44269504 
    # 1.44269504 = 1/ln(2), 
    # where ln = log base euler number(2.7182)

    for bh_tile, q_tile in hl.tile([batch*head, q_tokens]):
        m_fp32 = hl.full([bh_tile, q_tile, 1], float("-inf"), dtype=torch.float32)
        l_fp32 = hl.zeros([bh_tile, q_tile, 1], dtype=torch.float32) 
        O_fp32 = hl.zeros([bh_tile, q_tile, head_dim], dtype=torch.float32)

        r_fp32 = hl.zeros([bh_tile, q_tile, 1], dtype=torch.float32)

        A_fp32 = hl.zeros([bh_tile, q_tile, head_dim], dtype=torch.float32)
        B_fp32 = hl.zeros([bh_tile, q_tile, head_dim], dtype=torch.float32)

        q_fp32 = q_bh_fp32[bh_tile, q_tile, :]
        tan_q_fp32 = tan_q_bh_fp32[bh_tile, q_tile, :]
        # [batch*head_slice, q_token_slice, q_head_dim]

        for k_tile in hl.tile(k_tokens):
            # load k
            k_fp32_T = k_bh_fp32[bh_tile, :, k_tile]
            tan_k_fp32_T = tan_k_bh_fp32[bh_tile, :, k_tile]

            S_fp32 = torch.bmm(q_fp32, k_fp32_T)
            tan_Q_dot_k_T_fp32 = torch.bmm(tan_q_fp32, k_fp32_T)
            Q_dot_tk_T_fp32 = torch.bmm(q_fp32, tan_k_fp32_T)

            tS_fp32 = tan_Q_dot_k_T_fp32 + Q_dot_tk_T_fp32
            tS_fp32 = tS_fp32 * sm_scale
            
            next_m_fp32 = torch.max(
                m_fp32, 
                torch.amax(S_fp32, -1, keepdim=True) * qk_scale
            ) #.to(torch.bfloat16)

            S_fp32 = S_fp32 * qk_scale - next_m_fp32
            P_fp32 = torch.exp2(S_fp32)
            next_l_fp32 = torch.sum(P_fp32, -1, keepdim=True)

            rescale = torch.exp2((m_fp32 - next_m_fp32))
            l_fp32 = l_fp32 * rescale + next_l_fp32
            m_fp32= next_m_fp32
            O_fp32 = O_fp32 * rescale
            v_fp32 = v_bh_fp32[bh_tile, k_tile, :]
            tan_v_fp32 = tan_v_bh_fp32[bh_tile, k_tile, :]

            O_fp32 = torch.baddbmm(O_fp32, P_fp32, v_fp32) 

            A_fp32 = A_fp32 * rescale
            A_fp32 = torch.baddbmm(A_fp32, P_fp32, tan_v_fp32)

            H_fp32 = P_fp32 * tS_fp32

            r_fp32 = r_fp32 * rescale + torch.sum(H_fp32, dim=-1, keepdim=True)

            B_fp32 = B_fp32 * rescale
            B_fp32 = torch.baddbmm(B_fp32, H_fp32, v_fp32) 

        l_bh_fp32[bh_tile, q_tile] = m_fp32.squeeze(-1) + torch.log2(l_fp32).squeeze(-1)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile
        # usage qk = qk - bwd_normalizing_blk_fp32
        # exp_qk = torch.exp2(qk)

        O_final_fp32 = O_fp32 / l_fp32
        O_bh_fp32[bh_tile, q_tile, :] = O_final_fp32 #.to(torch.float32)
        tO_bh_fp32[bh_tile, q_tile, : ] = (A_fp32 + B_fp32 - r_fp32 * O_final_fp32) / l_fp32


    return O_bh_fp32.view([batch, head, q_tokens, head_dim]), \
        tO_bh_fp32.view([batch, head, q_tokens, head_dim]), \
        l_bh_fp32

def baseline_pytorch_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
) -> torch.Tensor:

    """

    Attention in Pytorch

    """

    batch, head, tokens, head_dim = q.shape
    s = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

    # p = torch._safe_softmax(p.float(), dim=-1).to(torch.float32)

    p = torch.softmax(s.to(torch.float32), dim=-1).to(torch.float32)
    return torch.matmul(p, v)

def test_forward_jvp(
    z: int,
    h: int,
    n_ctx: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> None:
    causal = True
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
    q_fp32, k_fp32, v_fp32 = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    tan_q_fp32, tan_k_fp32, tan_v_fp32 = [
        torch.ones((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]


    O_helion_fp32, tO_helion_fp32, l = helion_attention_jvp_forward_fp32(
        q_fp32, k_fp32, v_fp32,
        tan_q_fp32, tan_k_fp32, tan_v_fp32,
        )

    pytorch_t = baseline_pytorch_attention( q_fp32, k_fp32, v_fp32)
    O_pt, tO_pt = jvp(
        baseline_pytorch_attention,
        (q_fp32, k_fp32, v_fp32),
        (tan_q_fp32, tan_k_fp32, tan_v_fp32)
    )

    abs_tol = 1e-2
    O_jvp_is_close_bool_t = torch.isclose(
        O_pt.to(torch.float32),
        O_helion_fp32,
        atol=abs_tol,
        rtol=0
    )

    correct = torch.sum(O_jvp_is_close_bool_t)
    total = O_pt.numel()
    error = total - correct
    O_mse_diff = torch.nn.functional.mse_loss(O_pt, O_helion_fp32)

    print("")
    print("O_pt:")
    print("at absolute tolerance: ", abs_tol)
    print("elements error: ", error)
    print("total elements: ", total)
    print("mse_error :", O_mse_diff)

    tO_jvp_is_close_bool_t = torch.isclose(
        tO_pt.to(torch.float32),
        tO_helion_fp32,
        atol=abs_tol,
        rtol=0
    )

    correct = torch.sum(tO_jvp_is_close_bool_t)
    total = tO_pt.numel()
    error = total - correct

    tO_mse_diff = torch.nn.functional.mse_loss(tO_pt, tO_helion_fp32)

    print("")
    print("tO_pt:")
    print("at absolute tolerance: ", abs_tol)
    print("elements error: ", error)
    print("total elements: ", total)
    print("mse_error :", tO_mse_diff)
    

if __name__ == "__main__":
    test_forward_jvp(
        8, 35, 1024, 64, torch.float32, device=DEVICE
    ) 
    """
    O_pt:
    at absolute tolerance:  0.01
    elements error:  tensor(0, device='cuda:0')
    total elements:  18350080
    mse_error : tensor(6.6253e-09, device='cuda:0')

    tO_pt:
    at absolute tolerance:  0.01
    elements error:  tensor(0, device='cuda:0')
    total elements:  18350080
    mse_error : tensor(1.2681e-07, device='cuda:0')
    """