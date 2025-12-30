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

from torch.nn.functional import mse_loss

class FlashAttention_2_BF16_autograd_function(Function):
    """

    forward in bf16
    backward in fp32

    """
    @staticmethod
    def forward(
        q_fp16: torch.Tensor, 
        k_fp16: torch.Tensor, 
        v_bf16: torch.Tensor, 
        causal: bool
    ):
        """
        Docstring for forward
        Args: 
            q, k, v = query, key, value inputs with dim [batch, head, tokens, head_dim]
        return:
        """
        o_fp32, lse_fp32 = helion_atten_bf16_fwd_training(
            q_fp16,
            k_fp16,
            v_bf16, 
            causal
        ) 

        # lse = const to normalize exp values to between 0 and 1, 
        # so that they become probability values

        # assert torch.isnan(atten_fp32).sum() == 0
        # assert torch.isnan(lse).sum() == 0

        return o_fp32, lse_fp32

    @staticmethod
    def setup_context(ctx, inputs, output):
        q_fp16, k_fp16, v_bf16, causal = inputs
        O_fp32, lse_fp32 = output
        ctx.mark_non_differentiable(lse_fp32)
        ctx.save_for_backward(q_fp16, k_fp16, v_bf16, O_fp32, lse_fp32)
        ctx.args = causal

    @staticmethod
    def backward(
        ctx,
        dO,
        _lse
    ):
        """

        Backward in standard fp32

        """

        q_fp16, k_fp16, v_bf16, O_fp32, lse_fp32 = ctx.saved_tensors
        causal = ctx.args
        dq_fp32, dk_fp32, dv_fp32 = helion_flash_atten_2_algo_4(
            q_fp16,
            k_fp16,
            v_bf16,   
            O_fp32,
            lse_fp32,
            causal,
            dO, 
        )

        return dq_fp32, dk_fp32, dv_fp32, None # causal

def flash_atten_2_bf16(q_fp16, k_fp16, v_bf16, causal):
    """
    Flash Attention 2 intermediate calculations in bf16

    Args:
        q_fp16: query fp16 [batch, head, token, q_head_dim]
        k_fp16: key fp16 [batch, head, token, k_head_dim]
        v_bf16: value bf16 [batch, head, token, v_head_dim]
        causal: True/False

    returns:
        o_fp32: output fp32 = softmax(qk)/sqrt(d) * v

    """
    o_fp32, _lse_fp32 = FlashAttention_2_BF16_autograd_function.apply(
        q_fp16, k_fp16, v_bf16, causal
    )

    return o_fp32 

@helion.kernel(
    autotune_effort="none",
    static_shapes=True
)
def helion_atten_bf16_fwd_training(
    q_fp16_input: torch.Tensor,
    k_fp16_input: torch.Tensor,
    v_bf16_input: torch.Tensor,
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
        o: output = softmax(qk)/sqrt(d) * v shape [batch, head, q_tokens, head_dim] 
        lse: normalizing const shape [batch*head, q_tokens] used for backward

    """


    BETA = 2.0 
    # at BETA=8.0, exp(QK) / Sum(exp(QK)) = overflows to NaN

    batch, head, q_tokens, q_head_dim = q_fp16_input.shape
    k_batch, k_head, k_tokens, k_head_dim = k_fp16_input.shape
    v_batch, v_head, v_tokens, v_head_dim = v_bf16_input.shape
    cuda_device = q_fp16_input.device

    assert k_tokens == v_tokens, "input k_tokens must match v_tokens"
    assert q_head_dim == k_fp16_input.size(-1) == v_bf16_input.size(-1), "all head dimensions must match for q, k, v tensors"

    head_dim = q_head_dim
    q_bh_fp16 = q_fp16_input.reshape([-1, q_tokens, head_dim])
    k_bh_fp16 = k_fp16_input.reshape([-1, k_tokens, head_dim]).transpose(1, 2)
    v_bh_bf16 = v_bf16_input.reshape([-1, v_tokens, head_dim])

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
        q_bh_fp16, 
        dtype=torch.float32, 
        device=cuda_device
        )


    sm_scale =  1.0 / math.sqrt(head_dim) 

    qk_scale = sm_scale * 1.44269504 
    # 1.44269504 = 1/ln(2), 
    # where ln = log base euler number(2.7182)


    for bh_tile, q_tile in hl.tile([batch*head, q_tokens]):

        m_bf16 = hl.full([bh_tile, q_tile, 1], float("-inf"), dtype=torch.bfloat16)
        l_fp32 = hl.full([bh_tile, q_tile, 1], 1.0, dtype=torch.float32) 
        O_fp32 = hl.zeros([bh_tile, q_tile, head_dim], dtype=torch.float32)

        q_fp16 = q_bh_fp16[bh_tile, q_tile, :]
        # [batch*head_slice, q_token_slice, q_head_dim]

        begin_q = q_tile.begin

        for k_tile in hl.tile(k_tokens):
            # load k
            begin_k = k_tile.begin
            end_k = k_tile.end

            k_fp16 = k_bh_fp16[bh_tile, :, k_tile]
            # [batch*head_slice, k_head_dim, k_token_slice] 

            # S = q dot k
            S_fp16 = torch.bmm(q_fp16, k_fp16)
            S_bf16 = S_fp16.to(torch.bfloat16) 

            # batch matrix multiply
            # fp16 inputs will still accumulate in fp32
            # S_fp16.shape = [bh_idx_list, q_idx_list, k_idx_list]

            if causal and begin_q < end_k:

                q_range_t = torch.arange(0, q_tile.block_size, device=cuda_device) + begin_q
                k_range_t = torch.arange(0, k_tile.block_size, device=cuda_device) + begin_k
                mask_tile = q_range_t[:, None] - k_range_t[None, :]

                bf16_inf_t = torch.tensor([-126], dtype=torch.bfloat16, device=cuda_device) 
                S_bf16 = torch.where(
                    mask_tile > 0,
                    S_bf16, # lower triangle
                    bf16_inf_t # upper triangle
                )

            # recalculate max tile
            next_m_bf16 = torch.max(
                m_bf16, 
                torch.amax(S_bf16, -1, keepdim=True) * qk_scale
            ).to(torch.bfloat16)

            """


            bf16 adjustment to avoid floating point rounding errors


            """
            approx_max = (S_bf16 >= (next_m_bf16 - 1e-3))
            num_approx_max = torch.sum(approx_max, dim=-1, keepdims=True)
            more_than_one_approx_max = (num_approx_max > 1)

            next_m_bf16 = torch.where(
                more_than_one_approx_max & (next_m_bf16 > 0), 
                BETA * next_m_bf16, 
                next_m_bf16
            )

            bf16_zero_t = torch.tensor(0.0, dtype=torch.bfloat16, device=cuda_device)

            next_m_bf16 = torch.where(
                more_than_one_approx_max & (next_m_bf16 < 0), 
                bf16_zero_t,
                next_m_bf16
            )

            # rescaling qk_tile
            S_bf16 = S_bf16 * qk_scale - next_m_bf16

            P_bf16 = torch.exp2(S_bf16.to(torch.float32)).to(torch.bfloat16)
            # torch.exp2 is 2 ^ qk, 
            # not e ^ qk, 
            # qk_scale 1/ln2 with convert to 2 ^ qk to e ^ qk

            next_l_fp32 = torch.sum(P_bf16.to(torch.float32), -1, keepdim=True)

            rescale_bf16 = torch.exp2((m_bf16 - next_m_bf16).to(torch.float32)).to(torch.bfloat16)
            m_bf16 = next_m_bf16

            l_fp32 = l_fp32 * rescale_bf16.to(torch.float32) + next_l_fp32
            O_fp32 = O_fp32 * rescale_bf16

            v_bf16 = v_bh_bf16[bh_tile, k_tile, :]
            # [batch*head_slice, k_token_slice, value_head_dim] 

            O_fp32 = torch.baddbmm(O_fp32, P_bf16, v_bf16) 
            # batch add accumulator to batch matrix multiply P@V

        l_bh_fp32[bh_tile, q_tile] = m_bf16.squeeze(-1) + torch.log2(l_fp32).squeeze(-1)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile
        # usage qk = qk - bwd_normalizing_blk_fp32
        # exp_qk = torch.exp2(qk)

        O_final_fp32 = O_fp32 / l_fp32
        O_bh_fp32[bh_tile, q_tile, :] = O_final_fp32 #.to(torch.float32)
            
    return O_bh_fp32.view([batch, head, q_tokens, head_dim]), l_bh_fp32 
   

@helion.kernel(
    autotune_effort="none",
    static_shapes=True,
    config=helion.Config(
        block_sizes=[2, 16, 16] 
        # block_size=[16, 16, 16] -> not enoughed shared memory for RTX3080
        # x=4, y=16, z=16, if y or z < 16 there is some funny join then permute bug
        
    )
)
def helion_flash_atten_2_algo_4(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,   
    O_input: torch.Tensor,
    lse_input: torch.Tensor,
    causal: bool,

    dO_input: torch.Tensor,
):
    """
    Computes backward for Attention in Float32 
    Using FlashAtten 2 Algo 4
        Attention = Softmax(Q * K_T/SQRT(d_k)) * V

    Args:
        q: Query shape [batch, head, q_tokens, head_dim]
        k: Key shape [batch, head, k_tokens, head_dim]
        v: Value shape [batch, head, v_tokens, head_dim]
        O: Output shape [batch, head, v_tokens, head_dim] = softmax(QK) * V
        lse: logit of (exp - max_row) / sum_exp, shape [batch*head, q_tokens]

        dO: Output shape [batch, head, q_tokens, head_dim]

    Returns:

    """

    batch, head, q_tokens, q_head_dim = q_input.shape
    k_batch, k_head, k_tokens, k_head_dim = k_input.shape
    v_batch, v_head, v_tokens, v_head_dim = v_input.shape

    head_dim = q_head_dim
    q_input = q_input.to(torch.float32)
    k_input = k_input.to(torch.float32)
    v_input = v_input.to(torch.float32)

    q_bh = q_input.reshape([-1, q_tokens, head_dim])
    k_bh = k_input.reshape([-1, k_tokens, head_dim]).transpose(-1, -2)
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
        # v.shape = [bh, k, head_dim]

        dk = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float32)
        dv = hl.zeros((bh_tile, k_tile, head_dim), device=cuda_device, dtype=torch.float32)
        begin_k = k_tile.begin
        end_k = k_tile.end

        for q_tile in hl.tile(q_tokens):
            begin_q = q_tile.begin

            q = q_bh[bh_tile, q_tile, :]
            S = torch.bmm(q, k)  # batch matrix multiply
            S = qk_scale * S

            if causal and begin_q < end_k:

                q_range_t = torch.arange(0, q_tile.block_size, device=cuda_device) + begin_q
                k_range_t = torch.arange(0, k_tile.block_size, device=cuda_device) + begin_k
                mask_tile = q_range_t[:, None] - k_range_t[None, :] 
                inf_t = torch.tensor([-128], device=cuda_device) # float32 log2(min_value) = 128
                S = torch.where(
                    mask_tile> 0,
                    S,
                    inf_t
                )

            l = lse_input[bh_tile, q_tile]
            P = torch.exp2(S - l[:, :, None])
            # P.shape = [bh, q, k]
            P_T = torch.transpose(P, 1, 2)

            dO = dO_bh[bh_tile, q_tile, :]
            # dO.shape = [bh, q, head_dim]

            dv = torch.baddbmm(dv, P_T, dO)
            # dv.shape 
            # [bh, k, head_dim] = [bh, k, q] @ [bh, q, head_dim]


            v_T = torch.transpose(v, 1, 2)
            dP = torch.bmm(dO, v_T)
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

            dS = S * (dP - D)

            # [bh, q, k] * [bh, q, k] 

            k_T = k.transpose(-1,-2)

            dq = dq_bh[bh_tile, q_tile, :]
            dq_bh[bh_tile, q_tile, :] = torch.baddbmm(
                dq, 
                qk_scale * dS,
                k_T 
            )

            # dq_bh.shape 
            # [bh, q, head_dim] = [bh, q, k] @ [bh, k, head_dim]
            dS_T = qk_scale * dS.transpose(-1, -2)
            dk = torch.baddbmm(
                dk, 
                dS_T, 
                q
            )

        dk_bh[bh_tile, k_tile, :] = dk
        dv_bh[bh_tile, k_tile, :] = dv

    return dq_bh.view([batch, head, q_tokens, head_dim]), \
        dk_bh.view([batch, head, k_tokens, head_dim]), \
        dv_bh.view([batch, head, v_tokens, head_dim])

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


# def test_forward(
#     z: int,
#     h: int,
#     n_ctx: int,
#     head_dim: int,
#     dtype: torch.dtype = torch.float32,
#     device: torch.device | str = "cuda",
# ) -> None:
#     """
#     Test the attention kernel implementation against PyTorch's native attention functions.

#     Args:
#         z: Batch size
#         h: Number of attention heads
#         n_ctx: Sequence length (context size)
#         head_dim: Dimension of each attention head
#         dtype: Data type for the tensors
#         device: Device to run the test on
#     """
#     q, k, v = [
#         torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
#         for _ in range(3)
#     ]

#     def ref_attention(
#         q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
#     ) -> torch.Tensor:
#         """Reference manual attention implementation"""
#         p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
#         p = torch.softmax(p.float(), dim=-1).to(dtype)
#         return torch.matmul(p, v)

#     flex_compiled = cast(
#         "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
#     )
#     baselines = {
#         "torch": torch.nn.functional.scaled_dot_product_attention,
#         "flex": flex_compiled,
#         "ref": ref_attention,
#     }

#     run_example(atten_fwd_training, baselines, (q, k, v))

#     helion_t = atten_fwd_training(q, k, v)
#     pytorch_t = ref_attention(q, k, v)
#     torch.testing.assert_close(pytorch_t, helion_t, atol=1e-2, rtol=0)

def test_forward_bf16(
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
    q, k, v = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    q_fp16 = q.to(torch.float16)
    k_fp16 = k.to(torch.float16)
    v_bf16 = v.to(torch.bfloat16)

    helion_fp32_t, _ = helion_atten_bf16_fwd_training(q_fp16, k_fp16, v_bf16, causal=causal)
    pytorch_t = baseline_pytorch_attention(q, k, v, head_dim=head_dim, causal=causal)
    mse_diff = torch.nn.functional.mse_loss(pytorch_t, helion_fp32_t)
    print(mse_diff)

    # torch.testing.assert_close(pytorch_t, helion_fp32_t, atol=1e-2, rtol=0)
    # 915 out of 18350080 elems not close

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

    helion_bf16_t = flash_atten_2_bf16(
        q_fp16, 
        k_fp16, 
        v_bf16, 
        causal=causal
    )
    pytorch_t = baseline_pytorch_attention(q, k, v, head_dim=head_dim, causal=causal)

    helion_mse_diff = mse_loss(helion_bf16_t, ground_truth)
    pytorch_mse_diff = mse_loss(pytorch_t, ground_truth)

    helion_mse_diff.backward()
    pytorch_mse_diff.backward()

    abs_tol = 1e-2
    forward_close_bool_t = torch.isclose(
        helion_bf16_t.to(torch.float32),
        pytorch_t,
        atol=abs_tol,
        rtol=0
    )

    correct = torch.sum(forward_close_bool_t)
    total = helion_bf16_t.numel()
    error = total - correct
    mse_error = mse_loss(helion_bf16_t.to(torch.float32), pytorch_t)

    print("Helion Bf16 Forward Output: ")
    print("at absolute tolerance: ", abs_tol)
    print("elements error: ", error)
    print("total elemetns: ", total)
    print("mse_error :", mse_error)

    print("")
    print("------------------------------")
    print("")

    print("Helion fp32 Backward Output: ")
    
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
    test_forward_and_backward()