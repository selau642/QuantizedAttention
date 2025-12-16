import torch
import helion
import helion.language as hl
from torch.nn.attention.flex_attention import flex_attention

import math
from typing import Tuple
from typing import cast

from helion._testing import DEVICE
from helion._testing import run_example

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
    bwd_normalizing_const = torch.zeros([batch*head, q_tokens], device=q_bh.device)
    atten = torch.empty_like(q_bh)
    sm_scale =  1.0 / math.sqrt(head_dim) 

    qk_scale = sm_scale * 1.44269504 
    # 1.44269504 = 1/ln(2), 
    # where ln = log base euler number(2.7182)

    for bh_idx_list, q_idx_list in hl.tile([batch*head, q_tokens]):

        # make buffers
        qk_max_tile = hl.full([bh_idx_list, q_idx_list], float("-inf"), dtype=torch.float32)
        exp_qk_sum_tile = torch.full_like(qk_max_tile, 1.0) 
        exp_qk_v = hl.zeros([bh_idx_list, q_idx_list, head_dim], dtype=torch.float32)

        # load q
        q_tile = q_bh[bh_idx_list, q_idx_list, :]

        for k_idx_list in hl.tile(k_tokens):
            # load k
            k_tile = k_bh[bh_idx_list, :, k_idx_list]
            
            # q dot k
            qk_tile = torch.bmm(q_tile, k_tile) # batch matrix multiply

            # recalculate max tile
            next_qk_max_tile = torch.max(qk_max_tile, torch.amax(qk_tile, -1) * qk_scale)
            qk_tile = qk_tile * qk_scale - next_qk_max_tile[:, :, None]

            exp_qk= torch.exp2(qk_tile) 
            # torch.exp2 is 2 power qk_tile, 
            # not e power qk_tile, 
            # qk_scale 1/ln2 with convert to 2 power qk_tile to e power qk_tile

            next_exp_qk_sum_tile = torch.sum(exp_qk, -1)

            rescale = torch.exp2(qk_max_tile - next_qk_max_tile)
            exp_qk_sum_tile = exp_qk_sum_tile * rescale + next_exp_qk_sum_tile
            exp_qk_v = exp_qk_v * rescale[:, :, None]

            v_tile = v_bh[bh_idx_list, k_idx_list, :]

            exp_qk = exp_qk.to(v.dtype)
            exp_qk_v = torch.baddbmm(exp_qk_v, exp_qk, v_tile) 
            # batch add accumulator to batch matrix multiply exp_qk, v_tile
            qk_max_tile = next_qk_max_tile

        # if training: 
        #    qk_max_tile += torch.log2(exp_qk_sum_tile)


        # if training: 
        # store qk_max_tile + log2(exp_qk_sum_tile) to bwd_normalizing_const for backprop
        bwd_normalizing_const[bh_idx_list, q_idx_list] = qk_max_tile + torch.log2(exp_qk_sum_tile)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile

        atten_tile = exp_qk_v/ exp_qk_sum_tile[:, :, None]
        atten[bh_idx_list, q_idx_list, :] = atten_tile.to(atten.dtype)
    
    return atten.view([batch, head, q_tokens, head_dim]) #, bwd_normalizing_const

@helion.kernel(
    static_shapes=True
)
def atten_bf16_fwd_training(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes forward for Attention in BrainFloat16
        Attention = Softmax(Q * K_T/SQRT(d_k)) * V

    Args:
        q: Query shape [batch, head, q_tokens, head_dim]
        k: Key shape [batch, head, k_tokens, head_dim]
        v: Value shape [batch, head, v_tokens, head_dim]

    Returns:
        Attention: shape [batch, head, q_tokens, head_dim] 
        bwd_normalizing_const: shape [batch*head, q_tokens] used for backward
    """
    BETA = 2.0 # suggest testing 2.0 to 8.0 for diff data distributions

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
    bwd_normalizing_const = torch.zeros([batch*head, q_tokens])
    atten = torch.empty_like(q_bh)
    sm_scale =  1.0 / math.sqrt(head_dim) 

    qk_scale = sm_scale * 1.44269504 
    # 1.44269504 = 1/ln(2), 
    # where ln = log base euler number(2.7182)

    for bh_idx_list, q_idx_list in hl.tile([batch*head, q_tokens]):

        # make buffers
        qk_max_tile = hl.full([bh_idx_list, q_idx_list], float("-inf"), dtype=torch.float32)
        exp_qk_sum_tile = torch.full_like(qk_max_tile, 1.0) 
        exp_qk_v = hl.zeros([bh_idx_list, q_idx_list, head_dim], dtype=torch.float32)

        # load q
        q_tile = q_bh[bh_idx_list, q_idx_list, :]

        for k_idx_list in hl.tile(k_tokens):
            # load k
            k_tile = k_bh[bh_idx_list, :, k_idx_list]
            
            # q dot k
            qk_tile = torch.bmm(q_tile, k_tile) # batch matrix multiply
            # qk_tile.shape = [bh_idx_list, q_idx_list, k_idx_list]

            # recalculate max tile
            next_qk_max_tile = torch.max(qk_max_tile, torch.amax(qk_tile, -1) * qk_scale)
            # next_qk_max_tile.shape = [bh_idx_list, q_idx_list]
            """


            bf16 adjustment to avoid floating point rounding errors


            """
            approx_max = (qk_tile >= (next_qk_max_tile[:, :, None] - 1e-3))
            num_approx_max = torch.sum(approx_max, dim=-1, keepdims=True)
            # num_approx_max.shape = [bh_idx_list, q_idx_list, 1]
            more_than_one_approx_max = (num_approx_max > 1)

            next_qk_max_tile = torch.where(
                more_than_one_approx_max & (next_qk_max_tile > 0), 
                BETA * next_qk_max_tile, 
                next_qk_max_tile
            )

            next_qk_max_tile = torch.where(
                more_than_one_approx_max & (next_qk_max_tile < 0), 
                0, 
                next_qk_max_tile
            )

            # rescaling qk_tile
            qk_tile = qk_tile * qk_scale - next_qk_max_tile[:, :, None]

            exp_qk= torch.exp2(qk_tile) 
            # torch.exp2 is 2 power qk_tile, 
            # not e power qk_tile, 
            # qk_scale 1/ln2 with convert to 2 power qk_tile to e power qk_tile

            next_exp_qk_sum_tile = torch.sum(exp_qk, -1)

            rescale = torch.exp2(qk_max_tile - next_qk_max_tile)
            exp_qk_sum_tile = exp_qk_sum_tile * rescale + next_exp_qk_sum_tile
            exp_qk_v = exp_qk_v * rescale[:, :, None]

            v_tile = v_bh[bh_idx_list, k_idx_list, :]

            exp_qk = exp_qk.to(v.dtype)
            exp_qk_v = torch.baddbmm(exp_qk_v, exp_qk, v_tile) 
            # batch add accumulator to batch matrix multiply exp_qk, v_tile
            qk_max_tile = next_qk_max_tile

        # if training: 
        #    qk_max_tile += torch.log2(exp_qk_sum_tile)


        # if training: 
        # store qk_max_tile + log2(exp_qk_sum_tile) to bwd_normalizing_const for backprop
        bwd_normalizing_const[bh_idx_list, q_idx_list] = qk_max_tile + torch.log2(exp_qk_sum_tile)
        # 2 power minus bwd_normalizing_const = (e power minus max_m ) / exp_qk_sum_tile

        atten_tile = exp_qk_v/ exp_qk_sum_tile[:, :, None]
        atten[bh_idx_list, q_idx_list, :] = atten_tile.to(atten.dtype)
    
    return atten.view([batch, head, q_tokens, head_dim]), bwd_normalizing_const


helion.kernel(
    static_shapes=True
)
def atten_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,   
    atten: torch.Tensor,
    bwd_normalizing_const: torch.Tensor,
    d_atten: torch.Tensor
):
    """
    Computes backward for Attention in BrainFloat16
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

    for bh_idx_list, q_idx_list in hl.tile([batch*head, q_tokens]):

        q_tile = q_bh[bh_idx_list, q_idx_list, :]
        v_tile = v_bh[bh_idx_list, q_idx_list, :]

        bwd_normalizing_const_tile = bwd_normalizing_const[bh_idx_list, q_idx_list]

        for k_idx_list in hl.tile(k_tokens):
            k_tile = k_bh[bh_idx_list, :, k_idx_list]
            atten_tile = atten_bh[bh_idx_list, q_idx_list, k_idx_list]
            # q dot k
            qk_tile = torch.bmm(q_tile, k_tile) * qk_scale # batch matrix multiply
            exp_qk= torch.exp2(qk_tile) * bwd_normalizing_const_tile 

            d_atten_tile = d_atten_bh[bh_idx_list, q_idx_list, ]

            dv_tile = torch.bmm(exp_qk, d_atten_tile)
            dp_tile = torch.bmm(d_atten_tile, v_tile)
            atten_jvp = torch.sum(d_atten_tile * atten_tile, dim=-1, keepdims=True)
            d_softmax = qk_tile * qk_scale * (dp_tile - atten_jvp)

            dk_tile = torch.bmm(d_softmax, k_tile)
            dq_tile = torch.bmm(d_softmax, q_tile)

            dq_bh[bh_idx_list, q_idx_list, :] = dq_tile 
            dk_bh[bh_idx_list, k_idx_list, :] = dk_tile
            dv_bh[bh_idx_list, k_idx_list, :] = dv_tile

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

if __name__ == "__main__":

    # Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    test_forward(2, 32, 1024, 64, torch.float16, device=DEVICE)

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