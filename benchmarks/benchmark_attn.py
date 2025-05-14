# -*- coding: utf-8 -*-

import math
import torch
import triton
from attention_cuda import self_attention_cuda, flash_attention_v1_cuda, flash_attention_v2_cuda
from attention_cutlass import flash_attention_v2_cutlass
from flash_attn import flash_attn_func as flash_attn_func_offical
from flash_attention_cuda.flash_attn_triton import flash_attn_triton
from torch.nn.functional import scaled_dot_product_attention

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 7)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['tiny_fa2', 'tiny_fa2_cutlass', 'fa_official', 'sdpa'],
        # label name for the lines
        line_names=['tiny_fa2', 'tiny_fa2_cutlass', 'fa_official', 'sdpa'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'),
                ('blue', 'dotted'), ('red', 'dotted'), ('cyan', '-'), ('cyan', 'dotted')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    device = torch.device("cuda")
    dtype = torch.float16
    requires_grad = False
    # B, H, HQ, D = 4, 4, 64, 128
    B, H, D = 4, 64, 128
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)

    q_trans = q.transpose(1, 2)
    k_trans = k.transpose(1, 2)
    v_trans = v.transpose(1, 2)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider == 'tiny_fa2':
        results = triton.testing.do_bench(
            lambda: flash_attention_v2_cuda(q, k, v),
            quantiles=quantiles
        )
    elif provider == 'tiny_fa2_cutlass':
        results = triton.testing.do_bench(
            lambda: flash_attention_v2_cutlass(q, k, v, False, sm_scale),
            quantiles=quantiles
        )
    elif provider == 'fa_official':
        results = triton.testing.do_bench(
            lambda: flash_attn_func_offical(q, k, v, causal=False, softmax_scale=sm_scale),
            quantiles=quantiles
        )
    elif provider == 'sdpa':
        results = triton.testing.do_bench(
            lambda: scaled_dot_product_attention(q_trans, k_trans, v_trans, is_causal=False, scale=sm_scale),
            quantiles=quantiles
        )
    # elif provider == 'fa_triton':
    #     results = triton.testing.do_bench(
    #         lambda: flash_attn_triton(q, k, v, causal=False),
    #         quantiles=quantiles
    #     )
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')