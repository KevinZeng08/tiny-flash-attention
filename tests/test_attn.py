# -*- coding: utf-8 -*-

import os
import math

import pytest
import torch

from attention_cuda import self_attention_cuda, flash_attention_v1_cuda, flash_attention_v2_cuda
from attention_cutlass import flash_attention_v2_cutlass
from flash_attn import flash_attn_func as flash_attn_func_offical
from flash_attention_cuda.flash_attn_triton import flash_attn_triton

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False


test_b_list = [2, 4]
test_t_list = [1, 15, 63, 286, 300, 1024, 2048]
# test_t_varlen_list = [63, 286, 300, 512]
test_d_list = [32, 64, 100]
test_h_list = [8, 16, 32, 40, 64]
# TODO: support GQA
# test_hq_list = [8, 16]
# test_h_list = [2]
device = torch.device("cuda")

@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.skipif(
    not HAS_FLASH,
    reason="Skipping test because flash-attn is not installed"
)
def test_parallel(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
):
    # if not check_shared_mem('hopper') and D > 128:
    #     pytest.skip(reason="Skip test, do not have enough shard mem")
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(False)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(False)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(False)
    scale = 1.0 / math.sqrt(D)

    ref = flash_attn_func(q=q, k=k, v=v, softmax_scale=scale, causal=False)

    ting_fa_cuda = flash_attention_v2_cuda(q, k, v)
    torch.allclose(ref, ting_fa_cuda, rtol=0, atol=5e-3)

    tiny_fa_cutlass, _ = flash_attention_v2_cutlass(q, k, v, False, scale)
    torch.allclose(ref, tiny_fa_cutlass, rtol=0, atol=5e-3)
