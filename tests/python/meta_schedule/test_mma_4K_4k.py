# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Integration test for CUDA with Tensor Core"""
# pylint: disable=missing-function-docstring
import pytest
import tvm.meta_schedule.testing.te_workload as te_workload
import tvm
import tvm.meta_schedule.testing.tir_tensor_intrin as tir_tensor_intrin  # pylint: disable=unused-import
from tvm import te, tir
from tvm.script import tir as T
import tvm.testing
import numpy as np
import os
from tvm.contrib import nvcc
import sys

TARGET = tvm.target.Target("nvidia/geforce-rtx-3070")

TASK = "gemm"
USE_MANUAL_CODE = False


@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target_format="ptx")
    return ptx


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


@T.prim_func
def matmul_16(
    A: T.Buffer[(16, 8), "float16"], B: T.Buffer[(8, 8), "float16"], C: T.Buffer[(16, 8), "float32"]
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(16, 8, 8):
        with T.block("C"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C[i, j], A[i, k], B[k, j])
            T.writes(C[i, j])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + T.cast(A[i, k], "float32") * T.cast(B[k, j], "float32")


def test_integration_matmul():
    N = 64
    M = 64
    K = 16
    workload = te_workload.matmul_fp16(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    # a = mk b = nk / kk c = nn
    # scope = local
    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        # Step 1. Rule-Auto-Tensorize
        # pylint: disable=invalid-name
        i, i_tc = sch.split(i, factors=[None, 16])
        j, j_tc = sch.split(j, factors=[None, 8])
        k, k_tc = sch.split(k, factors=[None, 8])
        sch.reorder(
            # fmt: off
            i, j, k,
            # tensor core
            i_tc, j_tc, k_tc,
            # fmt: on
        )
        block_inner = sch.blockize(i_tc)
        block_outer, block_inner = block_inner, block
        del block
        # Step 2. Rule-Multi-Level-Tiling
        # i_factors = sch.sample_perfect_tile(i, n=5, decision=[8, 1, 2, 1, 4])
        # j_factors = sch.sample_perfect_tile(j, n=5, decision=[1, 8, 4, 1, 2])
        i_factors = [1, 1, 2, 1, 2]
        j_factors = [1, 1, 2, 1, 4]
        k_factors = [1, 2, 1]
        i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
        k0, k1, k2 = sch.split(k, k_factors)
        # pylint: enable=invalid-name
        sch.reorder(
            # fmt: off
            i0, j0,   # S => blockIdx.x
            i1, j1,   # S => blockIdx.y
            j2, i2,   # S => threadIdx.y
            # cache_write here
            k0,       # R
            # vectorized cooperative fetching here
            k1,       # R
            i3, j3,   # S
            k2,       # R
            i4, j4,
            # S
            # fmt: on
        )
        block_idx = sch.fuse(i0, j0, i1, j1)
        thread_idy = sch.fuse(j2, i2)  # threadidx.y is actually number of warps here
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(thread_idy, "threadIdx.y")

        num_ty = i_factors[2] * j_factors[2]

        def fetch_to_shared(block, idx, ndim):
            # shared [128 x 32]
            block_read = sch.cache_read(block, idx, "shared")
            # block_read_local = sch.cache_read(block_read, 0, "local")
            sch.compute_at(block_read, k0)
            vector_size = 8
            warp_size = 32
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
            f_0, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            #sch.storage_align(block_read, 0, axis=-2, factor=32, offset=8)


        fetch_to_shared(block_outer, 1, 2)
        fetch_to_shared(block_outer, 2, 2)

        # fetch to warp - A 4096 * 4096 -> 256 * 512 * 16 * 8 -> 256 * 512 * 32 * 4
        A_warp = sch.cache_read(block_outer, 1, "warp")
        sch.compute_at(A_warp, k1)
        def lambda_a(i, j):
            i_0 = i // 16
            j_0 = j // 8
            k_0 = i % 16
            l_0 = j % 8
            k_1 = (k_0 % 8) * 4 + l_0 // 2
            l_1 = (k_0 // 8) * 2 + l_0 % 2
            return i_0, j_0, k_1, l_1

        sch.transform_layout(
            A_warp,
            buffer_index=0,
            is_write_index=True,
            index_map=lambda_a,
        )
        warp_loop1, warp_loop2 = sch.get_loops(A_warp)[-2:]
        # 32 * 8 -> 2 * 1 * 32 * 4
        i_0, k_0 = sch.split(warp_loop1, factors=[None, 16])
        j_0, l_0 = sch.split(warp_loop2, factors=[None, 8])
        k_0_0, k_0_1 = sch.split(k_0, [None, 8])
        l_0_0, l_0_1 = sch.split(l_0, [None, 2])
        sch.reorder(i_0, j_0, k_0_1, l_0_0, k_0_0, l_0_1)
        k_1 = sch.fuse(k_0_1, l_0_0)
        l_1 = sch.fuse(k_0_0, l_0_1)
        sch.bind(k_1, "threadIdx.x")

        # fetch to warp - B 4096 * 4096 -> 512 * 512 * 8 * 8 -> 512 * 512 * 32 * 2
        B_warp = sch.cache_read(block_outer, 2, "warp")
        sch.compute_at(B_warp, k1)
        def lambda_b(i, j):
            i_0 = i // 8
            j_0 = j // 8
            k_0 = i % 8
            l_0 = j % 8
            k_1 = k_0 // 2 + l_0 * 4
            l_1 = k_0 % 2
            return i_0, j_0, k_1, l_1
        
        sch.transform_layout(
            B_warp,
            buffer_index=0,
            is_write_index=True,
            index_map=lambda_b,
        )
        warp_loop1, warp_loop2 = sch.get_loops(B_warp)[-2:]
        # 8 * 32 -> 4 * 32 * 2
        i_0, k_0 = sch.split(warp_loop1, factors=[None, 8]) # 1 * 8
        j_0, l_0 = sch.split(warp_loop2, factors=[None, 8]) # 4 * 8
        k_0_1, l_0_1 = sch.split(k_0, [None, 2]) # 4 * 2
        # 1 * 4 * 2 * 4 * 8
        sch.reorder(i_0, j_0, l_0, k_0_1, l_0_1)
        k_1 = sch.fuse(l_0, k_0_1)
        sch.bind(k_1, "threadIdx.x")

        # fetch to C 4096 * 4096 -> 256 * 512 * 16 * 8 -> 256 * 512 * 32 * 4
        C_warp = sch.cache_write(block_outer, 0, "warp")
        sch.reverse_compute_at(C_warp, sch.get_loops(block_outer)[2])
        # need to do a reverse_compute_at to place it under blockidx.x
        def lambda_c(i, j):
            i_0 = i // 16
            j_0 = j // 8
            k_0 = i % 16
            l_0 = j % 8
            k_1 = (k_0 % 8) * 4 + l_0 // 2
            l_1 = (k_0 // 8) * 2 + l_0 % 2
            return i_0, j_0, k_1, l_1

        sch.transform_layout(
            C_warp,
            buffer_index=0,
            is_write_index=False,
            index_map=lambda_c,
        )
        warp_loop1, warp_loop2 = sch.get_loops(C_warp)[-2:]
        # 32 * 32 -> 2 * 4 * 32 * 4
        i_0, k_0 = sch.split(warp_loop1, factors=[None, 16])
        j_0, l_0 = sch.split(warp_loop2, factors=[None, 8])
        k_0_0, k_0_1 = sch.split(k_0, [None, 8])
        l_0_0, l_0_1 = sch.split(l_0, [None, 2])
        sch.reorder(i_0, j_0, k_0_1, l_0_0, k_0_0, l_0_1)
        k_1 = sch.fuse(k_0_1, l_0_0)
        l_1 = sch.fuse(k_0_0, l_0_1)
        sch.bind(k_1, "threadIdx.x")

        # Step 3.3. Decompose -> this may be needed
        loop = sch.get_loops(block_outer)[2]
        block_init_c = sch.decompose_reduction(block_outer, loop)

        # C_init() 16 * 8 -> 32 * 4
        # as binding is already transformed by previous step
        # only split/reorder/fuse is needed here
        C_init = sch.get_block("C_init")
        init_loop1, init_loop2 = sch.get_loops(C_init)[-2:]
        # print(sch.show(C_init))
        # for loop in sch.get_loops(C_init):
        #     print(sch.show(loop))
        #     print("+++++\n")
        f_0, f_1 = sch.split(init_loop1, factors=[None, 8])
        f_2, f_3 = sch.split(init_loop2, factors=[None, 2])
        sch.reorder(f_1, f_2, f_0, f_3)
        fused_1 = sch.fuse(f_1, f_2)
        fused_2 = sch.fuse(f_0, f_3)
        sch.bind(fused_1, "threadIdx.x")


        # tensorize
        loop1, loop2, loop3 = sch.get_loops(block_inner)
        sch.tensorize(loop1, "mma_sync")
        print(sch.mod["main"].script())

    sch = tir.Schedule(workload)
    schedule(sch)

    # if sch is None:
    #     print("No valid schedule found")
    # else:
    #     print(sch.mod["main"].script())
    #     print(tvm.lower(sch.mod["main"], None, simple_mode=True))

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(N, K)).astype("float16")
    b_np = np.random.uniform(size=(K, M)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
    # sys.exit(0)
    f = tvm.build(sch.mod["main"], target="cuda", name="dense")
    f(a, b, c)
    print(f.imported_modules[0].get_source())
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    # evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    # gflops = (N * M * K) * 2 / 1e9
    # time_ms = evaluator(a, b, c).mean * 1e3
    # print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


if __name__ == "__main__":
    test_integration_matmul()
    # test_matmul()
