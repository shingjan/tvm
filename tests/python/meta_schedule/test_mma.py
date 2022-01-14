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
    N = 16
    M = 8
    K = 8
    workload = te_workload.matmul_fp16(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)
    # a = mk b = nk / kk c = nn
    # scope = local
    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)

        # Step 2. Rule-Multi-Level-Tiling
        # i_factors = sch.sample_perfect_tile(i, n=5, decision=[8, 1, 2, 1, 4])
        # j_factors = sch.sample_perfect_tile(j, n=5, decision=[1, 8, 4, 1, 2])
        i1, i2 = sch.split(i, factors=[None, 16])
        sch.bind(i1, "blockIdx.x")

        # num_ty = sch.get(i_factors[2]) * sch.get(j_factors[2])

        def fetch_to_shared(block, idx, ndim):
            # shared [128 x 32]
            block_read = sch.cache_read(block, idx, "shared")
            # block_read_local = sch.cache_read(block_read, 0, "local")
            sch.compute_at(block_read, i1)
            warp_size = 32
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
            f_0, f_1 = sch.split(fused, factors=[None, warp_size])
            # sch.reorder(f_1, f_2, f_0, f_3)
            sch.bind(f_1, "threadIdx.x")
            # sch.bind(f_0, 'vthread.z')
            # sch.compute_at(block_read_local, f_2)

        fetch_to_shared(block, 1, 2)

        fetch_to_shared(block, 2, 2)

        # Step 3. Postproc-Rewrite-Tensorize
        # Step 3.1. Cache read

        # sch.cache_read("local")
        # or just do it in shared memory
        # nvidia wmma/warp memory -> TVM abstraction -> ends up in local memory
        # register tensor intrinsics
        # better just do it in shared memory using tensorize

        # Step 3.3. Decompose -> this may be needed
        loop = sch.get_loops(block)[1]

        block_init_c = sch.decompose_reduction(block, loop)

        sch.tensorize(i2, "mma_sync")

    sch = tir.Schedule(workload)
    schedule(sch)

    if sch is None:
        print("No valid schedule found")
    else:
        print(sch.mod["main"].script())
        print(tvm.lower(sch.mod["main"], None, simple_mode=True))

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
    # print(f.imported_modules[0].get_source())
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    gflops = (N * M * K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


if __name__ == "__main__":
    test_integration_matmul()
