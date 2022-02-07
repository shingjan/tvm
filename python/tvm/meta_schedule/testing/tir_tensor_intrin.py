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
"""A collection of TIR tensor intrinsics"""
# pylint: disable=missing-function-docstring
import tvm
from tvm import tir
from tvm.script import tir as T

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks
# fmt: off

@T.prim_func
def tensorcore_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        vk = T.axis.R(16, 0)
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii = T.axis.S(16, vi + i)
                vjj = T.axis.S(16, vj + j)
                vkk = T.axis.R(16, vk + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def tensorcore_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        vk = T.axis.R(16, 0)
        T.reads([
            C[vi : vi + 16, vj : vj + 16],
            A[vi : vi + 16, vk : vk + 16],
            B[vj : vj + 16, vk : vk + 16],
        ])
        T.writes(C[vi : vi + 16, vj : vj + 16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256,
                A.data,
                A.elem_offset // 256,
                B.data,
                B.elem_offset // 256,
                C.data,
                C.elem_offset // 256,
                dtype="handle",
            )
        )


@T.prim_func
def dot_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,))
    B = T.match_buffer(b, (4,))
    C = T.match_buffer(c, (1,))

    with T.block("root"):
        v0 = T.axis.R(4, 0)
        for i in range(0, 4):
            with T.block("update"):
                vi = T.axis.R(4, v0 + i)
                C[0] = C[0] + A[vi] * B[vi]


@T.prim_func
def dot_product_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,))
    B = T.match_buffer(b, (4,))
    C = T.match_buffer(c, (1,))

    with T.block("root"):
        v0 = T.axis.R(4, 0)
        T.reads([C[0 : 1], A[v0 : v0 + 4], B[v0 : v0 + 4]])
        T.writes([C[0 : 1]])
        T.evaluate(T.call_extern(  # pylint: disable=redundant-keyword-arg
            "vec4add",
            C.data, C.elem_offset,
            A.data, A.elem_offset,
            B.data, B.elem_offset,
            dtype="int32",
        ))

@T.prim_func
def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=1, scope="wmma.accumulator")

    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        vk = T.axis.R(16, 0)
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii = T.axis.S(16, vi + i)
                vjj = T.axis.S(16, vj + j)
                vkk = T.axis.R(16, vk + k)
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], "float32") * T.cast(B[vkk, vjj],
                                                                                        "float32")


@T.prim_func
def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16,
                         scope="wmma.accumulator")

    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        vk = T.axis.R(16, 0)
        T.reads([C[vi: vi+16, vj: vj+16], A[vi: vi+16, vk: vk+16], B[vk: vk+16, vj: vj+16]])
        T.writes(C[vi: vi+16, vj: vj+16])
        T.evaluate(T.tvm_mma_sync(C.data, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                                      A.data, A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                                      B.data, B.elem_offset // 256 + T.floordiv(T.floormod(B.elem_offset, 256), 16),
                                      C.data, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                                      dtype="handle"))


@T.prim_func
def wmma_load_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                         scope="shared")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_a")

    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii = T.axis.S(16, vi + i)
                vjj = T.axis.S(16, vj + j)
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        T.reads(A[vi: vi+16, vj: vj+16])
        T.writes(C[vi: vi+16, vj: vj+16])
        T.evaluate(T.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), A.access_ptr("r"), s1, "row_major",
            dtype="handle"))


@T.prim_func
def wmma_load_b_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii = T.axis.S(16, vi + i)
                vjj = T.axis.S(16, vj + j)
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        T.reads(A[vi: vi+16, vj: vj+16])
        T.writes(C[vi: vi+16, vj: vj+16])
        T.evaluate(T.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), A.access_ptr("r"), s1, "row_major",
            dtype="handle"))


@T.prim_func
def wmma_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii = T.axis.S(16, vi + i)
                vjj = T.axis.S(16, vj + j)
                C[vii, vjj] = T.float32(0)


@T.prim_func
def wmma_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        T.reads([])
        T.writes(C[vi : vi + 16, vj : vj + 16])
        T.evaluate(T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), T.float32(0), dtype="handle"))


@T.prim_func
def wmma_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii = T.axis.S(16, vi + i)
                vjj = T.axis.S(16, vj + j)
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s1, s0])
    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        T.reads(A[vi: vi + 16, vj: vj + 16])
        T.writes(C[vi: vi+16, vj: vj+16])
        T.evaluate(T.tvm_store_matrix_sync(
            A.data, 16, 16, 16, A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16), C.access_ptr("w"), s1, "row_major",
            dtype="handle"))


@T.prim_func
def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [32, 4], dtype="float16", scope="warp")
    B = T.match_buffer(b, [32, 2], dtype="float16", scope="warp")
    C = T.match_buffer(c, [32, 4], dtype="float32", scope="warp")
    with T.block("root"):
        T.reads(C[0 : 32, 0 : 4], A[0 : 32, 0 : 4], B[0 : 32, 0 : 2])
        T.writes(C[0 : 32, 0 : 4])
        for i0, i1, i2 in T.grid(16, 8, 8):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2], A[i % 8 * 4 + k // 2, i // 8 * 2 + k % 2], B[j * 4 + k // 2, k % 2])
                T.writes(C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2])
                C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2] = C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2] + T.cast(A[i % 8 * 4 + k // 2, i // 8 * 2 + k % 2], "float32") * T.cast(B[j * 4 + k // 2, k % 2], "float32")


@T.prim_func
def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32, 4), "float16", align=128, offset_factor=1, scope="warp")
    B = T.match_buffer(b, (32, 2), "float16", align=128, offset_factor=1, scope="warp")
    C = T.match_buffer(c, (32, 4), "float32", align=128, offset_factor=1, scope="warp")

    with T.block("root"):
        T.reads(C[0 : 32, 0 : 4], A[0 : 32, 0 : 4], B[0 : 32, 0 : 2])
        T.writes(C[0 : 32, 0 : 4])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)
        T.evaluate(T.call_extern("mma_m16n8k8_row_row_fp16fp16fp32", A.data, tx * 4, B.data, tx * 4, C.data, tx * 4, dtype="float32",))

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks

TENSORCORE_WMMA = tir.TensorIntrin.register(
    "test.tensorcore.wmma",
    tensorcore_desc,
    tensorcore_impl,
)

NEON_DOT = tir.TensorIntrin.register(
    "test.neon.dot",
    dot_product_desc,
    dot_product_impl,
)

WMMA_SYNC = tir.TensorIntrin.register(
    "wmma_sync",
    wmma_sync_desc,
    wmma_sync_impl,
)

WMMA_LOAD_A = tir.TensorIntrin.register(
    "wmma_load_a",
    wmma_load_a_desc,
    wmma_load_a_impl,
)

WMMA_LOAD_B = tir.TensorIntrin.register(
    "wmma_load_b",
    wmma_load_b_desc,
    wmma_load_b_impl,
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_fill",
    wmma_fill_desc,
    wmma_fill_impl,
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_store",
    wmma_store_desc,
    wmma_store_impl,
)

MMA_SYNC = tir.TensorIntrin.register(
    "mma_sync",
    mma_sync_desc,
    mma_sync_impl,
)
