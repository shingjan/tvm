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
# pylint: disable=missing-function-docstring,missing-module-docstring
<<<<<<< HEAD
import pytest
import numpy as np
=======
import sys
import pytest
>>>>>>> 6c33791db... squash blockize commits
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
<<<<<<< HEAD
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg

@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def desc_func(a: T.handle, b: T.handle, c: T.handle) -> None:
=======
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@T.prim_func
def mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
>>>>>>> 6c33791db... squash blockize commits
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
<<<<<<< HEAD
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        vk = T.axis.R(16, 0)
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii = T.axis.S(16, vi + i)
                vjj = T.axis.S(16, vj + j)
                vkk = T.axis.R(16, vk + k)
=======
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
>>>>>>> 6c33791db... squash blockize commits
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
<<<<<<< HEAD
def intrin_func(a: T.handle, b: T.handle, c: T.handle) -> None:
=======
def mma_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
>>>>>>> 6c33791db... squash blockize commits
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
<<<<<<< HEAD
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        vk = T.axis.R(16, 0)
        # These access region must be explicitly stated. Otherwise the auto-completed region starts from (0, 0) instead of (vi, vj)
        T.reads([A[vi: vi+16, vk: vk+16], B[vj: vj+16, vk: vk+16], C[vi:vi+16, vj:vj+16]])
        T.writes([C[vi: vi+16, vj: vj+16]])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + B[vjj, vkk] * A[vii, vkk]



@T.prim_func
def lower_intrin_func(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with T.block("root"):
        vi = T.axis.S(16, 0)
        vj = T.axis.S(16, 0)
        vk = T.axis.R(16, 0)
        T.reads([C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16], B[vj:vj + 16, vk:vk + 16]])
        T.writes(C[vi:vi + 16, vj:vj + 16])
        T.evaluate(T.tvm_mma_sync(C.data, C.elem_offset // 256,
                                      A.data, A.elem_offset // 256,
                                      B.data, B.elem_offset // 256,
                                      C.data, C.elem_offset // 256,
                                      dtype="handle"))


@T.prim_func
def tensorized_func(a: T.handle, b: T.handle, c: T.handle) -> None:
    # function attr dict
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
=======
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
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
    C = T.match_buffer(c, ())

    with T.block("root"):
        T.reads(C[()], A[0 : 4], B[0 : 4])
        T.writes(C[()])
        for i in range(0, 4):
            with T.block("update"):
                vi = T.axis.remap("R", [i])
                C[()] = C[()] + A[vi] * B[vi]


@T.prim_func
def dot_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), offset_factor=1)
    B = T.match_buffer(b, (4,), offset_factor=1)
    C = T.match_buffer(c, (), offset_factor=1)

    with T.block("root"):
        T.reads(C[()], A[0 : 4], B[0 : 4])
        T.writes(C[()])
        T.evaluate(
            T.call_extern(
                "vec4add",
                C.data,
                C.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int32",
            )
        )


@T.prim_func
def outer_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 1), offset_factor=1)
    B = T.match_buffer(b, (16, 1), offset_factor=1)
    C = T.match_buffer(c, (16, 16), offset_factor=1)

    with T.block("root"):
        T.reads(
            C[0 : 16, 0 : 16],
            A[0 : 16, 0 : 1],
            B[0 : 16, 0 : 1],
        )
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("update"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C[vii, vjj] + A[vii, 0] * B[vjj, 0]


@T.prim_func
def outer_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 1), offset_factor=1)
    B = T.match_buffer(b, (16, 1), offset_factor=1)
    C = T.match_buffer(c, (16, 16), offset_factor=1)

    with T.block("root"):
        T.reads(
            C[0 : 16, 0 : 16],
            A[0 : 16, 0 : 1],
            B[0 : 16, 0 : 1],
        )
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(
            T.call_extern(
                "outer_product",
                C.data,
                C.elem_offset,
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                dtype="int32",
            )
        )


@T.prim_func
def matmul(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(128, 128), "float32"],
    C: T.Buffer[(128, 128), "float32"],
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def tensorized_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)

>>>>>>> 6c33791db... squash blockize commits
    for i_outer, j_outer in T.grid(8, 8):
        for i_inner_init, j_inner_init in T.grid(16, 16):
            with T.block("init"):
                vi_init = T.axis.S(128, ((i_outer * 16) + i_inner_init))
                vj_init = T.axis.S(128, ((j_outer * 16) + j_inner_init))
                C[vi_init, vj_init] = T.float32(0)
        for k_outer in T.grid(8):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i_outer, j_outer, k_outer])
<<<<<<< HEAD
                T.reads([C[vi*16:vi*16 + 16, vj*16:vj*16 + 16], A[vi*16:vi*16 + 16, vk*16:vk*16 + 16], B[vj*16:vj*16 + 16, vk*16:vk*16 + 16]])
                T.writes(C[vi*16:vi*16 + 16, vj*16:vj*16 + 16])
                A_elem_offset = T.var('int32')
                B_elem_offset = T.var('int32')
                C_elem_offset = T.var('int32')
                A_sub = T.match_buffer(A[vi*16:vi*16+16, vk*16:vk*16+16], [16, 16], elem_offset=A_elem_offset)
                B_sub = T.match_buffer(B[vj*16:vj*16+16, vk*16:vk*16+16], [16, 16], elem_offset=B_elem_offset)
                C_sub = T.match_buffer(C[vi*16:vi*16+16, vj*16:vj*16+16], [16, 16], elem_offset=C_elem_offset)
                T.evaluate(
                    T.tvm_mma_sync(C_sub.data, T.floordiv(C_sub.elem_offset, 256),
                                     A_sub.data, T.floordiv(A_sub.elem_offset, 256),
                                     B_sub.data, T.floordiv(B_sub.elem_offset, 256),
                                     C_sub.data, T.floordiv(C_sub.elem_offset, 256),
                                     dtype="handle"))


@T.prim_func
def batch_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 128, 128])
    B = T.match_buffer(b, [16, 128, 128])
    C = T.match_buffer(c, [16, 128, 128])

=======
                T.reads(
                    [
                        C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                        A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                        B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    ]
                )
                T.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_elem_offset = T.var("int32")
                B_elem_offset = T.var("int32")
                C_elem_offset = T.var("int32")
                A_sub = T.match_buffer(
                    A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    elem_offset=A_elem_offset,
                )
                B_sub = T.match_buffer(
                    B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    elem_offset=B_elem_offset,
                )
                C_sub = T.match_buffer(
                    C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    [16, 16],
                    elem_offset=C_elem_offset,
                )
                T.evaluate(
                    T.tvm_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        dtype="handle",
                    )
                )


@T.prim_func
def batch_matmul(
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
) -> None:
>>>>>>> 6c33791db... squash blockize commits
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            C[vn, vi, vj] = T.float32(0)

    for n, i, j, k in T.grid(16, 128, 128, 128):
        with T.block("update"):
            vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
            C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@T.prim_func
<<<<<<< HEAD
def tensorized_batch_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    # function attr dict
    C = T.match_buffer(c, [16, 128, 128])
    B = T.match_buffer(b, [16, 128, 128])
    A = T.match_buffer(a, [16, 128, 128])

    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            C[vn, vi, vj] = T.float32(0)
    # body
=======
def tensorized_batch_matmul_mma(
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads()
            T.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
>>>>>>> 6c33791db... squash blockize commits
    for n in range(0, 16):
        for i, j, k in T.grid(8, 8, 8):
            with T.block("update"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
<<<<<<< HEAD
                T.reads([C[vn:vn + 1, vi*16:vi*16 + 16, vj*16:vj*16 + 16], A[vn:vn + 1, vi*16:vi*16 + 16, vk*16:vk*16 + 16],
                           B[vn:vn + 1, vj*16:vj*16 + 16, vk*16:vk*16 + 16]])
                T.writes(C[vn:vn + 1, vi*16:vi*16 + 16, vj*16:vj*16 + 16])
                A_elem_offset = T.var('int32')
                B_elem_offset = T.var('int32')
                C_elem_offset = T.var('int32')
                A_sub = T.match_buffer(A[vn:vn + 1, vi*16:vi*16+16,vk*16:vk*16+16], (16, 16), elem_offset=A_elem_offset)
                B_sub = T.match_buffer(B[vn:vn + 1, vj*16:vj*16+16,vk*16:vk*16+16], (16, 16), elem_offset=B_elem_offset)
                C_sub = T.match_buffer(C[vn:vn + 1, vi*16:vi*16+16,vj*16:vj*16+16], (16, 16), elem_offset=C_elem_offset)
                T.evaluate(
                    T.tvm_mma_sync(C_sub.data, T.floordiv(C_sub.elem_offset, 256),
                                     A_sub.data, T.floordiv(A_sub.elem_offset, 256),
                                     B_sub.data, T.floordiv(B_sub.elem_offset, 256),
                                     C_sub.data, T.floordiv(C_sub.elem_offset, 256),
                                     dtype="handle"))


@T.prim_func
def batch_matmul_dot_product(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [1, 4, 4], "float32")
    B = T.match_buffer(b, [1, 4, 4], "float32")
    C = T.match_buffer(c, [1, 4, 4], "float32")

    t = T.var("int32")
    T.attr(T.iter_var(t, None, "DataPar", ""), "pragma_import_llvm",
             "; ModuleID = '/tmp/tmpur44d1nu/input0.cc'\n\
source_filename = \"/tmp/tmpur44d1nu/input0.cc\"\n\
target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n\
target triple = \"x86_64-pc-linux-gnu\"\n\
\n\
; Function Attrs: noinline nounwind optnone uwtable\n\
define dso_local i32 @vec4add(float* %0, i32 %1, float* %2, i32 %3, float* %4, i32 %5) #0 {\n\
  %7 = alloca float*, align 8\n\
  %8 = alloca i32, align 4\n\
  %9 = alloca float*, align 8\n\
  %10 = alloca i32, align 4\n\
  %11 = alloca float*, align 8\n\
  %12 = alloca i32, align 4\n\
  %13 = alloca i32, align 4\n\
  store float* %0, float** %7, align 8\n\
  store i32 %1, i32* %8, align 4\n\
  store float* %2, float** %9, align 8\n\
  store i32 %3, i32* %10, align 4\n\
  store float* %4, float** %11, align 8\n\
  store i32 %5, i32* %12, align 4\n\
  store i32 0, i32* %13, align 4\n\
  br label %14\n\
\n\
14:                                               ; preds = %39, %6\n\
  %15 = load i32, i32* %13, align 4\n\
  %16 = icmp slt i32 %15, 4\n\
  br i1 %16, label %17, label %42\n\
\n\
17:                                               ; preds = %14\n\
  %18 = load float*, float** %9, align 8\n\
  %19 = load i32, i32* %13, align 4\n\
  %20 = load i32, i32* %10, align 4\n\
  %21 = add nsw i32 %19, %20\n\
  %22 = sext i32 %21 to i64\n\
  %23 = getelementptr inbounds float, float* %18, i64 %22\n\
  %24 = load float, float* %23, align 4\n\
  %25 = load float*, float** %11, align 8\n\
  %26 = load i32, i32* %13, align 4\n\
  %27 = load i32, i32* %12, align 4\n\
  %28 = add nsw i32 %26, %27\n\
  %29 = sext i32 %28 to i64\n\
  %30 = getelementptr inbounds float, float* %25, i64 %29\n\
  %31 = load float, float* %30, align 4\n\
  %32 = fmul float %24, %31\n\
  %33 = load float*, float** %7, align 8\n\
  %34 = load i32, i32* %8, align 4\n\
  %35 = sext i32 %34 to i64\n\
  %36 = getelementptr inbounds float, float* %33, i64 %35\n\
  %37 = load float, float* %36, align 4\n\
  %38 = fadd float %37, %32\n\
  store float %38, float* %36, align 4\n\
  br label %39\n\
\n\
39:                                               ; preds = %17\n\
  %40 = load i32, i32* %13, align 4\n\
  %41 = add nsw i32 %40, 1\n\
  store i32 %41, i32* %13, align 4\n\
  br label %14\n\
\n\
42:                                               ; preds = %14\n\
  ret i32 0\n\
}\n\
\n\
attributes #0 = { noinline nounwind optnone uwtable \"correctly-rounded-divide-sqrt-fp-math\"=\"false\" \"disable-tail-calls\"=\"false\" \"frame-pointer\"=\"all\" \"less-precise-fpmad\"=\"false\" \"min-legal-vector-width\"=\"0\" \"no-infs-fp-math\"=\"false\" \"no-jump-tables\"=\"false\" \"no-nans-fp-math\"=\"false\" \"no-signed-zeros-fp-math\"=\"false\" \"no-trapping-math\"=\"true\" \"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"x86-64\" \"target-features\"=\"+cx8,+fxsr,+mmx,+sse,+sse2,+x87\" \"unsafe-fp-math\"=\"false\" \"use-soft-float\"=\"false\" }\n\
\n\
!llvm.module.flags = !{!0}\n\
!llvm.ident = !{!1}\n\
\n\
!0 = !{i32 1, !\"wchar_size\", i32 4}\n\
!1 = !{!\"Ubuntu clang version 11.0.0-++20200928083541+eb83b551d3e-1~exp1~20200928184208.110\"}\n\
\n\
             ")

    for n, i, j in T.grid(1, 4, 4):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            C[vn, vi, vj] = T.float32(0)

    for n, i, j, k in T.grid(1, 4, 4, 4):
        with T.block("update"):
            vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
            C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


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
    A = T.match_buffer(a, (4,), offset_factor=1)
    B = T.match_buffer(b, (4,), offset_factor=1)
    C = T.match_buffer(c, (1,), offset_factor=1)

    with T.block("root"):
        v0 = T.axis.R(4, 0)
        T.reads([C[0 : 1], A[v0 : v0 + 4], B[v0 : v0 + 4]])
        T.writes([C[0 : 1]])
        T.evaluate(T.call_extern("vec4add", C.data, C.elem_offset, A.data, A.elem_offset, B.data, B.elem_offset, dtype="int32"))


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg
# fmt: on

# pylint: disable=invalid-name


tir.TensorIntrin.register("test_identity_intrin", desc_func, intrin_func)
tir.TensorIntrin.register("test_mma_intrin", desc_func, lower_intrin_func)
tir.TensorIntrin.register("test_dot_product_intrin", dot_product_desc, dot_product_impl)


def test_tensorize_gemm():
    func = matmul
    # schedule
    s = tir.Schedule(func, debug_mask="all")
    update = s.get_block("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    s.tensorize(ii, "test_identity_intrin")

    func = tvm.build(s.mod["main"])
    a_np = np.random.uniform(size=(128, 128)).astype("float32")
    b_np = np.random.uniform(size=(128, 128)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((128, 128)).astype("float32"))
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), np.dot(a_np, b_np.transpose()), rtol=1e-6)


def test_tensorize_buffer_bind():
=======
                T.reads(
                    C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    A[vn : vn + 1, vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    B[vn : vn + 1, vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                )
                T.writes(C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_elem_offset = T.var("int32")
                B_elem_offset = T.var("int32")
                C_elem_offset = T.var("int32")
                A_sub = T.match_buffer(
                    A[vn : vn + 1, vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    (16, 16),
                    elem_offset=A_elem_offset,
                )
                B_sub = T.match_buffer(
                    B[vn : vn + 1, vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    (16, 16),
                    elem_offset=B_elem_offset,
                )
                C_sub = T.match_buffer(
                    C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    (16, 16),
                    elem_offset=C_elem_offset,
                )
                T.evaluate(
                    T.tvm_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub.elem_offset, 256),
                        dtype="handle",
                    )
                )


@T.prim_func
def tensorized_batch_matmul_dot_product(
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads()
            T.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n, i, j, k_0 in T.grid(16, 128, 128, 32):
        with T.block("blockized_update"):
            vn, vi, vj, vko = T.axis.remap("SSSR", [n, i, j, k_0])
            T.reads(
                C[vn, vi, vj], A[vn, vi, vko * 4 : vko * 4 + 4], B[vn, vj, vko * 4 : vko * 4 + 4]
            )
            T.writes(C[vn, vi, vj])
            A_1 = T.match_buffer(
                A[vn, vi, vko * 4 : vko * 4 + 4], [4], dtype="float32", offset_factor=1
            )
            B_1 = T.match_buffer(
                B[vn, vj, vko * 4 : vko * 4 + 4], [4], dtype="float32", offset_factor=1
            )
            C_1 = T.match_buffer(C[vn, vi, vj], [], dtype="float32", offset_factor=1)
            T.evaluate(
                T.call_extern(
                    "vec4add",
                    C_1.data,
                    C_1.elem_offset,
                    A_1.data,
                    A_1.elem_offset,
                    B_1.data,
                    B_1.elem_offset,
                    dtype="int32",
                )
            )


@T.prim_func
def tensorized_batch_matmul_outer_product(
    A: T.Buffer[(16, 128, 128), "float32"],
    B: T.Buffer[(16, 128, 128), "float32"],
    C: T.Buffer[(16, 128, 128), "float32"],
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with T.block("init"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads()
            T.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n, i_0, j_0, k in T.grid(16, 8, 8, 128):
        with T.block("blockized_update"):
            vn, vio, vjo, vk = T.axis.remap("SSSR", [n, i_0, j_0, k])
            T.reads(
                C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                A[vn, vio * 16 : vio * 16 + 16, vk],
                B[vn, vjo * 16 : vjo * 16 + 16, vk],
            )
            T.writes(C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            A_1 = T.match_buffer(A[vn, vio * 16 : vio * 16 + 16, vk], [16, 1], dtype="float32", offset_factor=1)
            B_1 = T.match_buffer(B[vn, vjo * 16 : vjo * 16 + 16, vk], [16, 1], dtype="float32", offset_factor=1
            )
            C_1 = T.match_buffer(
                C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16], [16, 16], dtype="float32", offset_factor=1
            )
            T.evaluate(
                T.call_extern("outer_product", C_1.data, C_1.elem_offset, A_1.data, A_1.elem_offset,
                              B_1.data, B_1.elem_offset, dtype="int32"
                )
            )


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)
tir.TensorIntrin.register("test_dot_product_intrin", dot_product_desc, dot_product_intrin)
tir.TensorIntrin.register("test_outer_product_intrin", outer_product_desc, outer_product_intrin)


def test_tensorize_matmul():
>>>>>>> 6c33791db... squash blockize commits
    func = matmul
    # schedule
    s = tir.Schedule(func, debug_mask="all")
    update = s.get_block("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    s.tensorize(ii, "test_mma_intrin")
<<<<<<< HEAD
    tvm.ir.assert_structural_equal(tensorized_func, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_high_dim_tensorize():
=======
    tvm.ir.assert_structural_equal(tensorized_matmul, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_tensorize_batch_matmul():
>>>>>>> 6c33791db... squash blockize commits
    func = batch_matmul
    s = tir.Schedule(func, debug_mask="all")
    update = s.get_block("update")
    _, i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.tensorize(ii, "test_mma_intrin")
<<<<<<< HEAD
    tvm.ir.assert_structural_equal(tensorized_batch_matmul, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=batch_matmul)


@pytest.mark.skip("failed")
def test_tensorize_dot_product():
    func = batch_matmul_dot_productt
=======
    tvm.ir.assert_structural_equal(tensorized_batch_matmul_mma, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=batch_matmul)


def test_tensorize_dot_product():
    func = batch_matmul
>>>>>>> 6c33791db... squash blockize commits
    s = tir.Schedule(func, debug_mask="all")
    C = s.get_block("update")
    _, _, _, k = s.get_loops(C)
    _, ki = s.split(k, factors=[None, 4])
    s.tensorize(ki, "test_dot_product_intrin")
<<<<<<< HEAD
    target = "llvm"
    ctx = tvm.device(target, 0)
    a_np = np.random.uniform(size=(1, 4, 4)).astype("float32")
    b_np = np.random.uniform(size=(1, 4, 4)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((1, 4, 4), dtype="float32"), ctx)
    func = tvm.build(s.mod["main"], target=target)
    func(a, b, c)
    tvm.testing.assert_allclose(
        c.numpy(),
        np.matmul(a.numpy(), b.numpy().transpose(0, 2, 1)),
        rtol=1e-5,
    )
=======
    tvm.ir.assert_structural_equal(tensorized_batch_matmul_dot_product, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_tensorize_outer_product():
    func = batch_matmul
    s = tir.Schedule(func, debug_mask="all")
    C = s.get_block("update")
    _, i, j, k = s.get_loops(C)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    s.reorder(io, jo, k, ii, ji)
    s.tensorize(ii, "test_outer_product_intrin")
    tvm.ir.assert_structural_equal(tensorized_batch_matmul_outer_product, s.mod["main"])
>>>>>>> 6c33791db... squash blockize commits
    verify_trace_roundtrip(sch=s, mod=func)


if __name__ == "__main__":
<<<<<<< HEAD
    test_tensorize_gemm()
    test_tensorize_buffer_bind()
    test_high_dim_tensorize()
    # test_tensorize_dot_product()
=======
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
>>>>>>> 6c33791db... squash blockize commits
