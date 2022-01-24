<<<<<<< HEAD
import sys
import pytest
import tvm
from tvm import tir, te
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip


@T.prim_func
def elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))
=======
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
import sys
import pytest
import tvm
from tvm.script import tir as T
from tvm import tir
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@T.prim_func
def single_elementwise(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
>>>>>>> 6c33791db... squash blockize commits
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
<<<<<<< HEAD
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def blockize(a: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, (128, 128), "float32")
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(8, 8):
        with T.block("blockized_B"):
            vi, vj = T.axis.remap("SS", [i, j])
            for ii, jj in T.grid(16, 16):
                with T.block("B"):
                    vii = T.axis.S(128, vi * 16 + ii)
                    vjj = T.axis.S(128, vj * 16 + jj)
                    B[vii, vjj] = A[vii, vjj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
=======


@T.prim_func
def single_elementwise_blockized1(
    A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]
) -> None:
    with T.block("blockized_B"):
        vio = T.axis.spatial(1, 0)
        vjo = T.axis.spatial(1, 0)
        T.reads(A[0:128, 0:128])
        T.writes(B[0:128, 0:128])
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * T.float32(2)


@T.prim_func
def single_elementwise_blockized2(
    A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]
) -> None:
    for i in T.serial(128):
        with T.block("blockized_B"):
            vi = T.axis.spatial(128, i)
            vjo = T.axis.spatial(1, 0)
            T.reads(A[vi, 0:128])
            T.writes(B[vi, 0:128])
            for j in T.serial(128):
                with T.block("B"):
                    vj = T.axis.remap("S", [j])
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * T.float32(2)


@T.prim_func
def two_elementwise(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer([128, 128], dtype="float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj])
            T.writes(C[vi, vj])
>>>>>>> 6c33791db... squash blockize commits
            C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
<<<<<<< HEAD
def blockize_schedule_1(a: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with T.block("root"):
        T.reads([])
        T.writes([])
        B = T.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
        for i0_outer in range(0, 8):
            for i1_outer in range(0, 8):
                with T.block("blockized_B"):
                    vio = T.axis.S(8, i0_outer)
                    vjo = T.axis.S(8, i1_outer)
                    T.reads([A[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    T.writes([B[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    for i0_inner in range(0, 16):
                        for i1_inner in range(0, 16):
                            with T.block("B"):
                                vi = T.axis.S(128, ((vio * 16) + i0_inner))
                                vj = T.axis.S(128, ((vjo * 16) + i1_inner))
                                T.reads([A[vi : (vi + 1), vj : (vj + 1)]])
                                T.writes([B[vi : (vi + 1), vj : (vj + 1)]])
                                B[vi, vj] = A[vi, vj] * T.float32(2)
                with T.block("blockized_C"):
                    vio = T.axis.S(8, i0_outer)
                    vjo = T.axis.S(8, i1_outer)
                    T.reads([B[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    T.writes([C[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    for ax0 in range(0, 16):
                        for ax1 in range(0, 16):
                            with T.block("C"):
                                vi = T.axis.S(128, ((vio * 16) + ax0))
                                vj = T.axis.S(128, ((vjo * 16) + ax1))
                                T.reads([B[vi : (vi + 1), vj : (vj + 1)]])
                                T.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                                C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
def blockize_schedule_2(a: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with T.block("root"):
        T.reads([])
        T.writes([])
        B = T.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
        for i0_outer in range(0, 4):
            for i1_outer in range(0, 4):
                for ax0 in range(0, 2):
                    for ax1 in range(0, 2):
                        with T.block("blockized_B"):
                            vio = T.axis.S(8, ((i0_outer * 2) + ax0))
                            vjo = T.axis.S(8, ((i1_outer * 2) + ax1))
                            T.reads(
                                [A[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]]
                            )
                            T.writes(
                                [B[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]]
                            )
                            for i0_inner in range(0, 16):
                                for i1_inner in range(0, 16):
                                    with T.block("B"):
                                        vi = T.axis.S(128, ((vio * 16) + i0_inner))
                                        vj = T.axis.S(128, ((vjo * 16) + i1_inner))
                                        T.reads([A[vi : (vi + 1), vj : (vj + 1)]])
                                        T.writes([B[vi : (vi + 1), vj : (vj + 1)]])
                                        B[vi, vj] = A[vi, vj] * T.float32(2)
                for i0_inner_1 in range(0, 32):
                    for i1_inner_1 in range(0, 32):
                        with T.block("C"):
                            vi = T.axis.S(128, ((i0_outer * 32) + i0_inner_1))
                            vj = T.axis.S(128, ((i1_outer * 32) + i1_inner_1))
                            T.reads([B[vi : (vi + 1), vj : (vj + 1)]])
                            T.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                            C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
def rowsum(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(
        b,
        [
            128,
        ],
    )
=======
def two_elementwise_blockized(
    A: T.Buffer[(128, 128), "float32"],
    C: T.Buffer[(128, 128), "float32"]
) -> None:
    B = T.alloc_buffer([128, 128], dtype="float32")
    for i_0, j_0 in T.grid(8, 8):
        with T.block("blockized_B"):
            vio, vjo = T.axis.remap("SS", [i_0, j_0])
            T.reads(A[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            T.writes(B[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            for i_1, j_1 in T.grid(16, 16):
                with T.block("B"):
                    vi, vj = T.axis.remap("SS", [i_1, j_1])
                    T.reads(A[vio * 16 + vi, vjo * 16 + vj])
                    T.writes(B[vio * 16 + vi, vjo * 16 + vj])
                    B[vio * 16 + vi, vjo * 16 + vj] = A[vio * 16 + vi, vjo * 16 + vj] * T.float32(2)
        with T.block("blockized_C"):
            vio, vjo = T.axis.remap("SS", [i_0, j_0])
            T.reads(B[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            T.writes(C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [ax0, ax1])
                    T.reads(B[vio * 16 + vi, vjo * 16 + vj])
                    T.writes(C[vio * 16 + vi, vjo * 16 + vj])
                    C[vio * 16 + vi, vjo * 16 + vj] = B[vio * 16 + vi, vjo * 16 + vj] + T.float32(1)


@T.prim_func
def rowsum(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128,), "float32"]) -> None:
>>>>>>> 6c33791db... squash blockize commits
    for k, i in T.grid(128, 128):
        with T.block("B"):
            vk, vi = T.axis.remap("RS", [k, i])
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
<<<<<<< HEAD
def rowsum_blockized(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128])
=======
def rowsum_blockized(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128,), "float32"]) -> None:
>>>>>>> 6c33791db... squash blockize commits
    with T.block("blockized_B"):
        vko = T.axis.R(1, 0)
        vio = T.axis.S(1, 0)
        with T.init():
            for i1 in T.serial(0, 128):
                with T.block("B_init"):
                    vi_init = T.axis.S(128, i1)
                    B[vi_init] = T.float32(0)
        for i0, i1_1 in T.grid(128, 128):
            with T.block("B"):
                vk, vi = T.axis.remap("RS", [i0, i1_1])
                B[vi] = B[vi] + A[vi, vk]


<<<<<<< HEAD
def test_blockize():
    func = elementwise
    # schedule
    s = tir.Schedule(func, debug_mask="all")
    B = s.get_block("B")
    _ = s.get_block("C")
    x, y = s.get_loops(B)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    s.blockize(xi)
    tvm.ir.assert_structural_equal(blockize, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_schedule():
    func = elementwise
    # test 1
=======
# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

def test_blockize_outer():
    func = single_elementwise
    # schedule
    s = tir.Schedule(func, debug_mask="all")
    B = s.get_block("B")
    x, y = s.get_loops(B)
    s.blockize(x)
    print(s.mod['main'].script())
    tvm.ir.assert_structural_equal(s.mod["main"], single_elementwise_blockized1)
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_inner():
    func = single_elementwise
    # schedule
    s = tir.Schedule(func, debug_mask="all")
    B = s.get_block("B")
    x, y = s.get_loops(B)
    s.blockize(y)
    tvm.ir.assert_structural_equal(s.mod["main"], single_elementwise_blockized2)
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_reverse_compute_at():
    func = two_elementwise
>>>>>>> 6c33791db... squash blockize commits
    s = tir.Schedule(func, debug_mask="all")
    B = s.get_block("B")
    C = s.get_block("C")
    x, y = s.get_loops(B)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    s.blockize(xi)
    s.reverse_compute_at(C, yo)
    s.blockize(s.get_loops(C)[-2])
<<<<<<< HEAD
    tvm.ir.assert_structural_equal(s.mod["main"], blockize_schedule_1)
    verify_trace_roundtrip(sch=s, mod=func)
    # test 2
=======
    tvm.ir.assert_structural_equal(s.mod["main"], two_elementwise_blockized)
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_compute_at():
    func = two_elementwise
>>>>>>> 6c33791db... squash blockize commits
    s = tir.Schedule(func, debug_mask="all")
    B = s.get_block("B")
    C = s.get_block("C")
    x, y = s.get_loops(C)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    s.blockize(xi)
    s.compute_at(B, yo)
    s.blockize(s.get_loops(B)[-2])
<<<<<<< HEAD
    tvm.ir.assert_structural_equal(s.mod["main"], blockize_schedule_1)
    verify_trace_roundtrip(sch=s, mod=func)
    # test 3
    s = tir.Schedule(func, debug_mask="all")
    B = s.get_block("B")
    C = s.get_block("C")
    x, y = s.get_loops(B)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    b_outer = s.blockize(xi)
    xC, yC = s.get_loops(C)
    xCo, xCi = s.split(xC, factors=[None, 32])
    yCo, yCi = s.split(yC, factors=[None, 32])
    s.reorder(xCo, yCo, xCi, yCi)
    s.compute_at(b_outer, yCo)
    tvm.ir.assert_structural_equal(s.mod["main"], blockize_schedule_2)
=======
    tvm.ir.assert_structural_equal(s.mod["main"], two_elementwise_blockized)
>>>>>>> 6c33791db... squash blockize commits
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_init_loops():
    s = tir.Schedule(rowsum, debug_mask="all")
    k, _ = s.get_loops(s.get_block("B"))
    s.blockize(k)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_blockized)
    verify_trace_roundtrip(sch=s, mod=rowsum)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
