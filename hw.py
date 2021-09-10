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
import tvm
import os
from tvm.script import ty
from tvm import te, tir
import numpy as np
from tvm.contrib.tar import tar
import tvm.testing


def _check_workload(te_workload, tir_workload):
    func = te.create_prim_func(te_workload())
    tvm.ir.assert_structural_equal(func, tir_workload)
    # make sure that we can create schedule from the func
    s = tir.Schedule(func, debug_mask="all")
    assert s


def te_conv2d():
    batch = 16
    in_channel = 16
    out_channel = 32
    size = 14
    kernel = 3

    A = te.placeholder((batch, in_channel, size, size), name="A")
    W = te.placeholder((in_channel, kernel, kernel, out_channel), name="W")
    Apad = te.compute(
        (batch, in_channel, size + 2, size + 2),
        lambda nn, cc, yy, xx: tvm.tir.if_then_else(
            tvm.tir.all(yy >= 1, yy - 1 < size, xx >= 1, xx - 1 < size),
            A[nn, cc, yy - 1, xx - 1],
            0.0,
        ),
        name="Apad",
    )
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel), name="ry")
    rx = te.reduce_axis((0, kernel), name="rx")
    B = te.compute(
        (batch, out_channel, size, size),
        lambda nn, ff, yy, xx: te.sum(
            Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff], axis=[rc, ry, rx]
        ),
        name="B",
    )
    return [A, W, B]


def run():
    func = te.create_prim_func(te_conv2d())
    print(tvm.script.asscript(func))
    target = "llvm"
    built_func = tvm.build(func, target=target)
    load_path = os.path.join("./tmp/", "hw." + tar.output_format)
    mod = tvm.build(func, target=target)
    mod.export_library(load_path, tar)
    mod = tvm.runtime.load_module(load_path)


if __name__ == "__main__":
    run()
