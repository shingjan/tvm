import os
import sys
import torch
import tvm
from tvm import relay, autotvm
from tvm import meta_schedule as ms
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime

# from bert_rewrite import rewrite_reshape_gelu

import logging
import tempfile
import pytest
import numpy as np
from typing import Tuple, List

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.runtime.ndarray import cpu, cuda
from tvm.target.target import Target
from tvm.contrib import graph_executor
from tvm.meta_schedule import ReplayTraceConfig
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord, JSONDatabase
from tvm.meta_schedule.tune import tune_relay, tune_extracted_tasks
from tvm.meta_schedule.integration import ApplyHistoryBest
from tvm.meta_schedule import schedule_rule as M
from tvm.meta_schedule import postproc
from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.testing import DummyDatabase
from tvm.meta_schedule.testing.schedule_rule import (
    multi_level_tiling_memhammer,
    multi_level_tiling_memhammer_tensor_core,
)
from tvm.script import tir as T
from tvm import tir

import pickle

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def wmma_sync_desc_int8(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "int8", align=128, offset_factor=1, scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "int8", align=128, offset_factor=1, scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "int32", align=128, offset_factor=1, scope="wmma.accumulator")

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], "int32") * T.cast(
                    B[vjj, vkk], "int32"
                )


@T.prim_func
def wmma_sync_impl_int8(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "int8", align=128, offset_factor=16, scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "int8", align=128, offset_factor=16, scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "int32", align=128, offset_factor=16, scope="wmma.accumulator")

    with T.block("root"):
        T.reads(
            [
                C[0:16, 0:16],
                A[0:16, 0:16],
                B[0:16, 0:16],
            ]
        )
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.data,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                B.data,
                B.elem_offset // 256 + T.floordiv(T.floormod(B.elem_offset, 256), 16),
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                dtype="handle",
            )
        )


@T.prim_func
def wmma_load_a_desc_int8(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "int8", align=128, offset_factor=16, scope="shared")
    C = T.match_buffer(c, (16, 16), "int8", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl_int8(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "int8", align=128, offset_factor=16, scope="shared", strides=[s1, s0]
    )
    C = T.match_buffer(c, (16, 16), "int8", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "row_major",
                dtype="handle",
            )
        )


@T.prim_func
def wmma_load_b_desc_int8(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "int8", align=128, offset_factor=16, scope="shared")
    C = T.match_buffer(c, (16, 16), "int8", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_b_impl_int8(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "int8", align=128, offset_factor=16, scope="shared", strides=[s1, s0]
    )
    C = T.match_buffer(c, (16, 16), "int8", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "col_major",
                dtype="handle",
            )
        )


@T.prim_func
def wmma_fill_desc_int8(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "int32", align=128, offset_factor=16, scope="wmma.accumulator")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.int32(0)


@T.prim_func
def wmma_fill_impl_int8(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "int32", align=128, offset_factor=16, scope="wmma.accumulator")
    with T.block("root"):
        T.reads([])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_fill_fragment(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                T.int32(0),
                dtype="handle",
            )
        )


@T.prim_func
def wmma_store_desc_int8(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "int32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16), "int32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_store_impl_int8(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "int32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(
        c, (16, 16), "int32", align=128, offset_factor=16, scope="global", strides=[s1, s0]
    )
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_store_matrix_sync(
                A.data,
                16,
                16,
                16,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                C.access_ptr("w"),
                s1,
                "row_major",
                dtype="handle",
            )
        )


tir.TensorIntrin.register("wmma_load_a", wmma_load_a_desc_int8, wmma_load_a_impl_int8)
tir.TensorIntrin.register("wmma_load_b", wmma_load_b_desc_int8, wmma_load_b_impl_int8)
tir.TensorIntrin.register("wmma_sync", wmma_sync_desc_int8, wmma_sync_impl_int8)
tir.TensorIntrin.register("wmma_fill", wmma_fill_desc_int8, wmma_fill_impl_int8)
tir.TensorIntrin.register("wmma_store", wmma_store_desc_int8, wmma_store_impl_int8)


log_file = "logs/meta_schedule_bert.log"
json_path = "models/bert_base_int8.json"
params_path = "models/bert_base_int8.params"
with open(json_path, "r") as fi:
    mod = tvm.ir.load_json(fi.read())

with open(params_path, "rb") as fi:
    params = relay.load_param_dict(fi.read())

# mod = rewrite_reshape_gelu(mod)

target = tvm.target.Target("nvidia/geforce-rtx-3070")


def build_relay(database):
    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            return tvm.relay.build(mod, target=target, params=params)


def tune():
    # extract workloads from relay program
    # print("Extract tasks...")

    # tasks = ms.integration.extract_task_from_relay(mod, target=target, params=params)

    # pickle.dump(tasks, open("task.pkl", "wb"))
    # # run tuning tasks
    # print("Tuning...")
    tasks = pickle.load(open("task.pkl", "rb"))
    print("num_tasks", len(tasks))
    i = 0
    memhammer_select_task = [6, 10, 12, 15, 16, 19, 20]
    tune_tasks = []
    for tsk in tasks:
        if "dense" in tsk.task_name or "batch_matmul" in tsk.task_name:
            # if i in memhammer_select_task:
            print("task {}".format(i + 1) + tsk.task_name)
            print(tsk.mod)
            relay_func = list(tsk.mod.functions.values())[0]
            out_type = relay_func.body.checked_type

            if out_type.dtype != "float32":
                tune_tasks.append(tsk)
        i += 1
        # print(tsk.mod.script())

    def sch_rules_no_tensor_core():
        return [
            M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.x"],
                use_tensor_core=False,
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 4, 8],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=M.ReuseType(
                    req="must",
                    levels=[3],
                    scope="local",
                ),
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]

    def sch_rules_tensor_core():
        return [
            # multi_level_tiling_memhammer_tensor_core(target=target),
            M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
                use_tensor_core=True,
                max_innermost_factor=4,
                vector_load_lens=[1, 2, 4, 8],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=M.ReuseType(
                    req="no",
                    levels=[],
                    scope="",
                ),
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]

    def sch_rules_tensor_core_memhammer():
        return [
            multi_level_tiling_memhammer_tensor_core(target=target),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]

    def postprocs_no_tensor_core():
        return [
            postproc.RewriteCooperativeFetch(),
            postproc.RewriteUnboundBlock(),
            postproc.RewriteParallelVectorizeUnroll(),
            postproc.RewriteReductionBlock(),
            postproc.VerifyGPUCode(),
        ]

    def postprocs_tensor_core():
        return [
            postproc.RewriteCooperativeFetch(),
            # postproc.RewriteUnboundBlock(),
            postproc.RewriteParallelVectorizeUnroll(),
            postproc.RewriteReductionBlock(),
            postproc.RewriteTensorCore(),
            postproc.VerifyGPUCode(),
        ]

    with tempfile.TemporaryDirectory() as work_dir:
        database = DummyDatabase()
        # database = JSONDatabase("db/workload.json", "db/record.json")
        # for task in tasks:
        #     print(task)
        #     print(task.mod)
        #     assert database.has_workload(task.mod)
        # sys.exit(0)
        # tasks = [tasks[15]]
        memhammer_tasks = [
            tasks[6],
            tasks[10],
            tasks[12],
            tasks[15],
            tasks[16],
            tasks[19],
            tasks[20],
        ]
        tune_extracted_tasks(
            tune_tasks,
            target=target,
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                num_trials_total=32,
            ),
            sch_rules=sch_rules_tensor_core,
            postprocs=postprocs_tensor_core,
            work_dir=work_dir,
            database=database,
        )
        # tune_extracted_tasks(
        #     memhammer_tasks,
        #     target=target,
        #     config=ReplayTraceConfig(
        #         num_trials_per_iter=32,
        #         num_trials_total=32,
        #     ),
        #     sch_rules=sch_rules_tensor_core_memhammer,
        #     postprocs=postprocs_tensor_core,
        #     work_dir=work_dir,
        #     database=database,
        # )

        return build_relay(database)

        # rt_mod: tvm.module = tune_relay(
        #     mod=mod,
        #     params=params,
        #     target=target,
        #     config=ReplayTraceConfig(
        #         num_trials_per_iter=32,
        #         num_trials_total=1000,
        #     ),
        #     sch_rules=sch_rules,
        #     postprocs=postprocs,
        #     work_dir=work_dir,
        #     database=database,
        # )
        # return rt_mod


def evaluate(lib):
    dev = tvm.device(str(target), 0)
    module = runtime.GraphModule(lib["default"](dev))

    batch_size = 1
    seq_len = 384

    input_info = [
        ("input_ids", (batch_size, seq_len)),
        ("segment_ids", (batch_size, seq_len)),
        ("input_mask", (batch_size, seq_len)),
    ]
    inputs = []
    for name, shape in input_info:
        arr = np.random.uniform(1, 100, size=shape).astype("int64")
        module.set_input(name, arr)
        inputs.append(arr)

    module.run()

    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=1, repeat=50))


lib = tune()
evaluate(lib)