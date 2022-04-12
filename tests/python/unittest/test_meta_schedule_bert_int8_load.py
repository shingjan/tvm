import pytest
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.te import create_prim_func
from tvm.relay import build as relay_build
from tvm.meta_schedule.testing import te_workload
from tvm.target import Target
from tvm.meta_schedule import schedule_rule, ReplayTraceConfig, postproc, tune_tir, tune_relay
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord
from tvm.meta_schedule.tune import extract_task_from_relay, tune_extracted_tasks
from tvm.meta_schedule.utils import derived_object
from tvm import tir, te, relay
import tvm.relay.testing
from tvm.ir import IRModule
from tvm.relay import transform
from tvm.meta_schedule.tune import Parse, ApplyHistoryBest
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
import tvm
import numpy as np
import onnx
import tempfile
from typing import Tuple, List


@derived_object
class DummyDatabase(PyDatabase):
    def __init__(self):
        super().__init__()
        self.records = []
        self.workload_reg = []

    def has_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return True
        return False

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.records.append(record)

    def commit_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return workload
        workload = Workload(mod)
        self.workload_reg.append(workload)
        return workload

    def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
        return list(
            filter(
                lambda x: x.workload == workload,
                sorted(self.records, key=lambda x: sum(x.run_secs) / len(x.run_secs)),
            )
        )[: int(top_k)]

    def __len__(self) -> int:
        return len(self.records)

    def print_results(self) -> None:
        print("\n".join([str(r) for r in self.records]))


def test_bert_int8():

    name = "models/bert-base-qat.onnx"

    onnx_model = onnx.load(name)
    batch_size = 1
    seq_len = 384

    shape_dict = {
        "input_ids": (batch_size, seq_len),
        "segment_ids": (batch_size, seq_len),
        "input_mask": (batch_size, seq_len),
    }

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.SimplifyInference(),
            transform.FoldScaleAxis(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    mod = tvm.relay.transform.FakeQuantizationToInteger(use_qat=True)(mod)
    json_path = "models/bert_base_int8.json"
    params_path = "models/bert_base_int8.params"

    with open(json_path, "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open(params_path, "wb") as fo:
        fo.write(relay.save_param_dict(params))


def test_load_bert_int8():
    json_path = "models/bert_base_int8.json"
    params_path = "models/bert_base_int8.params"
    target = Target("nvidia/geforce-rtx-3070")

    batch_size = 1
    seq_len = 384

    shape_dict = {
        "input_ids": (batch_size, seq_len),
        "segment_ids": (batch_size, seq_len),
        "input_mask": (batch_size, seq_len),
    }

    with open(json_path, "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open(params_path, "rb") as fi:
        params = relay.load_param_dict(fi.read())

    # extracted_tasks = extract_task_from_relay(mod, target, params)
    # tune_tasks = list(
    #     filter(
    #         lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
    #         extracted_tasks,
    #     )
    # )

    extracted_tasks = extract_task_from_relay(mod, target, params)
    # for n in range(len(extracted_tasks)):
    #     task = extracted_tasks[n]
    #     if task.task_name == "fused_nn_batch_matmul_1":
    #         print(task.dispatched[0].script())
    #         break
    tune_tasks = []

    # for task in extracted_tasks:
    #     if task.task_name == "fused_nn_batch_matmul_1":
    #         print(task.mod)
    #         for mod in task.dispatched:
    #             print(mod["fused_nn.batch_matmul"].script())

    for task in filter(
        lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
        extracted_tasks,
    ):
        relay_func = list(task.mod.functions.values())[0]
        out_type = relay_func.body.checked_type

        if out_type.dtype != "float32":
            tune_tasks.append(task)

    with tempfile.TemporaryDirectory() as work_dir:
        target = Target(target)
        database = DummyDatabase()
        database_rt = tune_extracted_tasks(
            extracted_tasks=tune_tasks,
            target=target,
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                num_trials_total=32,
            ),
            work_dir=work_dir,
            database=database,
        )
    with ApplyHistoryBest(database_rt):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            relay_lib = relay_build(mod, target=target, params=params)
    dev = tvm.device("cuda", 0)
    runtime = tvm.contrib.graph_executor.GraphModule(relay_lib["default"](dev))

    inputs = []

    for name, shape in shape_dict.items():
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        runtime.set_input(name, arr)
        inputs.append(arr)

    print(runtime.benchmark(dev, number=1, repeat=50).mean)


def extract_task_qbert():
    mod, params, _ = load_quantized_bert_base(batch_size=1, seq_len=128)
    target = Target("nvidia/geforce-rtx-3070")
    extracted_tasks = extract_task_from_relay(mod, target, params)
    for n in range(len(extracted_tasks)):
        task = extracted_tasks[n]
        if task.task_name == "fused_nn_batch_matmul_1":
            print(task.dispatched[0].script())
            break
    tune_tasks = list(
        filter(
            lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
            extracted_tasks,
        )
    )
    # three int8 dense, two int8 bmm, and one fp32 dense
    assert len(tune_tasks) == 6

    for task in tune_tasks:
        relay_func = list(task.mod.functions.values())[0]
        out_type = relay_func.body.checked_type

        if out_type.dtype == "float32":
            continue

        mod = Parse._mod(task.dispatched[0])
        sch = tvm.tir.Schedule(mod)
        # print(sch.mod.script())


if __name__ == "__main__":
    # test_manual_matmul()
    # test_tune_matmul()
    # test_bert_int8()
    test_load_bert_int8()
    # extract_task_qbert()
