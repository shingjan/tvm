import os
import torch
import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
import pickle

target = tvm.target.Target("nvidia/geforce-rtx-3070")
dtype = "int8"

log_file = "logs/bert.log"


tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 50,
    "early_stopping": 250,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        tuner_obj = XGBTuner(tsk, loss_type="rank")

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


batch_size = 1
seq_len = 384
mod, params, input_info = load_quantized_bert_base(batch_size, seq_len)


def tune(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")

    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    print("num_tasks", len(tasks))
    new_tasks = []
    history = autotvm.record.ApplyHistoryBest(log_file)
    for task in tasks:
        if history._query_inside(task.target, task.workload) is None:
            new_tasks.append(task)

    print("num_new tasks", len(new_tasks))
    pickle.dump(tasks, open("task.pkl", "wb"))
    return

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)


def evaluate():
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = runtime.GraphModule(lib["default"](dev))

    inputs = []

    for name, shape in input_info:
        arr = torch.randint(high=100, size=(batch_size, seq_len), dtype=torch.int64)
        module.set_input(name, arr)
        inputs.append(arr)
    module.run()

    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=1, repeat=50))


tasks = pickle.load(open("task.pkl", "rb"))
# print(tasks[3])
tune_tasks(tasks, **tuning_option)
# tune(tuning_option)
evaluate()
