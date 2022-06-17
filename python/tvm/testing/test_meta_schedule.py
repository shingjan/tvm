"""Test Script"""
import logging
import tempfile
import sys
from os import path as osp
import onnx
import yaml
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.meta_schedule import TuneConfig
from tvm.meta_schedule.database import JSONDatabase
from tvm.meta_schedule.tune import tune_relay
from tvm.target.target import Target
from tvm.runtime.vm import VirtualMachine
from tvm.meta_schedule.testing.relay_workload import get_onnx_model


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)


def get_output(lib, dev, datas):
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(**datas)
    module.run()
    return module.get_output(0).numpy()


def get_relay_output(mod, params, target, dev, datas, input_dtype):
    print("Starting to build with relay.", flush=True)

    # Compile without meta-scheduler for correctness check
    try:

        with tvm.transform.PassContext(
            opt_level=3,
        ):
            rt_mod2 = relay.build(mod, target=target, params=params)
        expected_output = get_output(rt_mod2, dev, datas)
    except:
        with tvm.transform.PassContext(opt_level=3):
            vm_exec = relay.vm.compile(mod, target=target, params=params)

        vm = VirtualMachine(vm_exec, dev)
        vm.set_input("main", **datas)
        vm_output = vm.run()
        if isinstance(vm_output, tvm.runtime.container.ADT):
            expected_output = vmobj_to_list(vm_output, input_dtype)[0]  # only get the first result
        else:
            expected_output = vm_output.numpy()
    return expected_output


def vmobj_to_list(o, dtype):
    if isinstance(o, tvm.nd.NDArray):
        return [o]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f, dtype))
        return result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def get_relay_meta_schedule_output(mod, params, target, dev, datas, input_dtype):
    logger.info("Starting to tune with meta schedule.")
    with "/home/yj/tvm/work_dir" as work_dir:
        vm_exec = tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=20000,
            ),
            work_dir=work_dir,
            database=JSONDatabase(
                osp.join(work_dir, "workload.json"), osp.join(work_dir, "records.json")
            ),
        )
        logger.info("Finished tuning with meta schedule.")
        vm = VirtualMachine(vm_exec, dev)
        vm.set_input("main", **datas)
        vm_output = vm.run()
        if isinstance(vm_output, tvm.runtime.container.ADT):
            expected_output = vmobj_to_list(vm_output, input_dtype)[0]  # only get the first result
        else:
            expected_output = vm_output.numpy()
        return expected_output


def run_single_model(model_name, target_str, model_dir):
    sys.stdout = open("/home/yj/tvm/logs/func/" + model_name + "_" + target_str + ".log", "w+")
    sys.stderr = sys.stdout
    target = (
        Target("llvm --num-cores=8") if target_str == "llvm" else Target("nvidia/geforce-rtx-3070")
    )
    dev = tvm.cpu() if str(target.kind) == "llvm" else tvm.cuda()
    mod, params, (input_info, input_dtype) = get_onnx_model(
        model_name,
        cache_dir=model_dir,
    )
    datas = {}

    for input_name, input_shape in input_info.items():
        if "int" not in input_dtype:
            data = tvm.nd.array(np.random.randn(*input_shape).astype(input_dtype), dev)
        elif model_name == "bert":
            data = tvm.nd.array(
                np.random.randint(0, 30521, size=input_shape).astype(input_dtype), dev
            )
        elif model_name == "gpt2":
            data = tvm.nd.array(
                np.random.randint(0, 50256, size=input_shape).astype(input_dtype), dev
            )
        else:
            data = tvm.nd.array(np.random.randn(*input_shape).astype(input_dtype), dev)
    datas[input_name] = data

    # Check correctness
    try:
        expected_output = get_relay_output(mod, params, target, dev, datas, input_dtype)
        actual_output = get_relay_meta_schedule_output(mod, params, target, dev, datas, input_dtype)
        if np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4):
            print("The result is correct!")
        else:
            print(actual_output)
            print(expected_output)
            print(actual_output.shape)
            print(expected_output.shape)
            print("not close")
    except Exception as ex:
        print(ex)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def run_all_models():
    model_dir = "/home/yj/models/"
    docs = yaml.safe_load(open(model_dir + "models.yaml", "r"))
    need = [
        # "bert_large_v1_1_fake_quant",
        # "fbnet-a",
        # "hf-distilbert-base-uncased",
        # "hf-facebook-bart-base",
        # "hf-facebook-bart-large",
        # "hf-roberta-base",
        # "shufflenet-v2-12-int8",
        # "squeezenet1",
        # "unet",
        # "ssd",
        "yolov3",
        "tiny-yolov3",
        # "shufflenet",
        # "candy",
        # "mosaic",
        # "pointilism",
        # "udnie",
        # "rain-princess",
    ]

    for model_name in need:
        # model_set = entry["testset"]
        # if model_name not in need:
        #     continue
        # if model_set == "full-onnx" or model_set == "release-essential-onnx":
        print("running: {}".format(model_name))
        try:
            run_single_model(model_name, "llvm", model_dir)
            # run_single_model(model_name, "cuda", model_dir)
        except Exception as ex:
            print(ex)


if __name__ == "__main__":
    run_all_models()
