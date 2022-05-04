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
model_dir = "/home/yj/models/"
docs = yaml.safe_load(open(model_dir + "models.yaml", "r"))
onnx_models = {}
datas = {}
dev = None


def download_gcp(gs_url, model_name):
    from google.cloud import storage
    from urllib.parse import urlparse
    from pathlib import Path

    o = urlparse(gs_url)
    path = o.path[1:].split("/", 1)
    destination = model_dir + model_name + ".onnx"
    des_path = Path(destination).resolve()
    if des_path.exists() and des_path.is_file():
        print("file existed. Skipping downloading.")
        return destination
    bucket_name = path[0]
    file_path = path[1]
    print(
        "Downloading object from gcp. Bucket name: {}, file path: {}".format(bucket_name, file_path)
    )
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename(destination)
    return destination


def load_model(model_name):
    # return onnx.load(f"/home/yj/models/{model_name}.onnx")
    for entry in docs:
        if entry["name"] == model_name:
            gs_url = entry["gs_url"]
            print(gs_url)
            onnx_path = download_gcp(gs_url, model_name)
            print(onnx_path)
            onnx_model = onnx.load(onnx_path)
            # onnx_models[entry["name"]] = onnx.load(onnx_path)
            return onnx_model
    raise Exception("Model not found!")


def get_shape_dict(model_name):
    for entry in docs:
        if entry["name"] == model_name:
            doc = entry
            break
    shape_dict = {}
    for input in doc["input_shapes"]:
        shape_dict[input["name"]] = input["shape"]
    return shape_dict


def get_output(lib):
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(**datas)
    module.run()
    return module.get_output(0).numpy()


def get_relay_output(mod, params):
    print("Starting to build with relay.", flush=True)

    # Compile without meta-scheduler for correctness check
    try:

        with tvm.transform.PassContext(
            opt_level=3,
        ):
            rt_mod2 = relay.build(mod, target=target, params=params)
        expected_output = get_output(rt_mod2)
    except:
        with tvm.transform.PassContext(opt_level=3):
            vm_exec = relay.vm.compile(mod, target=target, params=params)

        vm = VirtualMachine(vm_exec, dev)
        vm.set_input("main", **datas)
        expected_output = vm.run().numpy()
    return expected_output


def get_relay_meta_schedule_output(mod, params):
    logger.info("Starting to tune with meta schedule.")
    with tempfile.TemporaryDirectory() as work_dir:
        rt_mod1: tvm.runtime.Module = tune_relay(
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
    return get_output(rt_mod1)


if __name__ == "__main__":
    model_name = str(sys.argv[1])
    target_str = str(sys.argv[2])
    target = (
        Target("llvm --num-cores=8") if target_str == "llvm" else Target("nvidia/geforce-rtx-3070")
    )
    dev = tvm.cpu() if str(target.kind) == "llvm" else tvm.cuda()
    mod, params, (input_name, input_shape, input_dtype) = get_onnx_model(
        model_name,
        model_dir + "models.yaml",
        model_dir,
    )

    if "int" not in input_dtype:
        data = tvm.nd.array(np.random.randn(*input_shape).astype(input_dtype), dev)
    elif model_name == "bert":
        data = tvm.nd.array(np.random.randint(0, 30521, size=input_shape).astype(input_dtype), dev)
    else:
        assert model_name == "gpt2"  # check embedding size 50257 here
        data = tvm.nd.array(np.random.randint(0, 50256, size=input_shape).astype(input_dtype), dev)
    datas[input_name] = data

    # Check correctness
    actual_output = get_relay_meta_schedule_output(mod, params)
    expected_output = get_relay_output(mod, params)
    assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)
    print("The result is correct!")
