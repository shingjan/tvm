import pytest
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.schedule_rule import (
    multi_level_tiling,
    multi_level_tiling_tensor_core,
)
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.te import create_prim_func
from tvm.meta_schedule.testing import te_workload
from tvm.target import Target
from tvm.meta_schedule import schedule_rule, ReplayTraceConfig, postproc, tune_tir, tune_relay
from tvm.meta_schedule.testing import tir_tensor_intrin, DummyDatabase
from tvm.meta_schedule.tune import extract_task_from_relay
from tvm import tir, te, relay
import tvm.relay.testing
from tvm.relay import transform
import tvm
import numpy as np
import onnx
import tempfile


def _create_context(mod, target, rules) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=rules,
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for sch_rule in ctx.sch_rules:
        sch_rule.initialize_with_tune_context(ctx)
    return ctx

def test_manual_matmul():
    N = 512
    M = 512
    K = 512
    workload = te_workload.matmul_int8(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        # Step 1. Rule-Auto-Tensorize
        # pylint: disable=invalid-name
        i, i_tc = sch.split(i, factors=[None, 16])
        j, j_tc = sch.split(j, factors=[None, 16])
        k, k_tc = sch.split(k, factors=[None, 16])
        sch.reorder(
            # fmt: off
            i, j, k,
            # tensor core
            i_tc, j_tc, k_tc,
            # fmt: on
        )
        block_inner = sch.blockize(i_tc)
        block_outer, block_inner = block_inner, block
        del block
        # Step 2. Rule-Multi-Level-Tiling
        # i_factors = sch.sample_perfect_tile(i, n=5, decision=[8, 1, 2, 1, 4])
        # j_factors = sch.sample_perfect_tile(j, n=5, decision=[1, 8, 4, 1, 2])
        i_factors = [8, 1, 2, 1, 2]
        j_factors = [8, 4, 2, 1, 4]  # swizzle: identity8
        j_factors = [1, 4, 2, 1, 4]
        k_factors = [16, 2, 1]
        i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
        k0, k1, k2 = sch.split(k, k_factors)
        # pylint: enable=invalid-name
        sch.reorder(
            # fmt: off
            i0, j0,   # S => blockIdx.x
            i1, j1,   # S => blockIdx.y
            j2, i2,   # S => threadIdx.y
            # cache_write here
            k0,       # R
            # vectorized cooperative fetching here
            k1,       # R
            i3, j3,   # S
            k2,       # R
            i4, j4,
            # S
            # fmt: on
        )
        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        # num_ty = sch.get(i_factors[2]) * sch.get(j_factors[2])
        num_ty = i_factors[2] * j_factors[2]
        def fetch_to_shared(block, idx, ndim):
            # shared [128 x 32]
            block_read = sch.cache_read(block, idx, "shared")
            # block_read_local = sch.cache_read(block_read, 0, "local")
            sch.compute_at(block_read, k0)
            vector_size = 8
            warp_size = 32
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
            f_0, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
            # sch.reorder(f_1, f_2, f_0, f_3)
            sch.bind(f_2, 'threadIdx.x')
            sch.bind(f_1, 'threadIdx.y')
            # sch.bind(f_0, 'vthread.z')
            # sch.compute_at(block_read_local, f_2)
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=32, offset=16)


        fetch_to_shared(block_outer, 1, 2)

        fetch_to_shared(block_outer, 2, 2)

        # Step 3. Postproc-Rewrite-Tensorize
        # Step 3.1. Cache read
        loop = sch.get_loops(block_outer)[-1]
        block_read_a = sch.cache_read(block_outer, 1, "wmma.matrix_a")
        block_read_b = sch.cache_read(block_outer, 2, "wmma.matrix_b")
        sch.compute_at(block_read_a, k1)
        sch.compute_at(block_read_b, k1)
        # Step 3.2. Cache write
        block_write_c = sch.cache_write(block_outer, 0, "wmma.accumulator")
        # block_outer, block_write_c = block_write_c, block_outer
        sch.reverse_compute_at(block_write_c, thread_idy)
        # Wuwei: we also need spliting the write back stage.
        ii, jj = sch.get_loops(block_write_c)[-2:]
        io, ii = sch.split(ii, factors=[None, 16])
        jo, ji = sch.split(jj, factors=[None, 16])
        sch.reorder(io, jo, ii, ji)
        # Step 3.3. Decompose
        loop = sch.get_loops(block_outer)[3]
        block_init_c = sch.decompose_reduction(block_outer, loop)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]
        # Step 3.4. Tensorize
        loop = sch.get_loops(block_inner)[-3]
        # print(tvm.script.asscript(sch.mod['main']))
        def tile_wmma_fragment(block_read):
            i, j = sch.get_loops(block_read)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            return i1

        print(sch.mod["main"].script())
        sch.tensorize(loop, "wmma_sync_int8")
        loop = tile_wmma_fragment(block_read_a)
        sch.tensorize(loop, "wmma_load_a_int8")
        loop = tile_wmma_fragment(block_read_b)
        sch.tensorize(loop, "wmma_load_b_int8")
        loop = sch.get_loops(block_init_c_inner)[-2]
        sch.tensorize(loop, "wmma_fill_int8")
        loop = sch.get_loops(block_write_c)[-2]
        sch.tensorize(loop, "wmma_store_int8")

    # task = ms.SearchTask(
    #         workload=workload,
    #         target=TARGET,
    #         target_host='llvm',
    #         task_name="cuda_matmul",
    #         log_file="./cuda_matmul.json",
    #     )
    # space = ms.space.ScheduleFn(
    #     schedule,
    #     postprocs=[
    #         ms.postproc.verify_gpu_code(),
    #     ],
    # )
    # Evolutionary search doesn't support using result of sch.get() as the split factor.
    # Enable this when we have postprocessors for auto tensorization.
    # evolutionary = ms.strategy.Evolutionary(
    #         total_measures=256,
    #         num_measures_per_iter=16,
    #         population=128,
    #         init_measured_ratio=0.2,
    #         genetic_algo_iters=10,
    #         p_mutate=0.85,
    #         mutator_probs={
    #             ms.mutator.mutate_tile_size(): 1.0,
    #         },
    #         cost_model=ms.XGBModel(
    #             num_warmup_samples=0,
    #         ),
    #         eps_greedy=0.05,
    #     )
    sch = tir.Schedule(workload)
    schedule(sch)

    # replay = ms.strategy.Replay(256)
    # sch = ms.autotune(
    #     task=task,
    #     space=space,
    #     strategy=replay,
    #     measurer=ms.ProgramMeasurer(
    #         measure_callbacks=[
    #             ms.RecordToFile(),
    #         ]
    #     ),
    # )
    # assert space.postprocess(task, sch)
    if sch is None:
        print("No valid schedule found")
    else:
        print(sch.mod["main"].script())
        print(tvm.lower(sch.mod["main"], None, simple_mode=True))

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(N, K)).astype(np.int8)
    b_np = np.random.uniform(size=(K, M)).astype(np.int8)
    c_np = np.dot(a_np.astype(np.int32), b_np.astype(np.int32))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype=np.int32), dev)
    # sys.exit(0)
    f = tvm.build(sch.mod["main"], target="cuda", name="dense")
    print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    gflops = (N * M * K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


def test_tune_matmul():
    def sch_rules():
        return [
            schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            schedule_rule.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
                use_tensor_core=True,
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 3, 4],
                reuse_read=schedule_rule.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=schedule_rule.ReuseType(
                    req="no",
                    levels=[],
                    scope="",
                ),
            ),
            schedule_rule.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]

    def postprocs():
        return [
            postproc.RewriteCooperativeFetch(),
            postproc.RewriteParallelVectorizeUnroll(),
            postproc.RewriteReductionBlock(),
            postproc.RewriteTensorCore(),
            postproc.VerifyGPUCode(),
        ]

    rule_list = [
        schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        schedule_rule.MultiLevelTiling(
            structure="SSSRRSRS",
            tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
            use_tensor_core=True,
            max_innermost_factor=64,
            vector_load_lens=[1, 2, 3, 4],
            reuse_read=schedule_rule.ReuseType(
                req="must",
                levels=[4],
                scope="shared",
            ),
            reuse_write=schedule_rule.ReuseType(
                req="no",
                levels=[],
                scope="",
            ),
        ),
        schedule_rule.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        ),
    ]
    postproc_list = [
        # postproc.RewriteCooperativeFetch(),
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorCore(),
        postproc.VerifyGPUCode(),
    ]

    n = 16
    mod = create_prim_func(te_workload.matmul_fp16(n, n, n))
    target = Target("nvidia/geforce-rtx-3070")
    config = ReplayTraceConfig(
        num_trials_per_iter=32,
        num_trials_total=320,
    )

    with tempfile.TemporaryDirectory() as work_dir:
        sch: tir.Schedule = tune_tir(
            mod=mod,
            target=target,
            config=config,
            work_dir=work_dir,
            space=PostOrderApply(),
            sch_rules=sch_rules,
            postprocs=postprocs,
            num_threads=None,
        )

    # func = tvm.build(sch.mod["main"], [], "cuda")
    # ctx = tvm.device("cuda", 0)
    # print(sch.trace)
    print(sch.mod.script())
    # print(func.imported_modules[0].get_source())
    # a_np = np.random.uniform(size=(n, n)).astype("float16")
    # b_np = np.random.uniform(size=(n, n)).astype("float16")
    # a = tvm.nd.array(a_np, ctx)
    # b = tvm.nd.array(b_np, ctx)
    # c = tvm.nd.array(np.zeros((n, n), dtype="float32"), ctx)
    # evaluator = func.time_evaluator(func.entry_name, ctx, number=3, repeat=1, min_repeat_ms=40)
    # print("matmul with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))

    # np.testing.assert_allclose(
    #     c.asnumpy(),
    #     np.matmul(a_np.astype("float32"), b_np.astype("float32")),
    #     rtol=1e-4,
    #     atol=1e-4,
    # )

    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_fp16(
                n=n,
                m=n,
                k=n,
            )
        ),
        target=target,
        rules=rule_list,
    )
    # spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    # for schedule in spaces:
    #     print(schedule.trace)

    # run postproc on the trace

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

    with open(json_path, "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    with open(params_path, "rb") as fi:
        params = relay.load_param_dict(fi.read())

    target = Target("nvidia/geforce-rtx-3070")

    extracted_tasks = extract_task_from_relay(mod, target, params)
    tune_tasks = list(
        filter(
            lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
            extracted_tasks,
        )
    )
    print(tune_tasks)


    print(mod)

def test_load_bert_int8():
    json_path = "models/bert_base_int8.json"
    params_path = "models/bert_base_int8.params"
    target = Target("nvidia/geforce-rtx-3070")
    
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
    #extracted_tasks = extract_task_from_relay(mod, target, params)
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target(target)
        database = DummyDatabase()
        rt_mod: tvm.module = tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                num_trials_total=32,
            ),
            work_dir=work_dir,
            database=database,
        )
    # print(mod)


if __name__ == "__main__":
    # test_manual_matmul()
    # test_tune_matmul()
    # test_bert_int8()
    test_load_bert_int8()