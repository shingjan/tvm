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
from tvm.meta_schedule import schedule_rule, ReplayTraceConfig, postproc, tune_tir
from tvm.meta_schedule.testing import tir_tensor_intrin
from tvm import tir
import tvm
import numpy as np


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

@pytest.mark.skip("Integeration test")
def test_matmul_schedule():
    N = 512
    M = 512
    K = 512
    workload = create_prim_func(
        te_workload.matmul_fp16(
            n=N,
            m=M,
            k=K,
        )
    )

    def schedule(sch: tir.Schedule):
        b0 = sch.get_block(name="C", func_name="main")
        b1 = sch.get_block(name="root", func_name="main")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        l2, l3, l4 = sch.get_loops(block=b0)
        v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 1, 16, 2])
        l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])
        v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[1, 1, 8, 2, 2])
        l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])
        v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64, decision=[2, 2, 8])
        l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])
        sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
        l31 = sch.fuse(l10, l20)
        sch.bind(loop=l31, thread_axis="blockIdx.x")
        l32 = sch.fuse(l11, l21)
        sch.bind(loop=l32, thread_axis="blockIdx.y")
        l33 = sch.fuse(l12, l22)
        sch.bind(loop=l33, thread_axis="threadIdx.y")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
        b34 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
        sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True)
        l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b34)
        l41 = sch.fuse(l39, l40)
        v42 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
        sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v42)
        b43 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")
        sch.compute_at(block=b43, loop=l28, preserve_unit_loops=True)
        l44, l45, l46, l47, l48, l49 = sch.get_loops(block=b43)
        l50 = sch.fuse(l48, l49)
        v51 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
        sch.annotate(block_or_loop=b43, ann_key="meta_schedule.cooperative_fetch", ann_val=v51)
        v52 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v52)

    sch = tir.Schedule(workload)

    schedule(sch)

    from tvm.meta_schedule import (  # pylint: disable=import-outside-toplevel
        postproc as M,
    )

    target = Target("nvidia/geforce-rtx-3070")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_fp16(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rules=multi_level_tiling_tensor_core(target=target),
    )

    postpocs = [
        M.RewriteCooperativeFetch(),
        M.RewriteParallelVectorizeUnroll(),
        M.RewriteReductionBlock(),
        M.RewriteTensorCore(),
        M.VerifyGPUCode(),
    ]
    for postproc in postpocs:
        postproc.initialize_with_tune_context(ctx)
        if postproc.apply(sch) == False:
            print("failed: ")

    # sch: tir.Schedule = tvm.meta_schedule.tune_tir(
    #     mod=workload,
    #     target=Target("llvm --num-cores=16"),
    #     config=ReplayTraceConfig(
    #         num_trials_per_iter=32,
    #         num_trials_total=32,
    #     ),
    #     work_dir=work_dir,
    # )

    if sch is None:
        print("No valid schedule found")
    else:
        print(sch.mod["main"].script())
        print(tvm.lower(sch.mod["main"], None, simple_mode=True))

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(N, K)).astype("float16")
    b_np = np.random.uniform(size=(K, N)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, N), dtype="float32"), dev)
    # sys.exit(0)
    target = Target("nvidia/geforce-rtx-3070")
    f = tvm.build(sch.mod["main"], target=target, name="dense")
    f(a, b, c)
    print(f.imported_modules[0].get_source())
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3, atol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    gflops = (N * M * K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


def test_tune_matmul_cuda_tensor_core():
    print(tir.TensorIntrin.get("wmma_sync"))
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
            #postproc.RewriteCooperativeFetch(),
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
            postproc.RewriteCooperativeFetch(),
            postproc.RewriteParallelVectorizeUnroll(),
            postproc.RewriteReductionBlock(),
            postproc.RewriteTensorCore(),
            postproc.VerifyGPUCode(),
            ]

    n = 32
    mod = create_prim_func(te_workload.matmul_fp16(n, n, n))
    target = Target("nvidia/geforce-rtx-3070")
    config = ReplayTraceConfig(
        num_trials_per_iter=32,
        num_trials_total=320,
    )

    # import tempfile

    # with tempfile.TemporaryDirectory() as work_dir:
    #     sch: tir.Schedule = tune_tir(
    #         mod=mod,
    #         target=target,
    #         config=config,
    #         work_dir=work_dir,
    #         space=PostOrderApply(),
    #         sch_rules=sch_rules,
    #         postprocs=postprocs,
    #         num_threads=None,
    #     )

    # func = tvm.build(sch.mod["main"], [], "cuda")
    # ctx = tvm.device("cuda", 0)
    # print(sch.trace)
    # print(sch.mod.script())
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
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    for schedule in spaces:
        print(schedule.trace)
    
    # run postproc on the trace


if __name__ == "__main__":
    # test_matmul_schedule()
    test_tune_matmul_cuda_tensor_core()
