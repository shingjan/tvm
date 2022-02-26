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
from tvm.meta_schedule import schedule_rule, ReplayTraceConfig
from tvm.meta_schedule.testing import tir_tensor_intrin
from tvm import tir
import tvm
import numpy as np


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for sch_rule in ctx.sch_rules:
        sch_rule.initialize_with_tune_context(ctx)
    return ctx


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
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        l1, l2, l3 = sch.get_loops(block=b0)
        l4, l5 = sch.split(loop=l1, factors=[32, 16])
        l6, l7 = sch.split(loop=l2, factors=[32, 16])
        l8, l9 = sch.split(loop=l3, factors=[32, 16])
        l10, l11, l12, l13, l14, l15 = sch.get_loops(block=b0)
        sch.reorder(l12, l14, l5, l7, l9)
        b16 = sch.blockize(loop=l5)
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync")
        sch.annotate(block_or_loop=b16, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill")
        b17 = sch.get_block(name="root", func_name="main")
        sch.annotate(block_or_loop=b17, ann_key="meta_schedule.tensor_core_enabled", ann_val="1")
        b18 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="local")
        b19 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="wmma.accumulator")
        sch.annotate(
            block_or_loop=b19, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store"
        )
        l20, l21, l22 = sch.get_loops(block=b16)
        v23, v24, v25, v26, v27 = sch.sample_perfect_tile(loop=l20, n=5, max_innermost_factor=64)
        l28, l29, l30, l31, l32 = sch.split(loop=l20, factors=[v23, v24, v25, v26, v27])
        v33, v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l21, n=5, max_innermost_factor=64)
        l38, l39, l40, l41, l42 = sch.split(loop=l21, factors=[v33, v34, v35, v36, v37])
        v43, v44, v45 = sch.sample_perfect_tile(loop=l22, n=3, max_innermost_factor=64)
        l46, l47, l48 = sch.split(loop=l22, factors=[v43, v44, v45])
        sch.reorder(l28, l38, l29, l39, l30, l40, l46, l47, l31, l41, l48, l32, l42)
        l49 = sch.fuse(l28, l38)
        sch.bind(loop=l49, thread_axis="blockIdx.x")
        l50 = sch.fuse(l29, l39)
        sch.bind(loop=l50, thread_axis="blockIdx.y")
        l51 = sch.fuse(l30, l40)
        sch.bind(loop=l51, thread_axis="threadIdx.y")
        b52 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="shared")
        sch.compute_at(block=b52, loop=l46, preserve_unit_loops=True)
        l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b52)
        l59 = sch.fuse(l57, l58)
        v60 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
        sch.annotate(block_or_loop=b52, ann_key="meta_schedule.cooperative_fetch", ann_val=v60)
        b61 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="shared")
        sch.compute_at(block=b61, loop=l46, preserve_unit_loops=True)
        l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b61)
        l68 = sch.fuse(l66, l67)
        v69 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
        sch.annotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch", ann_val=v69)
        b70 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="wmma.matrix_a")
        b71 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b70, loop=l48, preserve_unit_loops=True)
        sch.compute_at(block=b71, loop=l48, preserve_unit_loops=True)
        sch.annotate(
            block_or_loop=b70, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_a"
        )
        sch.annotate(
            block_or_loop=b71, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_b"
        )
        sch.reverse_compute_at(block=b19, loop=l51, preserve_unit_loops=True)
        sch.reverse_compute_at(block=b18, loop=l51, preserve_unit_loops=True)

    sch = tir.Schedule(workload)

    schedule(sch)
    from tvm.meta_schedule import (  # pylint: disable=import-outside-toplevel
        postproc as M,
    )

    target = Target("nvidia/geforce-rtx-3070")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_fp16(
                n=16,
                m=16,
                k=32,
            )
        ),
        target=target,
        rule=multi_level_tiling_tensor_core(target=target),
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
            print("failed")

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


@pytest.mark.skip("Integration test")
def test_tune_matmul_cuda_tensor_core():
    n = 512
    mod = create_prim_func(te_workload.matmul_fp16(n, n, n))
    target = Target("nvidia/geforce-rtx-3070")
    config = ReplayTraceConfig(
        num_trials_per_iter=32,
        num_trials_total=320,
    )

    class DefaultTensorCore:
        @staticmethod
        def _sch_rules():
            from tvm.meta_schedule import (  # pylint: disable=import-outside-toplevel
                schedule_rule as M,
            )

            return [
                M.AutoInline(
                    into_producer=False,
                    into_consumer=True,
                    inline_const_tensor=True,
                    disallow_if_then_else=False,
                    require_injective=False,
                    require_ordered=False,
                    disallow_op=None,
                ),
                M.MultiLevelTiling(
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
                M.AutoInline(
                    into_producer=True,
                    into_consumer=True,
                    inline_const_tensor=True,
                    disallow_if_then_else=False,
                    require_injective=False,
                    require_ordered=False,
                    disallow_op=None,
                ),
                M.ParallelizeVectorizeUnroll(
                    max_jobs_per_core=-1,  # disable parallelize
                    max_vectorize_extent=-1,  # disable vectorize
                    unroll_max_steps=[0, 16, 64, 512, 1024],
                    unroll_explicit=True,
                ),
            ]

        @staticmethod
        def _postproc():
            from tvm.meta_schedule import (  # pylint: disable=import-outside-toplevel
                postproc as M,
            )

            return [
                M.RewriteCooperativeFetch(),
                M.RewriteParallelVectorizeUnroll(),
                M.RewriteReductionBlock(),
                M.RewriteTensorCore(),
                M.VerifyGPUCode(),
            ]

    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_fp16(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling_tensor_core(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    trace = ...
    # run postproc on the trace


if __name__ == "__main__":
    test_matmul_schedule()
