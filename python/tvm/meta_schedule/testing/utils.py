from typing import Tuple, List

import tvm
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord
from tvm.ir import IRModule

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