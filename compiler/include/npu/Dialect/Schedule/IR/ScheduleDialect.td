#ifndef NPU_SCHEDULE_DIALECT
#define NPU_SCHEDULE_DIALECT

include "mlir/IR/DialectBase.td"

def NPUSchedule_Dialect : Dialect {
  let name = "npu_schedule";
  let cppNamespace = "::mlir::npu_schedule";
  let summary = "NPU scheduled pipeline dialect";
  let description = [{
    Public NPU layer that makes scheduling decisions explicit:
    tile boundaries, engine assignment, and async dependencies.
  }];
}

#endif // NPU_SCHEDULE_DIALECT
