#ifndef NPU_KERNEL_DIALECT
#define NPU_KERNEL_DIALECT

include "mlir/IR/DialectBase.td"

def NPUKernel_Dialect : Dialect {
  let name = "npu_kernel";
  let cppNamespace = "::mlir::npu_kernel";
  let summary = "NPU kernel semantic dialect";
  let description = [{
    Public NPU layer for kernel-level semantics that cannot be represented
    precisely in existing standard dialects.
  }];
}

#endif // NPU_KERNEL_DIALECT
