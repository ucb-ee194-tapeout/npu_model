#include "npu/InitAllDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "npu/Dialect/ISA/IR/ISADialect.h"
#include "npu/Dialect/Kernel/IR/KernelDialect.h"
#include "npu/Dialect/Schedule/IR/ScheduleDialect.h"

namespace npu {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::async::AsyncDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();

  registry.insert<mlir::npu_kernel::NPUKernelDialect>();
  registry.insert<mlir::npu_schedule::NPUScheduleDialect>();
  registry.insert<mlir::npu_isa::NPUISADialect>();
}

} // namespace npu
