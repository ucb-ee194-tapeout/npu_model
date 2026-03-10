#include "npu/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::npu_kernel;

#include "npu/Dialect/Kernel/IR/KernelDialect.cpp.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/Kernel/IR/KernelOps.cpp.inc"

void NPUKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/Dialect/Kernel/IR/KernelOps.cpp.inc"
      >();
}
