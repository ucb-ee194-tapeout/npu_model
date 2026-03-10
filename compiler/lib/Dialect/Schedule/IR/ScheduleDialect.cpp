#include "npu/Dialect/Schedule/IR/ScheduleDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::npu_schedule;

#include "npu/Dialect/Schedule/IR/ScheduleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/Schedule/IR/ScheduleOps.cpp.inc"

void NPUScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/Dialect/Schedule/IR/ScheduleOps.cpp.inc"
      >();
}
