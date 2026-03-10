#ifndef NPU_DIALECT_SCHEDULE_IR_SCHEDULEDIALECT_H_
#define NPU_DIALECT_SCHEDULE_IR_SCHEDULEDIALECT_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "npu/Dialect/Schedule/IR/ScheduleDialect.h.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/Schedule/IR/ScheduleOps.h.inc"
#undef GET_OP_CLASSES

#endif // NPU_DIALECT_SCHEDULE_IR_SCHEDULEDIALECT_H_
