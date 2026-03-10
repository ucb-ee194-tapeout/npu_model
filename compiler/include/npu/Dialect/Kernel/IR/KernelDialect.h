#ifndef NPU_DIALECT_KERNEL_IR_KERNELDIALECT_H_
#define NPU_DIALECT_KERNEL_IR_KERNELDIALECT_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "npu/Dialect/Kernel/IR/KernelDialect.h.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/Kernel/IR/KernelOps.h.inc"
#undef GET_OP_CLASSES

#endif // NPU_DIALECT_KERNEL_IR_KERNELDIALECT_H_
