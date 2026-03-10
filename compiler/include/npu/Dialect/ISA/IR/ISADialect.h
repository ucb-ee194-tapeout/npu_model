#ifndef NPU_DIALECT_ISA_IR_ISADIALECT_H_
#define NPU_DIALECT_ISA_IR_ISADIALECT_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "npu/Dialect/ISA/IR/ISADialect.h.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/ISA/IR/ISAOps.h.inc"
#undef GET_OP_CLASSES

#endif // NPU_DIALECT_ISA_IR_ISADIALECT_H_
