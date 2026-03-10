#include "npu/Dialect/ISA/IR/ISADialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::npu_isa;

#include "npu/Dialect/ISA/IR/ISADialect.cpp.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/ISA/IR/ISAOps.cpp.inc"

void NPUISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/Dialect/ISA/IR/ISAOps.cpp.inc"
      >();
}
