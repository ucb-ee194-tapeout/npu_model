#ifndef NPU_ISA_DIALECT
#define NPU_ISA_DIALECT

include "mlir/IR/DialectBase.td"

def NPUISA_Dialect : Dialect {
  let name = "npu_isa";
  let cppNamespace = "::mlir::npu_isa";
  let summary = "NPU internal near-ISA dialect";
  let description = [{
    Internal layer that is close to hardware instruction forms and is directly
    translated to simulator-compatible textual ISA.
  }];
}

#endif // NPU_ISA_DIALECT
