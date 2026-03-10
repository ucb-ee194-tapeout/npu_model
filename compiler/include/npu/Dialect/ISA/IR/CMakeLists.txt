set(LLVM_TARGET_DEFINITIONS ISADialect.td)
mlir_tablegen(ISADialect.h.inc -gen-dialect-decls -dialect=npu_isa)
mlir_tablegen(ISADialect.cpp.inc -gen-dialect-defs -dialect=npu_isa)

set(LLVM_TARGET_DEFINITIONS ISAOps.td)
mlir_tablegen(ISAOps.h.inc -gen-op-decls)
mlir_tablegen(ISAOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(NPUISAOpsIncGen)
