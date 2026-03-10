set(LLVM_TARGET_DEFINITIONS KernelDialect.td)
mlir_tablegen(KernelDialect.h.inc -gen-dialect-decls -dialect=npu_kernel)
mlir_tablegen(KernelDialect.cpp.inc -gen-dialect-defs -dialect=npu_kernel)

set(LLVM_TARGET_DEFINITIONS KernelOps.td)
mlir_tablegen(KernelOps.h.inc -gen-op-decls)
mlir_tablegen(KernelOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(NPUKernelOpsIncGen)
