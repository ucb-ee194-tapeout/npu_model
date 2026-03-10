add_mlir_dialect_library(NPUKernelDialect
  KernelDialect.cpp

  DEPENDS
  NPUKernelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAsyncDialect
)
