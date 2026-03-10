add_mlir_dialect_library(NPUScheduleDialect
  ScheduleDialect.cpp

  DEPENDS
  NPUScheduleOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAsyncDialect
)
