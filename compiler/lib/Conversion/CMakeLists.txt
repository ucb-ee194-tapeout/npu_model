add_mlir_library(NPUConversionPasses
  LinalgToKernel/LinalgToKernel.cpp
  KernelToSchedule/KernelToSchedule.cpp
  ScheduleToISA/ScheduleToISA.cpp
  Passes.cpp

  DEPENDS
  NPUDialectsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  NPUKernelDialect
  NPUScheduleDialect
  NPUISADialect
)
