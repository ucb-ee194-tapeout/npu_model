add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Translation)

add_mlir_library(NPUCompilerInit
  InitAllDialects.cpp
  InitAllPasses.cpp

  LINK_LIBS PUBLIC
  NPUKernelDialect
  NPUScheduleDialect
  NPUISADialect
  NPUConversionPasses
)
