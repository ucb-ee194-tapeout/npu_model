add_llvm_executable(npu-opt
  npu-opt.cpp
  PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(npu-opt)
target_link_libraries(npu-opt PRIVATE
  MLIRIR
  MLIROptLib
  NPUCompilerInit
)
mlir_check_all_link_libraries(npu-opt)

add_llvm_executable(npu-translate
  npu-translate.cpp
  PARTIAL_SOURCES_INTENDED
)
llvm_update_compile_flags(npu-translate)
target_link_libraries(npu-translate PRIVATE
  MLIRIR
  MLIRTranslateLib
  NPUCompilerInit
  NPUTextISATranslation
)
mlir_check_all_link_libraries(npu-translate)
