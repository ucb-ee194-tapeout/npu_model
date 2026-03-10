add_mlir_library(NPUTextISATranslation
  TextISATranslate.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTranslateLib
  NPUCompilerInit
  NPUISADialect
)
