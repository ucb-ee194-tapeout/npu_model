add_mlir_dialect_library(NPUISADialect
  ISADialect.cpp

  DEPENDS
  NPUISAOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
