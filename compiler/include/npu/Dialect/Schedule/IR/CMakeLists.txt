set(LLVM_TARGET_DEFINITIONS ScheduleDialect.td)
mlir_tablegen(ScheduleDialect.h.inc -gen-dialect-decls -dialect=npu_schedule)
mlir_tablegen(ScheduleDialect.cpp.inc -gen-dialect-defs -dialect=npu_schedule)

set(LLVM_TARGET_DEFINITIONS ScheduleOps.td)
mlir_tablegen(ScheduleOps.h.inc -gen-op-decls)
mlir_tablegen(ScheduleOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(NPUScheduleOpsIncGen)
