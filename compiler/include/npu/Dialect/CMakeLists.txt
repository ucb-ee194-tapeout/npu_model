add_subdirectory(Kernel/IR)
add_subdirectory(Schedule/IR)
add_subdirectory(ISA/IR)

add_custom_target(NPUDialectsIncGen)
add_dependencies(NPUDialectsIncGen
  NPUKernelOpsIncGen
  NPUScheduleOpsIncGen
  NPUISAOpsIncGen)
