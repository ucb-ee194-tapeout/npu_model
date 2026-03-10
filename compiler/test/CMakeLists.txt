configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NPU_TEST_DEPENDS
  npu-opt
  npu-translate
)

add_lit_testsuite(check-npu
  "Running the NPU compiler regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NPU_TEST_DEPENDS}
)

add_lit_testsuites(NPU ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NPU_TEST_DEPENDS})
