include_guard()

function(o2_setup_testing)

  # Create special .rootrc for testing compiled macros
  configure_file(tests.rootrc.in ${CMAKE_BINARY_DIR}/.rootrc @ONLY)

  # Create special script for testing root macros
  configure_file(test-root-macro.sh.in ${CMAKE_BINARY_DIR}/test-root-macro.sh @ONLY)

  # Create tests wrapper (and make it executable)
  configure_file(tests-wrapper.sh.in ${CMAKE_BINARY_DIR}/tests-wrapper.sh @ONLY)

  # Create test for executable naming convention
  configure_file(ensure-executable-naming-convention.sh.in
                 ${CMAKE_BINARY_DIR}/ensure-executable-naming-convention.sh @ONLY)

  add_test(NAME ensure-executable-naming-convention
           COMMAND ${CMAKE_BINARY_DIR}/ensure-executable-naming-convention.sh @ONLY)

  # Place the CTestCustom.cmake in the build dir
  configure_file(CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake)

endfunction()
