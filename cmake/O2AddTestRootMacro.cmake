function(o2_add_test_root_macro)
  if(NOT BUILD_TESTING)
    return()
  endif()

  if(NOT BUILD_TEST_ROOT_MACROS)
    return()
  endif()

  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        "NON_FATAL;LOAD_ONLY"
                        ""
                        "ENVIRONMENT;PUBLIC_LINK_LIBRARIES;LABELS")

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  get_filename_component(macroFileName ${ARGV0} ABSOLUTE)

  if(NOT EXISTS ${macroFileName})
    message(
      WARNING "Requested a test macro for non existing macro ${macroFileName}")
    return()
  endif()

  file(RELATIVE_PATH testName ${CMAKE_SOURCE_DIR} ${macroFileName})

  if(${A_IS_NON_FATAL})
    set(nonFatal "NON_FATAL")
  endif()

  message(
    STATUS
      "TODO: rewrite o2_add_test_root_macro ${testName} as an executable")

  set(tmp test-root-macro-${testName})
  string(REPLACE / - tmp2 ${tmp})
  get_filename_component(exeName ${tmp2} NAME_WLE)

  o2_add_executable(${exeName}
          SOURCES ${CMAKE_SOURCE_DIR}/tests/testRootMacro.cxx
          PUBLIC_LINK_LIBRARIES ROOT::RIO
          TARGETVARNAME exeTargetName)

  message(STATUS "exefile=${exeTargetName}")
  file(GENERATE OUTPUT ${exeTargetName}.dump CONTENT $<TARGET_PROPERTY:${exeTargetName},INCLUDE_DIRECTORIES>)

  # * list(APPEND libdir ${CMAKE_CURRENT_BINARY_DIR}) *
  # * # Get all the include and lib dir dependencies
  # * foreach(t IN LISTS A_PUBLIC_LINK_LIBRARIES)
  # * if(NOT TARGET ${t})
  # * message(
  # * WARNING
  # * "Trying to use non-existing target ${t} for ${testName} test so I won't be
  #   able to generate the required test"
  # * )
  # * return()
  # * endif()
  # * list(APPEND dependencies ${t})
  # * endforeach() *
  # * list(REMOVE_DUPLICATES dependencies) *
  # * foreach(t IN LISTS dependencies)
  # * list(APPEND incdir $<TARGET_PROPERTY:${t},INTERFACE_INCLUDE_DIRECTORIES>)
  # * get_property(targetType TARGET ${t} PROPERTY TYPE)
  # * # we use a custom function here as there is currently (CMake 3.14) nothing
  # * # like $<TARGET_PROPERTY:${t},INTERFACE_LINK_DIRECTORIES> that would give
  # * # us, _transitively_, all the libraries directories to be put on a
  # * # LD_LIBRARY_PATH
  # * o2_get_rpath(${t} RPATH rpath)
  # * list(APPEND libdir ${rpath})
  # * endforeach() *
  # * # FIXME: once CMake 3.15 is out, use $<REMOVE_DUPLICATES:list> to dedupe
  #   the
  # * # includePath list
  # * set(includePath $<JOIN:${incdir},:>) *
  # * list(APPEND libdir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}) *
  # * list(REMOVE_DUPLICATES libdir)
  # * set(libraryPath $<JOIN:${libdir},:>) *
  # * # list(APPEND testEnv "ROOT_HIST=0")
  # * # list(APPEND testEnv "PATH=$ENV{PATH}:${ROOT_BINARY_DIR}")
  # * list(APPEND testEnv "${A_ENVIRONMENT}") *
  # * o2_add_test_wrapper(${CMAKE_BINARY_DIR}/test-root-macro.sh
  # * NAME
  # * ${testName}
  # * WORKING_DIRECTORY
  # * ${CMAKE_BINARY_DIR}
  # * ${nonFatal}
  # * COMMAND_LINE_ARGS
  # * ${macroFileName}
  # * 0
  # * "${includePath}"
  # * "${libraryPath}"
  # * LABELS
  # * "macro;${A_LABELS}") *
  # * set_property(TEST ${testName} PROPERTY ENVIRONMENT "${testEnv}") *
  # * if(NOT A_LOAD_ONLY) *
  # * o2_add_test_wrapper(${CMAKE_BINARY_DIR}/test-root-macro.sh
  # * NAME
  # * ${testName}_compiled
  # * WORKING_DIRECTORY
  # * ${CMAKE_BINARY_DIR}
  # * ${nonFatal}
  # * COMMAND_LINE_ARGS
  # * ${macroFileName}
  # * 1
  # * "${includePath}"
  # * "${libraryPath}"
  # * LABELS
  # * "macro;compiled")
  # * set_property(TEST ${testName}_compiled PROPERTY ENVIRONMENT "${testEnv}")
  #   *
  # * endif()

endfunction()
