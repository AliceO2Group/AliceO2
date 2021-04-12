# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

include(O2AddTestCommand)

#
# o2_add_test_root_macro generate a test for a Root macro.
#
# That test is trying to load the macro within a root.exe session using 
# ".L macro.C"
#
# * arg COMPILE: if present we generate, in addition to the baseline "load
#   the macro test", a test to compile the library, i.e. using ".L macro.C++
# * arg COMPILE_ONLY: if present we discard the "load the macro" test and
#   only attempt the compilation test. It implies COMPILE
# * arg ENVIRONMENT: sets the running environment for the generated test(s).
# * arg PUBLIC_LINK_LIBRARIES: the list of targets this macro is depending on.
#   Required to be able to specify correctly the include and library paths to
#   test the (compiled version of) the macro.
# * arg LABELS: labels to be attached to the test. Note that the label "macro"
#   is automatically applied by this function, and the label "macro_compiled"
#   as well in case COMPILE is defined
#
function(o2_add_test_root_macro macro)

  if(NOT BUILD_TESTING)
    return()
  endif()

  if(NOT BUILD_TEST_ROOT_MACROS)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV
    1
    A
    "COMPILE;COMPILE_ONLY"
    ""
    "ENVIRONMENT;PUBLIC_LINK_LIBRARIES;PUBLIC_INCLUDE_DIRECTORIES;LABELS")

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  get_filename_component(macroFileName ${macro} ABSOLUTE)

  if(NOT EXISTS ${macroFileName})
    message(
      FATAL_ERROR
        "Requested a test macro for non existing macro ${macroFileName}")
    return()
  endif()

  file(RELATIVE_PATH testName ${CMAKE_SOURCE_DIR} ${macroFileName})

  list(APPEND incdir $ENV{ROOT_INCLUDE_PATH})
  list(APPEND incdir ${A_PUBLIC_INCLUDE_DIRECTORIES})

  # Get all the include dir dependencies
  foreach(t IN LISTS A_PUBLIC_LINK_LIBRARIES)
    string(FIND ${t} "::" NS)
    if(${NS} EQUAL -1)
      message(
        WARNING
          "Trying to use a non-namespaced target ${t} for ${testName} test so I won't be able to generate that test."
        )
      return()
    endif()
    list(APPEND dependencies ${t})
  endforeach()

  list(LENGTH dependencies nofDeps)
  if(${nofDeps} GREATER 0)
    list(REMOVE_DUPLICATES dependencies)
    foreach(t IN LISTS dependencies)
      list(APPEND incdir $<TARGET_PROPERTY:${t},INTERFACE_INCLUDE_DIRECTORIES>)
      list(APPEND libdir $<TARGET_PROPERTY:${t},INTERFACE_LINK_DIRECTORIES>)
    endforeach()
    set(includePath $<JOIN:$<REMOVE_DUPLICATES:${incdir}>,:>)
    set(libraryPath $<JOIN:$<REMOVE_DUPLICATES:${libdir}>,:>)
  endif()

  list(APPEND testEnv "ROOT_HIST=0")
  list(APPEND testEnv "${A_ENVIRONMENT}")

  # baseline test is to try and load the macro
  if (NOT A_COMPILE_ONLY)
    o2_add_test_command(COMMAND ${CMAKE_BINARY_DIR}/test-root-macro.sh
                        NAME ${testName}
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR} ${nonFatal}
                        COMMAND_LINE_ARGS ${macroFileName} 0 "${includePath}" "${libraryPath}"
                        LABELS "macro;${A_LABELS}")

    set_property(TEST ${testName} PROPERTY ENVIRONMENT "${testEnv}")

    set(LIST_OF_ROOT_MACRO_TESTS
        "${LIST_OF_ROOT_MACRO_TESTS};${testName}"
        CACHE INTERNAL "List of macros to test for loading")
  endif()

  # if (and only if) requested, try also to compile the macro
  if(A_COMPILE OR A_COMPILE_ONLY)
    o2_add_test_command(COMMAND ${CMAKE_BINARY_DIR}/test-root-macro.sh
                        NAME ${testName}_compiled
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR} ${nonFatal}
                        COMMAND_LINE_ARGS ${macroFileName} 1 "${includePath}" "${libraryPath}"
                        LABELS "macro;macro_compiled;${A_LABELS}")

    set_property(TEST ${testName}_compiled PROPERTY ENVIRONMENT "${testEnv}")
    set(LIST_OF_ROOT_MACRO_TESTS_COMPILED
        "${LIST_OF_ROOT_MACRO_TESTS_COMPILED};${testName}"
        CACHE INTERNAL "List of macros to test for compilation")

  endif()

endfunction()
