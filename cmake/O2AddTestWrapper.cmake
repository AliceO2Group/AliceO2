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

#
# o2_add_test_wrapper
#
# Same as o2_add_test() but optionally retry up to MAX_ATTEMPTS times upon
# failure. This is achieved by using a shell script wrapper.
#
# * TARGET or COMMAND (required) is either a target name or the full path to the
#   executable to be wrapped
#
# * NAME (optional): the test name.  If not present it is derived from the
#   target name (if TARGET was used) or from the executable name (if COMMAND was
#   given)
#
# * WORKING_DIRECTORY (optional) the wrapper will cd into this directory before
#   running the executable
# * DONT_FAIL_ON_TIMEOUT (optional) indicate the test will not fail on timeouts
# * MAX_ATTEMPTS (optional) the maximum number of attempts
# * TIMEOUT (optional) the test timeout (for each attempt)
# * COMMAND_LINE_ARGS (optional) extra arguments to the test executable, if
#   needed
# * NON_FATAL (optional) mark the test as non criticial for the CI
# * ENVIRONMENT: extra environment needed by the test to run properly
#
function(o2_add_test_wrapper)

  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV
    0
    "A"
    "DONT_FAIL_ON_TIMEOUT;NON_FATAL"
    "TARGET;COMMAND;WORKING_DIRECTORY;MAX_ATTEMPTS;TIMEOUT;NAME"
    "COMMAND_LINE_ARGS;LABELS;CONFIGURATIONS;ENVIRONMENT")

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  if(A_TARGET AND A_COMMAND)
    message(FATAL_ERROR "Should only use one of COMMAND or TARGET")
  endif()

  if(NOT A_TARGET AND NOT A_COMMAND)
    message(FATAL_ERROR "Must give at least one of COMMAND or TARGET")
  endif()

  if(A_TARGET)
    if(NOT TARGET ${A_TARGET})
      message(FATAL_ERROR "${A_TARGET} is not a target")
    endif()
    set(testExe $<TARGET_FILE:${A_TARGET}>)
  endif()

  if(A_COMMAND)
    set(testExe ${A_COMMAND})
  endif()

  if(A_NAME)
    set(testName "${A_NAME}")
  else()
    if(A_COMMAND)
      get_filename_component(testName ${testExe} NAME_WE)
    else()
      set(testName ${A_TARGET})
    endif()
  endif()

  if("${A_MAX_ATTEMPTS}" GREATER 1)
    # Warn only for tests where retry has been requested
    message(
      WARNING "Test ${testName} will be retried max ${A_MAX_ATTEMPTS} times")
  endif()
  if(A_NON_FATAL)
    message(WARNING "Failure of test ${testName} will not be fatal")
  endif()

  if(NOT A_TIMEOUT)
    set(A_TIMEOUT 100) # default timeout (seconds)
  endif()
  if(NOT A_MAX_ATTEMPTS)
    set(A_MAX_ATTEMPTS 1) # default number of attempts
  endif()
  if(A_DONT_FAIL_ON_TIMEOUT)
    set(A_DONT_FAIL_ON_TIMEOUT "--dont-fail-on-timeout")
  else()
    set(A_DONT_FAIL_ON_TIMEOUT "")
  endif()
  if(A_NON_FATAL)
    set(A_NON_FATAL "--non-fatal")
  else()
    set(A_NON_FATAL "")
  endif()

  # For now, we enforce 3 max attempts for all tests.
  # No need to ignore time out, since we have 3 attempts
  set(A_MAX_ATTEMPTS 3)
  set(A_DONT_FAIL_ON_TIMEOUT "")

  math(EXPR ctestTimeout "(20 + ${A_TIMEOUT}) * ${A_MAX_ATTEMPTS}")

  add_test(NAME "${testName}"
           COMMAND "${CMAKE_BINARY_DIR}/tests-wrapper.sh"
                   "--name"
                   "${testName}"
                   "--max-attempts"
                   "${A_MAX_ATTEMPTS}"
                   "--timeout"
                   "${A_TIMEOUT}"
                   ${A_DONT_FAIA_ON_TIMEOUT}
                   ${A_NON_FATAL}
                   "--"
                   ${testExe}
                   ${A_COMMAND_LINE_ARGS}
           WORKING_DIRECTORY "${A_WORKING_DIRECTORY}"
           CONFIGURATIONS "${A_CONFIGURATIONS}")

  set_tests_properties(${testName} PROPERTIES TIMEOUT ${ctestTimeout})
  if(A_LABELS)
    foreach(A IN LISTS A_LABELS)
      set_property(TEST ${testName} APPEND PROPERTY LABELS ${A})
    endforeach()
  endif()
  if(A_ENVIRONMENT)
    set_tests_properties(${testName} PROPERTIES ENVIRONMENT ${A_ENVIRONMENT})
  endif()
endfunction()
