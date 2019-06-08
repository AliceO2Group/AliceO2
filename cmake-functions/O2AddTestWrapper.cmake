include_guard()

# ------------------------------------------------------------------------------
# o2_add_test_wrapper(exe)
#
# Same as add_test() but optionally retry up to MAX_ATTEMPTS times upon failure.
# This is achieved by using a shell script wrapper.
#
# * COMMAND (required) is the full path to the executable to be wrapped
#
# * NAME (optional) if not present the name of the test is the command name
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
  cmake_parse_arguments(PARSE_ARGV
                        0
                        "A"
                        "DONT_FAIL_ON_TIMEOUT;NON_FATAL"
                        "COMMAND;WORKING_DIRECTORY;MAX_ATTEMPTS;TIMEOUT;NAME"
                        "COMMAND_LINE_ARGS;LABELS;CONFIGURATIONS;ENVIRONMENT")

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  set(testExe ${A_COMMAND})

  if(A_NAME)
    set(testName ${A_NAME})
  else()
    get_filename_component(testName ${testExe} NAME_WLE)
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
    set_tests_properties(${testName} PROPERTIES LABELS ${A_LABELS})
  endif()
  if(A_ENVIRONMENT)
    set_tests_properties(${testName} PROPERTIES ENVIRONMENT "${A_ENVIRONMENT}")
  endif()
endfunction()
