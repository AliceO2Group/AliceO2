include_guard()

include(O2AddExecutable)
include(O2AddTestWrapper)

#
# o2_add_test(testName SOURCES ...) adds a test. The test itself (in the CTest
# sense) is a wrapper around an executable. Both the test wrapper and the
# executable are setup by this function.
#
# If BUILD_TESTING if not set this function does nothing at all.
#
# This function accepts two sets of parameters : one for the executable and one
# for the test wrapper.
#
# Parameters of the test executable :
#
# * NO_BOOST_TEST :  we assume most of the tests are using the Boost::Test
#   framework and thus link the test with the Boost::unit_test_framework target.
#   If the test is known to not depend on Boost::Test, then the NO_BOOST_TEST
#   option can be given to forego this dependency
# * INSTALL : by default tests are _not_ installed. If that option is present
#   then the test is installed (under ${CMAKE_INSTALL_PREFIX}/tests, not in
#   ${CMAKE_INSTALL_PREFIX}/bin like other binaries)
#
# Parameters of the test wrapper :
#
# * MAX_ATTEMPTS : the number of time the test will be tried (upon failures)
#   before it is actually considered as failed
# * TIMEOUT : the number of seconds allowed for the test to run. Past this time
#   failure is assumed.
# * NON_FATAL : true if the failing of this test is not causing the CI to fail

function(o2_add_test)

  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV
    1
    A
    "INSTALL;NO_BOOST_TEST;NON_FATAL"
    "COMPONENT_NAME;MAX_ATTEMPTS;TIMEOUT;WORKING_DIRECTORY"
    "SOURCES;PUBLIC_LINK_LIBRARIES;COMMAND_LINE_ARGS;LABELS;CONFIGURATIONS")

  if(A_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  set(testName ${ARGV0})

  set(linkLibraries ${A_PUBLIC_LINK_LIBRARIES})

  if(NOT A_NO_BOOST_TEST)
    set(linkLibraries ${linkLibraries} Boost::unit_test_framework)
    if(A_COMMAND_LINE_ARGS)
      # Boost test programs are to be called like this : testProgram -- arg1
      # arg2 ... if they have arguments.
      set(A_COMMAND_LINE_ARGS "--" ${A_COMMAND_LINE_ARGS})
    endif()
  endif()

  set(noInstall NO_INSTALL)

  if(A_INSTALL)
    set(noInstall "")
  endif()

  # create the executable
  o2_add_executable(${testName}
                    SOURCES ${A_SOURCES}
                    PUBLIC_LINK_LIBRARIES ${linkLibraries}
                    COMPONENT_NAME ${A_COMPONENT_NAME}
                    IS_TEST ${noInstall}
                    EXEVARNAME exe)

  set(nonFatal "")
  if(NON_FATAL)
    set(nonFatal NON_FATAL)
  endif()

  # create a test with a script wrapping the executable above
  o2_add_test_wrapper(COMMAND ${exe}
                      DONT_FAIL_ON_TIMEOUT
                      MAX_ATTEMPTS ${A_MAX_ATTEMPTS}
                      TIMEOUT ${A_TIMEOUT} ${nonFatal}
                      WORKING_DIRECTORY ${A_WORKING_DIRECTORY}
                      COMMAND_LINE_ARGS ${A_COMMAND_LINE_ARGS}
                      LABELS ${A_LABELS}
                      CONFIGURATIONS ${A_CONFIGURATIONS})
endfunction()
