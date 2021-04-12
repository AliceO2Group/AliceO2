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

include(O2AddExecutable)

#
# o2_add_test_command(COMMAND) adds a test which uses a command as executable.
#
# Compared to o2_add_test, this function does _not_ add a new executable,
# but uses an already declared executable (or even a script that is 
# not build by cmake at all)
#
# If BUILD_TESTING if not set this function does nothing at all.
#
# o2_add_test_command accept the following parameters : 
#
# required 
#
# * COMMAND : command to be executed as a test
# * NAME : name of the test
#
# recommended/often needed : 
#
# * COMMAND_LINE_ARGS : the arguments to pass to the test, if any
# * COMPONENT_NAME : a short name that will be used to create the 
#   executable name (if not provided with NAME) : o2-test-[COMPONENT_NAME]-...
# * LABELS : labels attached to the test, that can then be used to filter
#   which tests are executed with the `ctest -L` command
#
# optional/less frequently used :
#
# * ENVIRONMENT: extra environment needed by the test to run properly
# * TIMEOUT : the number of seconds allowed for the test to run. Past this time
#   failure is assumed.
# * WORKING_DIRECTORY: the directory in which the test will be ran
#
# rarely needed/for special cases : 
#
# * CONFIGURATIONS : the test will only be ran for those named configurations
#

function(o2_add_test_command)

  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV
    0
    A
    ""
    "COMMAND;COMPONENT_NAME;TIMEOUT;WORKING_DIRECTORY;NAME"
    "COMMAND_LINE_ARGS;LABELS;CONFIGURATIONS;ENVIRONMENT"
    )

  if(A_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  set(noInstall NO_INSTALL)

  if(NOT A_NAME)
    message(FATAL_ERROR "Must give a name to the test")
  endif()

  if(NOT A_COMMAND)
    message(FATAL_ERROR "Must give a command for the test")
  endif()

  add_test(NAME ${A_NAME} 
           COMMAND ${A_COMMAND} ${A_COMMAND_LINE_ARGS}
           WORKING_DIRECTORY ${A_WORKING_DIRECTORY}
           CONFIGURATIONS ${A_CONFIGURATIONS}
          )

  set_property(TEST ${A_NAME} PROPERTY LABELS ${A_LABELS})
  set_property(TEST ${A_NAME} PROPERTY ENVIRONMENT "${A_ENVIRONMENT}")
  set_property(TEST ${A_NAME} PROPERTY TIMEOUT ${A_TIMEOUT})

endfunction()
