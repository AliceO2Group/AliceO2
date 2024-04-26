# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

include_guard()

include(O2AddExecutable)

#
# o2_add_test(testName SOURCES) adds an executable that is also a test.
#
# It is a convenience function that groups the actions of cmake intrinsics
# add_executable and add_test.
#
# If BUILD_TESTING if not set this function does nothing at all.
#
# The test name is the name of the first source file in SOURCES, unless the NAME
# parameter is given (see below).
#
# o2_add_test accept the following parameters :
#
# required
#
# * SOURCES : same meaning as for o2_add_executable
#
# recommended/often needed :
#
# * PUBLIC_LINK_LIBRARIES : same meaning as for
#   o2_add_executable
# * COMMAND_LINE_ARGS : the arguments to pass to the executable, if any
# * COMPONENT_NAME : a short name that will be used to create the
#   executable name (if not provided with NAME) : o2-test-[COMPONENT_NAME]-...
# * LABELS : labels attached to the test, that can then be used to filter
#   which tests are executed with the `ctest -L` command
#
# optional/less frequently used :
#
# * NAME: if given, will be used verbatim as the test name
# * ENVIRONMENT: extra environment needed by the test to run properly
# * TIMEOUT : the number of seconds allowed for the test to run. Past this time
#   failure is assumed.
# * WORKING_DIRECTORY: the directory in which the test will be ran
#
# rarely needed/for special cases :
#
# * NO_BOOST_TEST :  we assume most of the tests are using the Boost::Test
#   framework and thus link the test with the Boost::unit_test_framework target.
#   If the test is known to not depend on Boost::Test, then the NO_BOOST_TEST
#   option can be given to forego this dependency
# * INSTALL : by default tests are _not_ installed. If that option is present
#   then the test is installed (under ${CMAKE_INSTALL_PREFIX}/tests, not in
#   ${CMAKE_INSTALL_PREFIX}/bin like other binaries)
# * CONFIGURATIONS : the test will only be ran for those named configurations
# * TARGETVARNAME : same meaning as for o2_add_executable
#

function(o2_add_test)

  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV
    1
    A
    "INSTALL;NO_BOOST_TEST"
    "COMPONENT_NAME;TIMEOUT;WORKING_DIRECTORY;NAME;TARGETVARNAME;HIPIFIED"
    "SOURCES;PUBLIC_LINK_LIBRARIES;COMMAND_LINE_ARGS;LABELS;CONFIGURATIONS;ENVIRONMENT"
    )

  if(A_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unexpected unparsed arguments: ${A_UNPARSED_ARGUMENTS}")
  endif()

  set(testName ${ARGV0})

  set(linkLibraries ${A_PUBLIC_LINK_LIBRARIES})

  if(NOT A_NO_BOOST_TEST)
    set(linkLibraries ${linkLibraries} Boost::unit_test_framework)
    if(A_COMMAND_LINE_ARGS)
      # Boost test programs are to be called like this :
      #
      # testProgram -- arg1 arg2 ...
      #
      # if they have arguments.
      set(A_COMMAND_LINE_ARGS "--" ${A_COMMAND_LINE_ARGS})
    endif()
  endif()

  set(noInstall NO_INSTALL)

  if(A_INSTALL)
    set(noInstall "")
  endif()

  # create the executable
  if (NOT A_HIPIFIED)
    o2_add_executable(${testName}
                      SOURCES ${A_SOURCES}
                      PUBLIC_LINK_LIBRARIES ${linkLibraries}
                      COMPONENT_NAME ${A_COMPONENT_NAME}
                      IS_TEST ${noInstall} TARGETVARNAME targetName)
  else()
    o2_add_hipified_executable(${testName}
                        SOURCES ${A_SOURCES}
                        DEST_SRC_REL_PATH ${A_HIPIFIED}
                        PUBLIC_LINK_LIBRARIES ${linkLibraries}
                        COMPONENT_NAME ${A_COMPONENT_NAME}
                        IS_TEST ${noInstall} TARGETVARNAME targetName)
  endif()

  if(A_TARGETVARNAME)
    set(${A_TARGETVARNAME} ${targetName} PARENT_SCOPE)
  endif()

  # create a test for the executable above
  set(name "")
  if(A_NAME)
    set(name ${A_NAME})
  else()
    list(GET A_SOURCES 0 firstSource)
    get_filename_component(src ${firstSource} ABSOLUTE)
    file(RELATIVE_PATH name ${CMAKE_SOURCE_DIR} ${src})
  endif()

  add_test(NAME ${name}
           COMMAND ${targetName} ${A_COMMAND_LINE_ARGS}
           WORKING_DIRECTORY ${A_WORKING_DIRECTORY}
           CONFIGURATIONS ${A_CONFIGURATIONS}
          )

  set_property(TEST ${name} PROPERTY LABELS ${A_LABELS})
  set_property(TEST ${name} PROPERTY ENVIRONMENT "${A_ENVIRONMENT}")
  set_property(TEST ${name} PROPERTY TIMEOUT ${A_TIMEOUT})

endfunction()
