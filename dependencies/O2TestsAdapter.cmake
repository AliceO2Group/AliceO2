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

# We patch those tests that require some environment (most notably the O2_ROOT
# variable) to convert from O2_ROOT pointing to build tree to O2_ROOT pointing
# to install tree. Should not be needed in the long run if we consider (as we
# should, I would argue) that tests are running off the build tree, before
# installation (and are not installed, as there's probably no point in doing so)
# Should not be needed in the long run if we consider (as we should, I would
# argue) that tests are running off the build tree, before installation (and are
# not installed, as there's probably no point in doing so)
#

include_guard()

if(DEFINED ENV{ALIBUILD_O2_TESTS} AND PROJECT_NAME STREQUAL "O2")
  message(STATUS "!!!")
  message(
    STATUS
      "!!! ALIBUILD_O2_TESTS detected. Will patch my tests so they work off the install tree"
    )
  configure_file(${CMAKE_SOURCE_DIR}/tests/tmp-patch-tests-environment.sh.in
                 tmp-patch-tests-environment.sh)
  install(
    CODE [[ execute_process(COMMAND bash tmp-patch-tests-environment.sh) ]])
endif()

