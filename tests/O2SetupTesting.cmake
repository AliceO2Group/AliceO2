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

function(o2_setup_testing)

  # Create special .rootrc for testing compiled macros
  configure_file(tests.rootrc.in ${CMAKE_BINARY_DIR}/.rootrc @ONLY)

  # Create special script for testing root macros. This script needs
  # LD_LIBRARY_PATH
  if(NOT LD_LIBRARY_PATH)
    set(LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})
  endif()
  configure_file(test-root-macro.sh.in ${CMAKE_BINARY_DIR}/test-root-macro.sh
                 @ONLY)

  # Create tests wrapper (and make it executable)
  configure_file(tests-wrapper.sh.in ${CMAKE_BINARY_DIR}/tests-wrapper.sh @ONLY)

  # Create test for executable naming convention
  configure_file(ensure-executable-naming-convention.sh.in
                 ${CMAKE_BINARY_DIR}/ensure-executable-naming-convention.sh
                 @ONLY)

  add_test(NAME ensure-executable-naming-convention
           COMMAND ${CMAKE_BINARY_DIR}/ensure-executable-naming-convention.sh
                   @ONLY)

endfunction()
