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

function(o2_define_options)

  option(BUILD_SHARED_LIBS "Build shared libs" ON)

  option(BUILD_SIMULATION "Build simulation related parts" ON)

  option(BUILD_EXAMPLES "Build examples" ON)

  option(BUILD_TEST_ROOT_MACROS
         "Build the tests toload and compile the Root macros" ON)

endfunction()
