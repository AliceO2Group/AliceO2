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

function(o2_check_cxx_features)
  # FIXME: missing the make_unique here compared to previous version
  foreach(FEAT "cxx_aggregate_default_initializers" "cxx_binary_literals"
          "cxx_generic_lambdas" "cxx_user_literals")
    if(NOT "${FEAT}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
      message(FATAL_ERROR "We miss ${FEAT} feature with this compiler")
    endif()
  endforeach()
endfunction()
