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

function(o2_build_sanity_checks)
  if(NOT UNIX)
    message(
      FATAL_ERROR
        "You're not on an UNIX system. The project was up to now only tested on UNIX systems, so we break here. IF you want to go on please edit the CMakeLists.txt in the source directory."
      )
  endif()

  if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed.")
  endif()

  if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "CMAKE_BUILD_TYPE not set : will use Debug")
    set(CMAKE_BUILD_TYPE Debug)
  endif()
endfunction()
