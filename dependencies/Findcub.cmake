# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

find_path(CUB_INCLUDE_DIR cub/cub.cuh PATHS ${cub_ROOT} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(NOT CUB_INCLUDE_DIR)
  set(CUB_FOUND FALSE)
  return()
endif()

set(CUB_FOUND TRUE)

if(NOT TARGET cub::cub)
  add_library(cub::cub INTERFACE IMPORTED)
  target_include_directories(cub::cub SYSTEM INTERFACE ${CUB_INCLUDE_DIR})
endif()

mark_as_advanced(CUB_INCLUDE_DIR)
