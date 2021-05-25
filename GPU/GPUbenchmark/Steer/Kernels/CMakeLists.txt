# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(CUDA_ENABLED)
  message("Building CUDA benchmark library")
  add_subdirectory(cuda)
  # target_compile_definitions(${targetName} PRIVATE CUDA_ENABLED)
endif()
if(HIP_ENABLED)
  message("Building HIP benchmark library")
  add_subdirectory(hip)
  # target_compile_definitions(${targetName} PRIVATE HIP_ENABLED)
endif()