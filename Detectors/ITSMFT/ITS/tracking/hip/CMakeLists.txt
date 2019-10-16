# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED HIP_AMDGPUTARGET)
  set(TMP_TARGET "(GPU Target ${HIP_AMDGPUTARGET})")
endif()

message(STATUS "Building ITS HIP tracker")

o2_add_library(ITStrackingHIP
               SOURCES src/UtilsHIP.hip.cxx
                       src/ContextHIP.hip.cxx
               PUBLIC_LINK_LIBRARIES O2::ITStracking hip::host hip::device hip::hipcub
               TARGETVARNAME targetName)

target_compile_definitions(
  ${targetName} PRIVATE $<TARGET_PROPERTY:O2::ITStracking,COMPILE_DEFINITIONS>)

if(HIP_AMDGPUTARGET)
  target_link_options(${targetName} PUBLIC --amdgpu-target=${HIP_AMDGPUTARGET})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --amdgpu-target=${HIP_AMDGPUTARGET}")
endif()

target_compile_options(${targetName}
                       PUBLIC -Wno-invalid-command-line-argument
                              -Wno-unused-command-line-argument
                              -Wno-invalid-constexpr
                              -Wno-ignored-optimization-argument
                              -Wno-unused-private-field)

