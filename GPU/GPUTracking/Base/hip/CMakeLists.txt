# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE GPUTrackingHIP)

set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})

message(STATUS "Building GPUTracking with HIP support")

set(SRCS GPUReconstructionHIP.hip.cpp)
set(HDRS GPUReconstructionHIP.h GPUReconstructionHIPInternals.h)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PUBLIC_LINK_LIBRARIES O2::GPUTracking hip::host hip::device
    PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
                               ${CMAKE_SOURCE_DIR}/Detectors/Base/src
    TARGETVARNAME targetName)

  target_compile_definitions(
    ${targetName} PUBLIC GPUCA_GPULIBRARY=HIP
    $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

  install(FILES ${HDRS} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_GPULIBRARY=HIP)

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("Ali${MODULE}" "" "GPUReconstructionHIP.h" "${incdirs} .")

  # Generate the ROOT map Dependecies
  generate_rootmap("Ali${MODULE}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE} SHARED ${SRCS} G__Ali${MODULE}.cxx)
  # HIP run-time and driver
  target_link_libraries(Ali${MODULE} AliGPUTracking)

  # Additional compilation flags
  set_target_properties(Ali${MODULE} PROPERTIES COMPILE_FLAGS "")

  # Installation
  install(TARGETS Ali${MODULE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS} DESTINATION include)
  set(targetName "Ali${MODULE}")
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_definitions(-DGPUCA_GPULIBRARY=HIP)
  add_library(${MODULE} SHARED ${SRCS})
  set(targetName "${MODULE}")
  install(TARGETS GPUTrackingHIP)
endif()

if(HIP_AMDGPUTARGET)
  target_link_options(${targetName} PUBLIC --amdgpu-target=${HIP_AMDGPUTARGET})
endif()

target_compile_options(${targetName}
                       PUBLIC -Wno-unused-command-line-argument
                              -Wno-invalid-constexpr
                              -Wno-ignored-optimization-argument
                              -Wno-unused-private-field)
