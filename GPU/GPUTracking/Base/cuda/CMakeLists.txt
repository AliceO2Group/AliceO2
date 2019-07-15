# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE GPUTrackingCUDA)

message(STATUS "Building GPUTracking with CUDA support")

set(SRCS GPUReconstructionCUDA.cu)
set(HDRS GPUReconstructionCUDA.h GPUReconstructionCUDAInternals.h)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PRIVATE_INCLUDE_DIRECTORIES
      ${CMAKE_SOURCE_DIR}/Detectors/Base/src
      ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
      ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include
    PUBLIC_LINK_LIBRARIES O2::GPUTracking O2::ITStrackingCUDA cub::cub
    TARGETVARNAME targetName)

  target_compile_definitions(
    ${targetName} PUBLIC GPUCA_GPULIBRARY=CUDA
    $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

  target_compile_options(${targetName} PUBLIC --expt-relaxed-constexpr)

  set_target_properties(${targetName} PROPERTIES LINKER_LANGUAGE CXX)

  install(FILES ${HDRS} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_GPULIBRARY=CUDA)

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  message(STATUS "TEST ${incdirs}")
  generate_dictionary("Ali${MODULE}" "" "GPUReconstructionCUDA.h"
                      "${incdirs} .")

  # Generate the ROOT map Dependecies
  generate_rootmap("Ali${MODULE}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE} SHARED ${SRCS} G__Ali${MODULE}.cxx)
  # CUDA run-time and driver
  target_link_libraries(Ali${MODULE} AliGPUTracking)

  # Additional compilation flags
  set_target_properties(Ali${MODULE} PROPERTIES COMPILE_FLAGS "")

  # Installation
  install(TARGETS Ali${MODULE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS} DESTINATION include)

  set(targetName Ali${MODULE})
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_definitions(-DGPUCA_GPULIBRARY=CUDA)
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} PUBLIC GPUTracking cub::cub)
  set(targetName ${MODULE})
  install(TARGETS GPUTrackingCUDA)
endif()

# Since
target_link_libraries(${targetName} PRIVATE cuda cudart)
