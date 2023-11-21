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

set(MODULE GPUTrackingCUDA)

if(DEFINED CUDA_COMPUTETARGET)
  set(TMP_TARGET "(Compute Target ${CUDA_COMPUTETARGET})")
endif()
message(STATUS "Building GPUTracking with CUDA support ${TMP_TARGET}")

set(SRCS GPUReconstructionCUDA.cu GPUReconstructionCUDAKernels.cu GPUReconstructionCUDAGenRTC.cu)
set(HDRS GPUReconstructionCUDA.h GPUReconstructionCUDAInternals.h GPUReconstructionCUDADef.h GPUReconstructionCUDAIncludes.h CUDAThrustHelpers.h)

# -------------------------------- Prepare RTC -------------------------------------------------------
if(NOT ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  enable_language(ASM)
  if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(defineIncludeSrc "O2::${MODULE}")
  else()
    set(defineIncludeSrc "${MODULE}")
  endif()
  set(CURTC_DEFINES "-D$<JOIN:$<TARGET_PROPERTY:${defineIncludeSrc},COMPILE_DEFINITIONS>,$<SEMICOLON>-D>"
                    -DGPUCA_GPUCODE_GENRTC
  )
  set(CURTC_INCLUDES "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:${defineIncludeSrc},INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>"
                    -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
                    -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
  )
  if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(CURTC_INCLUDES ${CURTC_INCLUDES} "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:O2::ITStrackingCUDA,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>")
  endif()
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep")

  set(CUDARTC_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}} -std=c++${CMAKE_CUDA_STANDARD}")
  if(CUDA_COMPUTETARGET)
      foreach(CUDA_ARCH ${CUDA_COMPUTETARGET})
          set(CUDARTC_FLAGS "${CUDARTC_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
      endforeach()
      list (GET CUDA_COMPUTETARGET 0 RTC_CUDA_ARCH)
      set(RTC_CUDA_ARCH "${RTC_CUDA_ARCH}0")
  else()
      set(RTC_CUDA_ARCH "750")
  endif()
  separate_arguments(CUDARTC_FLAGS)

  # convenience variables
  if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
    get_filename_component(GPUDIR ${CMAKE_SOURCE_DIR}/../ ABSOLUTE)
  else()
    set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
  endif()
  set(CURTC_SRC ${GPUDIR}/Base/cuda/GPUReconstructionCUDArtc.cu)
  set(CURTC_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionCUDArtc)

  # cmake-format: off
  add_custom_command(
      OUTPUT ${CURTC_BIN}.src
      COMMAND cat ${GPUDIR}/Base/cuda/GPUReconstructionCUDAIncludes.h > ${CURTC_BIN}.src
      COMMAND ${CMAKE_CXX_COMPILER} ${CURTC_DEFINES} ${CURTC_INCLUDES} -std=c++${CMAKE_CUDA_STANDARD} -D__CUDA_ARCH__=${RTC_CUDA_ARCH} -D__CUDACC__ -x c++ -E ${CURTC_SRC} >> ${CURTC_BIN}.src
      MAIN_DEPENDENCY ${CURTC_SRC}
      IMPLICIT_DEPENDS CXX ${CURTC_SRC}
      COMMAND_EXPAND_LISTS
  )
  create_binary_resource(${CURTC_BIN}.src ${CURTC_BIN}.src.o)

  add_custom_command(
      OUTPUT ${CURTC_BIN}.command
      COMMAND echo -n "${CMAKE_CUDA_COMPILER} ${CUDARTC_FLAGS} ${CURTC_DEFINES}" > ${CURTC_BIN}.command
      COMMAND_EXPAND_LISTS
      VERBATIM
  )
  create_binary_resource(${CURTC_BIN}.command ${CURTC_BIN}.command.o)

  set(SRCS ${SRCS} ${CURTC_BIN}.src.o ${CURTC_BIN}.command.o)
endif()
# -------------------------------- End RTC -------------------------------------------------------

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PRIVATE_INCLUDE_DIRECTORIES
      ${CMAKE_SOURCE_DIR}/Detectors/Base/src
      ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
      ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
    PUBLIC_LINK_LIBRARIES O2::GPUTracking O2::ITStrackingCUDA O2::FrameworkFoundation3rdparty
    TARGETVARNAME targetName)

  target_compile_definitions(
    ${targetName} PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

  set_target_properties(${targetName} PROPERTIES LINKER_LANGUAGE CXX)

  install(FILES ${HDRS} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  set(targetName Ali${MODULE})
  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("${targetName}" "" "GPUReconstructionCUDA.h"
                      "${incdirs} .")

  # Generate the ROOT map Dependecies
  generate_rootmap("${targetName}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(${targetName} SHARED ${SRCS} G__${targetName}.cxx)
  # CUDA run-time and driver
  target_link_libraries(${targetName} PUBLIC AliGPUTracking)

  # Additional compilation flags
  set_target_properties(${targetName} PROPERTIES COMPILE_FLAGS "")

  # Installation
  install(TARGETS ${targetName} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS} DESTINATION include)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} PUBLIC GPUTracking)
  set(targetName ${MODULE})
  install(TARGETS GPUTrackingCUDA)
endif()

target_link_libraries(${targetName} PRIVATE cuda cudart nvrtc)
set_target_cuda_arch(${targetName})

if(OpenMP_CXX_FOUND)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
endif()
