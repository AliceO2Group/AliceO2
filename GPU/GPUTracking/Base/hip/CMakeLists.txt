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

set(MODULE GPUTrackingHIP)

# set(GPUCA_HIP_HIPIFY_FROM_CUDA 0) # Use local HIP source files

if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
  set(GPUCA_HIP_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipify)
  file(MAKE_DIRECTORY ${GPUCA_HIP_SOURCE_DIR})
  set(GPUCA_HIP_FILE_LIST GPUReconstructionCUDA.cu GPUReconstructionCUDAExternalProvider.cu GPUReconstructionCUDA.h GPUReconstructionCUDAInternals.h GPUReconstructionCUDAkernel.template.cu CUDAThrustHelpers.h GPUReconstructionCUDADef.h GPUReconstructionCUDAGenRTC.cxx GPUReconstructionCUDAKernels.cu GPUReconstructionCUDArtc.cu)
  set(GPUCA_HIP_LOCAL_FILE_LIST GPUReconstructionHIPIncludes.h)
  set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")
  set(HIP_SOURCES "")
  foreach(file ${GPUCA_HIP_FILE_LIST})
    get_filename_component(ABS_CUDA_SORUCE ../cuda/${file} ABSOLUTE)
    get_filename_component(CUDA_SOURCE ${file} NAME)
    string(REPLACE ".cu" ".hip" HIP_SOURCE1 ${CUDA_SOURCE})
    string(REPLACE "CUDA" "HIP" HIP_SOURCE ${HIP_SOURCE1})
    add_custom_command(
      OUTPUT ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
      COMMAND ${HIPIFY_EXECUTABLE} --quiet-warnings ${ABS_CUDA_SORUCE} | sed -e 's/CUDA/HIP/g' -e 's/cuda/hip/g' > ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
      DEPENDS ${ABS_CUDA_SORUCE}
      COMMENT "Hippifying ${HIP_SOURCE}"
    )
    list(APPEND HIP_SOURCES "${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}")
  endforeach()
  foreach(file ${GPUCA_HIP_LOCAL_FILE_LIST})
    get_filename_component(ABS_SORUCE ${file} ABSOLUTE)
    get_filename_component(HIP_SOURCE ${file} NAME)
    add_custom_command(
      OUTPUT ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
      COMMAND cp ${ABS_SORUCE} ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
      DEPENDS ${ABS_SORUCE}
      COMMENT "Copying ${HIP_SOURCE}"
    )
    list(APPEND HIP_SOURCES "${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}")
  endforeach()
else()
  get_filename_component(GPUCA_HIP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
endif()

# Setting flags as a global option for all HIP targets.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fno-gpu-rdc")
if(DEFINED HIP_AMDGPUTARGET)
  set(TMP_TARGET "(GPU Target ${HIP_AMDGPUTARGET})")
endif()

message(STATUS "Building GPUTracking with HIP support ${TMP_TARGET}")

set(SRCS ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIP.hip ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPKernels.hip)
set(SRCS_CXX ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPGenRTC.cxx)
set(HDRS ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIP.h ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPInternals.h ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPDef.h ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPIncludes.h ${GPUCA_HIP_SOURCE_DIR}/HIPThrustHelpers.h)

# -------------------------------- Prepare RTC -------------------------------------------------------
if(NOT ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  enable_language(ASM)
  if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(defineIncludeSrc "O2::${MODULE}")
  else()
    set(defineIncludeSrc "${MODULE}")
  endif()
  set(HIPRTC_DEFINES "-D$<JOIN:$<TARGET_PROPERTY:${defineIncludeSrc},COMPILE_DEFINITIONS>,$<SEMICOLON>-D>")
  set(HIPRTC_INCLUDES "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:${defineIncludeSrc},INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>"
                    -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
                    -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
  )
  if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(HIPRTC_INCLUDES ${HIPRTC_INCLUDES} "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:O2::ITStrackingHIP,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>")
  endif()

  set(HIPRTC_FLAGS "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -std=c++${CMAKE_HIP_STANDARD}")
  separate_arguments(HIPRTC_FLAGS)

  # convenience variables
  if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
    get_filename_component(GPUDIR ${CMAKE_SOURCE_DIR}/../ ABSOLUTE)
  else()
    set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
  endif()

  set(HIPRTC_SRC ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPrtc.hip)
  set(HIPRTC_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionHIPrtc)

  # cmake-format: off
  add_custom_command(
      OUTPUT ${HIPRTC_BIN}.src
      COMMAND cat ${GPUDIR}/Base/hip/GPUReconstructionHIPIncludes.h > ${HIPRTC_BIN}.src
      COMMAND ${CMAKE_CXX_COMPILER} ${HIPRTC_DEFINES} ${HIPRTC_INCLUDES} -std=c++${CMAKE_HIP_STANDARD} -D__HIPCC__ -D__HIP_DEVICE_COMPILE__ -x c++ -E ${HIPRTC_SRC} >> ${HIPRTC_BIN}.src
      MAIN_DEPENDENCY ${HIPRTC_SRC}
      IMPLICIT_DEPENDS CXX ${HIPRTC_SRC}
      COMMAND_EXPAND_LISTS
      COMMENT "Preparing HIP RTC source file ${HIPRTC_BIN}.src"
  )
  create_binary_resource(${HIPRTC_BIN}.src ${HIPRTC_BIN}.src.o)

  add_custom_command(
      OUTPUT ${HIPRTC_BIN}.command
      COMMAND echo -n "${CMAKE_HIP_COMPILER} ${HIPRTC_FLAGS} ${HIPRTC_DEFINES}" > ${HIPRTC_BIN}.command
      COMMAND_EXPAND_LISTS
      VERBATIM
      COMMENT "Preparing HIP RTC command file ${HIPRTC_BIN}.command"
  )
  create_binary_resource(${HIPRTC_BIN}.command ${HIPRTC_BIN}.command.o)

  set(SRCS ${SRCS} ${HIPRTC_BIN}.src.o ${HIPRTC_BIN}.command.o)
endif()
# -------------------------------- End RTC -------------------------------------------------------

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  set(TMP_BASELIB O2::GPUTracking)
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PUBLIC_LINK_LIBRARIES ${TMP_BASELIB} O2::ITStrackingHIP hip::host hip::device hip::hipcub roc::rocthrust
    PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
                               ${CMAKE_SOURCE_DIR}/Detectors/Base/src
                               ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
    TARGETVARNAME targetName)

  target_compile_definitions(
    ${targetName} PUBLIC GPUCA_GPULIBRARY=HIP
    $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

  install(FILES ${HDRS} DESTINATION include/GPU)

#  o2_add_test(GPUsortHIP NAME test_GPUsortHIP
#                  SOURCES test/testGPUsortHIP.hip.cxx
#                  PUBLIC_LINK_LIBRARIES O2::GPUCommon hip::host hip::device hip::hipcub roc::rocthrust
#                  COMPONENT_NAME GPU
#                  LABELS gpu)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 14)
    message(FATAL_ERROR "HIP support needs at least C++14")
  endif()
  add_definitions(-DGPUCA_GPULIBRARY=HIP)

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("Ali${MODULE}" "" "GPUReconstructionHIP.h" "${incdirs} .")

  # Generate the ROOT map Dependecies
  generate_rootmap("Ali${MODULE}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE} SHARED ${SRCS} G__Ali${MODULE}.cxx)
  # ROOT Cannot load this library, so we have to remove it from the list
  list(REMOVE_ITEM ALILIBSTESTED Ali${MODULE})
  set(ALILIBSTESTED ${ALILIBSTESTED} CACHE INTERNAL "ALILIBSTESTED")

  # HIP run-time and driver
  set(TMP_BASELIB AliGPUTracking)
  target_link_libraries(Ali${MODULE} PUBLIC ${TMP_BASELIB} hip::host hip::device hip::hipcub roc::rocthrust)

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
  set(TMP_BASELIB GPUTracking)
  target_link_libraries(${MODULE} PUBLIC ${TMP_BASELIB} hip::host hip::device hip::hipcub roc::rocthrust)
  set(targetName "${MODULE}")
  install(TARGETS GPUTrackingHIP)
endif()

add_library(${MODULE}_CXX OBJECT ${SRCS_CXX})
target_compile_definitions(${MODULE}_CXX PRIVATE $<TARGET_PROPERTY:${TMP_BASELIB},COMPILE_DEFINITIONS>)
target_include_directories(${MODULE}_CXX PRIVATE $<TARGET_PROPERTY:${TMP_BASELIB},INCLUDE_DIRECTORIES>)
target_link_libraries(${targetName} PRIVATE ${MODULE}_CXX)

if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
  add_custom_target(${MODULE}_HIPIFIED DEPENDS ${HIP_SOURCES})
  add_dependencies(${targetName} ${MODULE}_HIPIFIED)
  add_dependencies(${MODULE}_CXX ${MODULE}_HIPIFIED)
endif()

if(OpenMP_CXX_FOUND)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${MODULE}_CXX PRIVATE WITH_OPENMP)
  target_link_libraries(${MODULE}_CXX PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_hip_arch(${targetName})
