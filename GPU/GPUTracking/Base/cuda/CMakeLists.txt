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

# -------------------------------- Status Message -------------------------------------------------------
if(DEFINED CUDA_COMPUTETARGET)
  set(TMP_TARGET "(Compute Target ${CUDA_COMPUTETARGET})")
endif()
message(STATUS "Building GPUTracking with CUDA support ${TMP_TARGET}")

set(SRCS GPUReconstructionCUDA.cu GPUReconstructionCUDAGenRTC.cxx GPUReconstructionCUDAKernels.cu)
set(HDRS GPUReconstructionCUDA.h GPUReconstructionCUDAInternals.h GPUReconstructionCUDADef.h GPUReconstructionCUDAIncludes.h CUDAThrustHelpers.h)
# -------------------------------- Prepare RTC -------------------------------------------------------
if(NOT ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  enable_language(ASM)
  if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(defineIncludeSrc "O2::${MODULE}")
  else()
    set(defineIncludeSrc "${MODULE}")
  endif()
  set(GPU_RTC_DEFINES "-D$<JOIN:$<TARGET_PROPERTY:${defineIncludeSrc},COMPILE_DEFINITIONS>,$<SEMICOLON>-D>")
  set(GPU_RTC_INCLUDES "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:${defineIncludeSrc},INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>"
                    -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
                    -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
  )
  if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(GPU_RTC_INCLUDES ${GPU_RTC_INCLUDES} "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:O2::ITStrackingCUDA,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>")
  endif()
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep")

  # build flags to use for RTC
  set(GPU_RTC_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -std=c++${CMAKE_CUDA_STANDARD}")
  if(CUDA_COMPUTETARGET)
      foreach(CUDA_ARCH ${CUDA_COMPUTETARGET})
          set(GPU_RTC_FLAGS "${GPU_RTC_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
      endforeach()
      list (GET CUDA_COMPUTETARGET 0 RTC_CUDA_ARCH)
      set(RTC_CUDA_ARCH "${RTC_CUDA_ARCH}0")
  else()
      set(RTC_CUDA_ARCH "750")
  endif()
  set(GPU_RTC_FLAGS_SEPARATED "${GPU_RTC_FLAGS}")
  separate_arguments(GPU_RTC_FLAGS_SEPARATED)

  # convenience variables
  if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
    get_filename_component(GPUDIR ${CMAKE_SOURCE_DIR}/../ ABSOLUTE)
  else()
    set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
  endif()

  set(GPU_RTC_SRC ${GPUDIR}/Base/cuda/GPUReconstructionCUDArtc.cu)
  set(GPU_RTC_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionCUDArtc)

  # cmake-format: off
  add_custom_command(
      OUTPUT ${GPU_RTC_BIN}.src
      COMMAND cat ${GPUDIR}/Base/cuda/GPUReconstructionCUDAIncludes.h > ${GPU_RTC_BIN}.src
      COMMAND ${CMAKE_CXX_COMPILER} ${GPU_RTC_DEFINES} ${GPU_RTC_INCLUDES} -std=c++${CMAKE_CUDA_STANDARD} -D__CUDA_ARCH__=${RTC_CUDA_ARCH} -D__CUDACC__ -x c++ -E ${GPU_RTC_SRC} >> ${GPU_RTC_BIN}.src
      MAIN_DEPENDENCY ${GPU_RTC_SRC}
      IMPLICIT_DEPENDS CXX ${GPU_RTC_SRC}
      COMMAND_EXPAND_LISTS
      COMMENT "Preparing CUDA RTC source file ${GPU_RTC_BIN}.src"
  )
  create_binary_resource(${GPU_RTC_BIN}.src ${GPU_RTC_BIN}.src.o)

  add_custom_command(
      OUTPUT ${GPU_RTC_BIN}.command
      COMMAND echo -n "${CMAKE_CUDA_COMPILER} ${GPU_RTC_FLAGS_SEPARATED} ${GPU_RTC_DEFINES} -fatbin" > ${GPU_RTC_BIN}.command
      COMMAND_EXPAND_LISTS
      VERBATIM
      COMMENT "Preparing CUDA RTC command file ${GPU_RTC_BIN}.command"
  )
  create_binary_resource(${GPU_RTC_BIN}.command ${GPU_RTC_BIN}.command.o)

  set(SRCS ${SRCS} ${GPU_RTC_BIN}.src.o ${GPU_RTC_BIN}.command.o)
endif()
# -------------------------------- End RTC -------------------------------------------------------

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  set(TMP_BASELIB O2::GPUTracking)
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PUBLIC_LINK_LIBRARIES ${TMP_BASELIB} O2::ITStrackingCUDA
    PRIVATE_INCLUDE_DIRECTORIES
      ${CMAKE_SOURCE_DIR}/Detectors/Base/src
      ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
      ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
      ${CMAKE_CURRENT_SOURCE_DIR}
    TARGETVARNAME targetName)

  target_compile_definitions(${targetName} PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

  install(FILES ${HDRS} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  set(targetName "Ali${MODULE}")
  set(TMP_BASELIB AliGPUTracking)
  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("${targetName}" "" "GPUReconstructionCUDA.h" "${incdirs} .")

  # Generate the ROOT map Dependecies
  generate_rootmap("${targetName}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(${targetName} SHARED ${SRCS} G__${targetName}.cxx)
  target_link_libraries(${targetName} PUBLIC ${TMP_BASELIB})

  # Installation
  install(TARGETS ${targetName} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS} DESTINATION include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(targetName "${MODULE}")
  set(TMP_BASELIB GPUTracking)
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} PUBLIC ${TMP_BASELIB})
  install(TARGETS GPUTrackingCUDA)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Setting target architecture and adding GPU libraries
target_link_libraries(${targetName} PRIVATE cuda cudart nvrtc)
set_target_cuda_arch(${targetName})

if(OpenMP_CXX_FOUND)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
  target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
endif()

# Special handling of GPU kernels in case of per-kernel compilation / RDC
if(NOT DEFINED GPUCA_CUDA_COMPILE_MODE)
  if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    set(GPUCA_CUDA_COMPILE_MODE "onefile")
  else()
    set(GPUCA_CUDA_COMPILE_MODE "perkernel")
  endif()
endif()
if(GPUCA_CUDA_COMPILE_MODE STREQUAL "onefile")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=0)
elseif(GPUCA_CUDA_COMPILE_MODE STREQUAL "perkernel")
  add_library(GPUTrackingCUDAKernels OBJECT $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_>,APPEND,.cu>, >)
  set_property(TARGET GPUTrackingCUDAKernels PROPERTY CUDA_FATBIN_COMPILATION ON)
  set_property(TARGET GPUTrackingCUDAKernels PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=1)
  target_compile_definitions(GPUTrackingCUDAKernels PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
  target_include_directories(GPUTrackingCUDAKernels PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda_kernel_module_fatbin)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o
    COMMAND cp -u $<TARGET_OBJECTS:GPUTrackingCUDAKernels> ${CMAKE_CURRENT_BINARY_DIR}/cuda_kernel_module_fatbin/
    COMMAND ${CMAKE_LINKER} --relocatable --format binary --output ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o $<PATH:RELATIVE_PATH,$<LIST:TRANSFORM,$<PATH:GET_FILENAME,$<TARGET_OBJECTS:GPUTrackingCUDAKernels>>,PREPEND,${CMAKE_CURRENT_BINARY_DIR}/cuda_kernel_module_fatbin/>,${CMAKE_CURRENT_BINARY_DIR}>
    DEPENDS GPUTrackingCUDAKernels $<TARGET_OBJECTS:GPUTrackingCUDAKernels>
    COMMENT "Compiling fatbin kernels ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
  target_sources(${targetName} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)

  # Disable all non-deterministic floating point to make TPC track model encoding / decoding precise
  set_source_files_properties(${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCCompressionKernels_step0attached.cu
                              ${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCCompressionKernels_step1unattached.cu
                              ${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCDecompressionKernels_step0attached.cu
                              ${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCDecompressionKernels_step1unattached.cu
                              TARGET_DIRECTORY ${targetName}
                              PROPERTIES
                              COMPILE_FLAGS "${GPUCA_CUDA_NO_FAST_MATH_FLAGS}"
                              COMPILE_DEFINITIONS "GPUCA_NO_FAST_MATH")
elseif(GPUCA_CUDA_COMPILE_MODE STREQUAL "rdc")
  message(FATAL_ERROR "CUDA RDC compilation of GPUReconstruction ios not yet working!")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=2)
  set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_sources(${targetName} PRIVATE $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_>,APPEND,.cu>, >)
else()
  message(FATAL_ERROR "Invalid compile mode")
endif()
if(NOT GPUCA_CUDA_COMPILE_MODE STREQUAL "rdc")
  set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
  set_target_properties(${targetName} PROPERTIES LINKER_LANGUAGE CXX)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR CONFIG_O2_EXTENSIONS)
  add_library(GPUTrackingCUDAExternalProvider OBJECT GPUReconstructionCUDAExternalProvider.cu)
  add_library(O2::GPUTrackingCUDAExternalProvider ALIAS GPUTrackingCUDAExternalProvider)
  set_property(TARGET GPUTrackingCUDAExternalProvider PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_compile_definitions(GPUTrackingCUDAExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
  target_include_directories(GPUTrackingCUDAExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
endif()
