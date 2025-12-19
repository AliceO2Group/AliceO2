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

set(SRCS GPUReconstructionCUDA.cu GPUReconstructionCUDAGenRTC.cxx GPUReconstructionCUDAKernels.cu GPUReconstructionCUDARTCCalls.cu)
set(HDRS GPUReconstructionCUDA.h GPUReconstructionCUDAInternals.h GPUReconstructionCUDAHelpers.inc GPUReconstructionCUDADef.h GPUReconstructionCUDAIncludesSystem.h)
# -------------------------------- Prepare RTC -------------------------------------------------------
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
set(GPU_RTC_FLAGS_ARCH "")
if(CUDA_COMPUTETARGET)
    foreach(CUDA_ARCH ${CUDA_COMPUTETARGET})
        string(REGEX REPLACE "-.*$" "" CUDA_ARCH_STRIPPED "${CUDA_ARCH}")
        set(GPU_RTC_FLAGS_ARCH "${GPU_RTC_FLAGS_ARCH} -gencode arch=compute_${CUDA_ARCH_STRIPPED},code=sm_${CUDA_ARCH_STRIPPED}")
    endforeach()
    list (GET CUDA_COMPUTETARGET 0 RTC_CUDA_ARCH)
    string(REGEX REPLACE "-.*$" "" RTC_CUDA_ARCH "${RTC_CUDA_ARCH}")
    set(RTC_CUDA_ARCH "${RTC_CUDA_ARCH}0")
else()
    set(RTC_CUDA_ARCH "750")
endif()
if(GPUCA_CUDA_GCCBIN)
    set(GPU_RTC_FLAGS "${GPU_RTC_FLAGS} --compiler-bindir ${GPUCA_CUDA_GCCBIN}")
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
    COMMAND cp ${GPUDIR}/Base/GPUStdSystemHeaders.h ${GPU_RTC_BIN}.src
    COMMAND cat ${GPUDIR}/Base/cuda/GPUReconstructionCUDAIncludesSystem.h | grep -v GPUStdSystemHeaders.h >> ${GPU_RTC_BIN}.src
    COMMAND cat ${GPUDIR}/Base/GPUStdSystemHeaders.h >> ${GPU_RTC_BIN}.src
    COMMAND ${CMAKE_CUDA_COMPILER} ${GPU_RTC_DEFINES} ${GPU_RTC_INCLUDES} -std=c++${CMAKE_CUDA_STANDARD} -D__CUDA_ARCH__=${RTC_CUDA_ARCH} -Wno-deprecated-gpu-targets -D__CUDACC__ -x c++ -M -MD -MT ${GPU_RTC_BIN}.src -MF ${GPU_RTC_BIN}.src.d ${GPU_RTC_SRC}
    COMMAND ${CMAKE_CUDA_COMPILER} ${GPU_RTC_DEFINES} ${GPU_RTC_INCLUDES} -std=c++${CMAKE_CUDA_STANDARD} -D__CUDA_ARCH__=${RTC_CUDA_ARCH} -Wno-deprecated-gpu-targets -D__CUDACC__ -x c++ -E -Xcompiler "-nostdinc -P" ${GPU_RTC_SRC} >> ${GPU_RTC_BIN}.src
    DEPENDS ${GPU_RTC_SRC} ${GPUDIR}/Base/GPUStdSystemHeaders.h ${GPUDIR}/Base/cuda/GPUReconstructionCUDAIncludesSystem.h ${GPUDIR}/Base/GPUStdSystemHeaders.h GPU_PARAM_HEADER_AUTO_ALL
    DEPFILE ${GPU_RTC_BIN}.src.d
    COMMAND_EXPAND_LISTS
    COMMENT "Preparing CUDA RTC source file ${GPU_RTC_BIN}.src"
)
create_binary_resource(${GPU_RTC_BIN}.src ${GPU_RTC_BIN}.src.o)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_CUDA_SRC_CHK.done
                    COMMAND ! grep "# [0-9]* \"\\(/usr/\\|.*GCC-Toolchain\\)" ${GPU_RTC_BIN}.src > ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_CUDA_SRC_CHK.done || bash -c "echo ERROR: CUDA RTC sources contain system headers 1>&2 && exit 1"
                    COMMENT "Checking CUDA RTC File ${GPU_RTC_BIN}.src for system headers"
                    DEPENDS ${GPU_RTC_BIN}.src VERBATIM)
add_custom_target(${MODULE}_CUDA_SRC_CHK ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_CUDA_SRC_CHK.done)

add_custom_command(
    OUTPUT ${GPU_RTC_BIN}.command
    COMMAND echo -n "${CMAKE_CUDA_COMPILER} -forward-unknown-to-host-compiler ${GPU_RTC_DEFINES} ${GPU_RTC_FLAGS_SEPARATED} -x cu -fatbin -Xcudafe --diag_suppress=177" > ${GPU_RTC_BIN}.command
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "Preparing CUDA RTC command file ${GPU_RTC_BIN}.command"
)
create_binary_resource(${GPU_RTC_BIN}.command ${GPU_RTC_BIN}.command.o)

add_custom_command(
    OUTPUT ${GPU_RTC_BIN}.command.arch
    COMMAND echo -n "${GPU_RTC_FLAGS_ARCH}" > ${GPU_RTC_BIN}.command.arch
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "Preparing CUDA RTC ARCH command file ${GPU_RTC_BIN}.command.arch"
)
create_binary_resource(${GPU_RTC_BIN}.command.arch ${GPU_RTC_BIN}.command.arch.o)

add_custom_command(
    OUTPUT ${GPU_RTC_BIN}.command.no_fast_math
    COMMAND echo -n "${GPUCA_CUDA_NO_FAST_MATH_FLAGS}" > ${GPU_RTC_BIN}.command.no_fast_math
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "Preparing CUDA RTC NO_FAST_MATH command file ${GPU_RTC_BIN}.command.arch"
)
create_binary_resource(${GPU_RTC_BIN}.command.no_fast_math ${GPU_RTC_BIN}.command.no_fast_math.o)

set(SRCS ${SRCS} ${GPU_RTC_BIN}.src.o ${GPU_RTC_BIN}.command.o ${GPU_RTC_BIN}.command.arch.o ${GPU_RTC_BIN}.command.no_fast_math.o)
# -------------------------------- End RTC -------------------------------------------------------

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PUBLIC_LINK_LIBRARIES O2::GPUTracking O2::ITStrackingCUDA
    PRIVATE_INCLUDE_DIRECTORIES
      ${CMAKE_SOURCE_DIR}/Detectors/Base/src
      ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
      ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
      ${CMAKE_CURRENT_SOURCE_DIR}
    TARGETVARNAME targetName)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(targetName "${MODULE}")
  add_library(${MODULE} SHARED ${SRCS})
  add_library(O2::${MODULE} ALIAS ${MODULE})
  target_link_libraries(${MODULE} PUBLIC O2::GPUTracking)
  install(TARGETS GPUTrackingCUDA)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU)

target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

if (onnxruntime_FOUND)
  target_compile_definitions(${targetName} PRIVATE
                                           $<$<BOOL:${ORT_CUDA_BUILD}>:ORT_CUDA_BUILD>
                                           $<$<BOOL:${ORT_TENSORRT_BUILD}>:ORT_TENSORRT_BUILD>)
  target_link_libraries(${targetName} PRIVATE onnxruntime::onnxruntime)
endif()

# Setting target architecture and adding GPU libraries
target_link_libraries(${targetName} PRIVATE cuda cudart)
set_target_gpu_arch("CUDA" ${targetName})
#target_link_options(${targetName} PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version_script.ld")
#set_target_properties(${targetName} PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version_script.ld)

target_link_libraries(${targetName} PRIVATE TBB::tbb)

# Special handling of GPU kernels in case of per-kernel compilation / RDC
if(NOT DEFINED GPUCA_CUDA_COMPILE_MODE)
  set(GPUCA_CUDA_COMPILE_MODE "perkernel")
endif()
if(GPUCA_CUDA_COMPILE_MODE STREQUAL "onefile")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=0)
elseif(GPUCA_CUDA_COMPILE_MODE STREQUAL "perkernel")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=1)
  if(NOT DEFINED GPUCA_RTC_NO_COMPILED_KERNELS OR NOT GPUCA_RTC_NO_COMPILED_KERNELS)
    add_library(GPUTrackingCUDAKernels OBJECT $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_>,APPEND,.cu>, >)
    set_property(TARGET GPUTrackingCUDAKernels PROPERTY CUDA_FATBIN_COMPILATION ON)
    set_property(TARGET GPUTrackingCUDAKernels PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_definitions(GPUTrackingCUDAKernels PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
    target_include_directories(GPUTrackingCUDAKernels PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
    target_link_libraries(GPUTrackingCUDAKernels PRIVATE $<TARGET_PROPERTY:${targetName},LINK_LIBRARIES>)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda_kernel_module_fatbin)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o
      COMMAND cp -u $<TARGET_OBJECTS:GPUTrackingCUDAKernels> ${CMAKE_CURRENT_BINARY_DIR}/cuda_kernel_module_fatbin/
      COMMAND ${CMAKE_LINKER} -z noexecstack --relocatable --format binary --output ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o $<PATH:RELATIVE_PATH,$<LIST:TRANSFORM,$<PATH:GET_FILENAME,$<TARGET_OBJECTS:GPUTrackingCUDAKernels>>,PREPEND,${CMAKE_CURRENT_BINARY_DIR}/cuda_kernel_module_fatbin/>,${CMAKE_CURRENT_BINARY_DIR}>
      DEPENDS GPUTrackingCUDAKernels $<TARGET_OBJECTS:GPUTrackingCUDAKernels>
      COMMENT "Compiling fatbin kernels ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o"
      VERBATIM
      COMMAND_EXPAND_LISTS
    )
    target_sources(${targetName} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingCUDAKernelModules.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
  else()
    set_source_files_properties(GPUReconstructionCUDA.cu PROPERTIES COMPILE_DEFINITIONS GPUCA_RTC_NO_COMPILED_KERNELS)
  endif()

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

add_library(GPUTrackingCUDAExternalProvider OBJECT GPUReconstructionCUDAExternalProvider.cu)
add_library(O2::GPUTrackingCUDAExternalProvider ALIAS GPUTrackingCUDAExternalProvider)
set_property(TARGET GPUTrackingCUDAExternalProvider PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(GPUTrackingCUDAExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
target_include_directories(GPUTrackingCUDAExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
add_dependencies(GPUTrackingCUDAExternalProvider O2::GPUTracking) # must not depend on GPU backend to avoid cyclic dependencies
