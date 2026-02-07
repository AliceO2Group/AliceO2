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

if(GPUCA_BUILD_DEBUG AND GPUCA_BUILD_DEBUG_HOSTONLY)
  set(CMAKE_BUILD_TYPE RELEASE)
  set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -O3 -march=native -ggdb -fno-sanitize=all -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ggdb -fno-sanitize=all -DNDEBUG")
endif()

# -------------------------------- Options -------------------------------------------------------
# set(GPUCA_HIP_HIPIFY_FROM_CUDA 0) # Use local HIP source files

# -------------------------------- Status Message -------------------------------------------------------
if(DEFINED HIP_AMDGPUTARGET)
  set(TMP_TARGET "(GPU Target ${HIP_AMDGPUTARGET})")
endif()
message(STATUS "Building GPUTracking with HIP support ${TMP_TARGET}")

# -------------------------------- Optionally hipify from CUDA (default) -------------------------------------------------------
if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
  set(GPUCA_HIP_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipify)
  file(MAKE_DIRECTORY ${GPUCA_HIP_SOURCE_DIR})
  set(GPUCA_HIP_FILE_LIST GPUReconstructionCUDA.cu GPUReconstructionCUDAExternalProvider.cu GPUReconstructionCUDA.h GPUReconstructionCUDAInternals.h GPUReconstructionCUDAHelpers.inc GPUReconstructionCUDAkernel.template.cu GPUReconstructionCUDADef.h GPUReconstructionCUDAGenRTC.cxx GPUReconstructionCUDAKernels.cu GPUReconstructionCUDAKernelsSpecialize.inc GPUReconstructionCUDArtc.cu GPUReconstructionCUDARTCCalls.cu)
  set(GPUCA_HIP_LOCAL_FILE_LIST GPUReconstructionHIPIncludesSystem.h)
  set(HIP_SOURCES "")
  foreach(file ${GPUCA_HIP_FILE_LIST})
    get_filename_component(ABS_CUDA_SORUCE ../cuda/${file} ABSOLUTE)
    get_filename_component(CUDA_SOURCE ${file} NAME)
    get_filename_component(CUDA_SOURCE_EXT ${file} EXT)
    string(REPLACE ".cu" ".hip" HIP_SOURCE1 ${CUDA_SOURCE})
    string(REPLACE "CUDA" "HIP" HIP_SOURCE ${HIP_SOURCE1})
    if(CUDA_SOURCE_EXT STREQUAL ".cu" OR CUDA_SOURCE_EXT STREQUAL ".h")
      add_custom_command(
        OUTPUT ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
        COMMAND ${hip_HIPIFY_PERL_EXECUTABLE} --quiet-warnings ${ABS_CUDA_SORUCE} | sed -e 's/CUDA/HIP/g' -e 's/cuda/hip/g' > ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
        DEPENDS ${ABS_CUDA_SORUCE}
        COMMENT "Hippifying ${HIP_SOURCE}"
      )
    else()
      add_custom_command(
        OUTPUT ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
        COMMAND sed -e 's/CUDA/HIP/g' -e 's/cuda/hip/g' ${ABS_CUDA_SORUCE} > ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
        DEPENDS ${ABS_CUDA_SORUCE}
        COMMENT "Generating HIP source ${HIP_SOURCE}"
      )
    endif()
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

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_HIPIFIED_CHK.done
                     COMMAND bash -c "diff -u <(sed 's,//.*$,,g' ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip) <(sed 's,//.*$,,g' ${CMAKE_CURRENT_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip) > ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_HIPIFIED_CHK.done"
                     DEPENDS ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip ${CMAKE_CURRENT_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip
                     VERBATIM
                     COMMENT "Checking HIPified file ${CMAKE_CURRENT_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip")
  add_custom_target(${MODULE}_HIPIFIED_CHK ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_HIPIFIED_CHK.done)
else()
  get_filename_component(GPUCA_HIP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
endif()

set(SRCS ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIP.hip ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPKernels.hip ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPRTCCalls.hip)
set(SRCS_CXX ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPGenRTC.cxx)
set(HDRS ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIP.h ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPInternals.h ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPHelpers.inc ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPDef.h ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPIncludesSystem.h)

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
  set(GPU_RTC_INCLUDES ${GPU_RTC_INCLUDES} "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:O2::ITStrackingHIP,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>")
endif()

# build flags to use for RTC
set(GPU_RTC_FLAGS "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_${CMAKE_BUILD_TYPE_UPPER}} -std=c++${CMAKE_HIP_STANDARD}")
set(GPU_RTC_FLAGS_ARCH "")
foreach(HIP_ARCH ${CMAKE_HIP_ARCHITECTURES})
  set(GPU_RTC_FLAGS_ARCH "${GPU_RTC_FLAGS_ARCH} --offload-arch=${HIP_ARCH}")
endforeach()

set(GPU_RTC_FLAGS_SEPARATED "${GPU_RTC_FLAGS}")
separate_arguments(GPU_RTC_FLAGS_SEPARATED)

# convenience variables
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  get_filename_component(GPUDIR ${CMAKE_SOURCE_DIR}/../ ABSOLUTE)
else()
  set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
endif()

set(GPU_RTC_SRC ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPrtc.hip)
set(GPU_RTC_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionHIPrtc)

# cmake-format: off
add_custom_command(
    OUTPUT ${GPU_RTC_BIN}.src
    COMMAND cp ${GPUDIR}/Base/GPUStdSystemHeaders.h ${GPU_RTC_BIN}.src
    COMMAND cat ${GPUDIR}/Base/hip/GPUReconstructionHIPIncludesSystem.h | grep -v GPUStdSystemHeaders.h >> ${GPU_RTC_BIN}.src
    COMMAND cat ${GPUDIR}/Base/GPUStdSystemHeaders.h >> ${GPU_RTC_BIN}.src
    COMMAND ${CMAKE_HIP_COMPILER} ${GPU_RTC_DEFINES} ${GPU_RTC_INCLUDES} -std=c++${CMAKE_HIP_STANDARD} -D__HIPCC__ -D__HIP_DEVICE_COMPILE__ -x c++ -nostdinc -E -P ${GPU_RTC_SRC} -MD -MT ${GPU_RTC_BIN}.src -MF ${GPU_RTC_BIN}.src.d >> ${GPU_RTC_BIN}.src
    DEPENDS ${GPU_RTC_SRC} ${GPUDIR}/Base/GPUStdSystemHeaders.h ${GPUDIR}/Base/hip/GPUReconstructionHIPIncludesSystem.h ${GPUDIR}/Base/GPUStdSystemHeaders.h ${MODULE}_HIPIFIED GPU_PARAM_HEADER_TARGET
    DEPFILE ${GPU_RTC_BIN}.src.d
    COMMAND_EXPAND_LISTS
    COMMENT "Preparing HIP RTC source file ${GPU_RTC_BIN}.src"
)
create_binary_resource(${GPU_RTC_BIN}.src ${GPU_RTC_BIN}.src.o)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_HIP_SRC_CHK.done
                    COMMAND ! grep "# [0-9]* \"\\(/usr/\\|.*GCC-Toolchain\\)" ${GPU_RTC_BIN}.src > ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_HIP_SRC_CHK.done || bash -c "echo ERROR: HIP RTC sources contain system headers 1>&2 && exit 1"
                    COMMENT "Checking HIP RTC File ${GPU_RTC_BIN}.src for system headers"
                    DEPENDS ${GPU_RTC_BIN}.src VERBATIM)
add_custom_target(${MODULE}_HIP_SRC_CHK ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MODULE}_HIP_SRC_CHK.done)

add_custom_command(
    OUTPUT ${GPU_RTC_BIN}.command
    COMMAND echo -n "${CMAKE_HIP_COMPILER} ${GPU_RTC_DEFINES} ${GPU_RTC_FLAGS_SEPARATED} -x hip --cuda-device-only -Wno-unused-const-variable" > ${GPU_RTC_BIN}.command
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "Preparing HIP RTC command file ${GPU_RTC_BIN}.command"
)
create_binary_resource(${GPU_RTC_BIN}.command ${GPU_RTC_BIN}.command.o)

add_custom_command(
    OUTPUT ${GPU_RTC_BIN}.command.arch
    COMMAND echo -n "${GPU_RTC_FLAGS_ARCH}" > ${GPU_RTC_BIN}.command.arch
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "Preparing HIP RTC ARCH command file ${GPU_RTC_BIN}.command.arch"
)
create_binary_resource(${GPU_RTC_BIN}.command.arch ${GPU_RTC_BIN}.command.arch.o)

add_custom_command(
    OUTPUT ${GPU_RTC_BIN}.command.no_fast_math
    COMMAND echo -n "${GPUCA_CXX_NO_FAST_MATH_FLAGS}" > ${GPU_RTC_BIN}.command.no_fast_math
    COMMAND_EXPAND_LISTS VERBATIM
    COMMENT "Preparing HIP RTC NO_FAST_MATH command file ${GPU_RTC_BIN}.command.no_fast_math"
)
create_binary_resource(${GPU_RTC_BIN}.command.no_fast_math ${GPU_RTC_BIN}.command.no_fast_math.o)

set(SRCS ${SRCS} ${GPU_RTC_BIN}.src.o ${GPU_RTC_BIN}.command.o ${GPU_RTC_BIN}.command.arch.o ${GPU_RTC_BIN}.command.no_fast_math.o)
# -------------------------------- End RTC -------------------------------------------------------

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PUBLIC_LINK_LIBRARIES O2::GPUTracking O2::ITStrackingHIP
    PRIVATE_INCLUDE_DIRECTORIES
      ${CMAKE_SOURCE_DIR}/Detectors/Base/src
      ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
      ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
      ${GPUCA_HIP_SOURCE_DIR}
    TARGETVARNAME targetName)

#  o2_add_test(GPUsortHIP NAME test_GPUsortHIP
#                  SOURCES test/testGPUsortHIP.hip
#                  PUBLIC_LINK_LIBRARIES O2::GPUCommon hip::host hip::device hip::hipcub roc::rocthrust
#                  COMPONENT_NAME GPU
#                  LABELS gpu)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(targetName "${MODULE}")
  add_library(${MODULE} SHARED ${SRCS})
  add_library(O2::${MODULE} ALIAS ${MODULE})
  target_link_libraries(${MODULE} PUBLIC O2::GPUTracking)
  install(TARGETS GPUTrackingHIP)
  include_directories(${GPUCA_HIP_SOURCE_DIR})
endif()

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU)

target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

if (onnxruntime_FOUND)
  target_compile_definitions(${targetName} PRIVATE
                                           $<$<BOOL:${ORT_ROCM_BUILD}>:ORT_ROCM_BUILD>
                                           $<$<BOOL:${ORT_MIGRAPHX_BUILD}>:ORT_MIGRAPHX_BUILD>)
  target_link_libraries(${targetName} PRIVATE onnxruntime::onnxruntime)
endif()

add_library(${MODULE}_CXX OBJECT ${SRCS_CXX}) # Adding a C++ library for the .cxx code of the HIP library, such that it does not link to HIP libraries, and CMake HIP Language doesn't add HIP compile flags.
target_compile_definitions(${MODULE}_CXX PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
target_include_directories(${MODULE}_CXX PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,INCLUDE_DIRECTORIES>)
target_link_libraries(${MODULE}_CXX PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,LINK_LIBRARIES>)
add_dependencies(${MODULE}_CXX O2::GPUTracking)
target_link_libraries(${targetName} PRIVATE ${MODULE}_CXX)

if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
  add_custom_target(${MODULE}_HIPIFIED DEPENDS ${HIP_SOURCES})
  add_dependencies(${targetName} ${MODULE}_HIPIFIED)
  add_dependencies(${MODULE}_CXX ${MODULE}_HIPIFIED)
endif()

# Setting target architecture and adding GPU libraries
target_link_libraries(${targetName} PRIVATE hip::host hip::device hip::hipcub roc::rocthrust)
set_target_gpu_arch("HIP" ${targetName})

target_link_libraries(${MODULE}_CXX PRIVATE TBB::tbb)

# Special handling of GPU kernels in case of per-kernel compilation / RDC
if(NOT DEFINED GPUCA_HIP_COMPILE_MODE)
  set(GPUCA_HIP_COMPILE_MODE "perkernel")
endif()
if(GPUCA_HIP_COMPILE_MODE STREQUAL "onefile")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=0)
elseif(GPUCA_HIP_COMPILE_MODE STREQUAL "perkernel")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=1)
  if(NOT DEFINED GPUCA_RTC_NO_COMPILED_KERNELS OR NOT GPUCA_RTC_NO_COMPILED_KERNELS)
    add_library(GPUTrackingHIPKernels OBJECT $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_>,APPEND,.hip>, >)
    target_compile_options(GPUTrackingHIPKernels PRIVATE "--cuda-device-only")
    target_compile_options(GPUTrackingHIPKernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>)
    target_link_options(GPUTrackingHIPKernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>)
    target_compile_definitions(GPUTrackingHIPKernels PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
    target_include_directories(GPUTrackingHIPKernels PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
    target_link_libraries(GPUTrackingHIPKernels PRIVATE $<TARGET_PROPERTY:${targetName},LINK_LIBRARIES>)
    if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
      add_dependencies(GPUTrackingHIPKernels ${MODULE}_HIPIFIED)
    endif()
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_module_fatbin)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o
      COMMAND cp -u $<TARGET_OBJECTS:GPUTrackingHIPKernels> ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_module_fatbin/
      COMMAND ${CMAKE_LINKER} -z noexecstack --relocatable --format binary --output ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o $<PATH:RELATIVE_PATH,$<LIST:TRANSFORM,$<PATH:GET_FILENAME,$<TARGET_OBJECTS:GPUTrackingHIPKernels>>,PREPEND,${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_module_fatbin/>,${CMAKE_CURRENT_BINARY_DIR}>
      DEPENDS GPUTrackingHIPKernels $<TARGET_OBJECTS:GPUTrackingHIPKernels>
      COMMENT "Compiling fatbin kernels ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o"
      VERBATIM
      COMMAND_EXPAND_LISTS
    )
    target_sources(${targetName} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
  else()
    set_source_files_properties(${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIP.hip PROPERTIES COMPILE_DEFINITIONS GPUCA_RTC_NO_COMPILED_KERNELS)
  endif()

elseif(GPUCA_HIP_COMPILE_MODE STREQUAL "rdc")
  message(FATAL_ERROR "HIP RDC compilation of GPUReconstruction ios not yet working!")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=2)
  target_compile_options(${targetName} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
  target_link_options(${targetName} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
  target_sources(${targetName} PRIVATE $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_>,APPEND,.hip>, >)
else()
  message(FATAL_ERROR "Invalid compile mode")
endif()
if(NOT GPUCA_HIP_COMPILE_MODE STREQUAL "rdc")
  target_compile_options(${targetName} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>)
  target_link_options(${targetName} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>)
endif()

add_library(GPUTrackingHIPExternalProvider OBJECT ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPExternalProvider.hip)
add_library(O2::GPUTrackingHIPExternalProvider ALIAS GPUTrackingHIPExternalProvider)
target_compile_options(GPUTrackingHIPExternalProvider PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
target_link_options(GPUTrackingHIPExternalProvider PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
target_compile_definitions(GPUTrackingHIPExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
target_include_directories(GPUTrackingHIPExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
add_dependencies(GPUTrackingHIPExternalProvider O2::GPUTracking) # must not depend on GPU backend to avoid cyclic dependencies
if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
  add_dependencies(GPUTrackingHIPExternalProvider ${MODULE}_HIPIFIED)
endif()
