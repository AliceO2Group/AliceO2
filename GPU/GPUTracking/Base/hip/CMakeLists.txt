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
  set(GPUCA_HIP_FILE_LIST GPUReconstructionCUDA.cu GPUReconstructionCUDAExternalProvider.cu GPUReconstructionCUDA.h GPUReconstructionCUDAInternals.h GPUReconstructionCUDAkernel.template.cu CUDAThrustHelpers.h GPUReconstructionCUDADef.h GPUReconstructionCUDAGenRTC.cxx GPUReconstructionCUDAKernels.cu GPUReconstructionCUDArtc.cu)
  set(GPUCA_HIP_LOCAL_FILE_LIST GPUReconstructionHIPIncludes.h)
  set(HIP_SOURCES "")
  foreach(file ${GPUCA_HIP_FILE_LIST})
    get_filename_component(ABS_CUDA_SORUCE ../cuda/${file} ABSOLUTE)
    get_filename_component(CUDA_SOURCE ${file} NAME)
    string(REPLACE ".cu" ".hip" HIP_SOURCE1 ${CUDA_SOURCE})
    string(REPLACE "CUDA" "HIP" HIP_SOURCE ${HIP_SOURCE1})
    add_custom_command(
      OUTPUT ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
      COMMAND ${hip_HIPIFY_PERL_EXECUTABLE} --quiet-warnings ${ABS_CUDA_SORUCE} | sed -e 's/CUDA/HIP/g' -e 's/cuda/hip/g' > ${GPUCA_HIP_SOURCE_DIR}/${HIP_SOURCE}
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

  add_custom_target(${MODULE}_HIPIFIED_CHK COMMAND diff -u ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip ${CMAKE_CURRENT_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip DEPENDS ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip ${CMAKE_CURRENT_SOURCE_DIR}/GPUReconstructionHIPkernel.template.hip)
else()
  get_filename_component(GPUCA_HIP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
endif()

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
      COMMAND cat ${GPUDIR}/Base/hip/GPUReconstructionHIPIncludes.h > ${GPU_RTC_BIN}.src
      COMMAND ${CMAKE_CXX_COMPILER} ${GPU_RTC_DEFINES} ${GPU_RTC_INCLUDES} -std=c++${CMAKE_HIP_STANDARD} -D__HIPCC__ -D__HIP_DEVICE_COMPILE__ -x c++ -E ${GPU_RTC_SRC} >> ${GPU_RTC_BIN}.src
      MAIN_DEPENDENCY ${GPU_RTC_SRC}
      IMPLICIT_DEPENDS CXX ${GPU_RTC_SRC}
      COMMAND_EXPAND_LISTS
      COMMENT "Preparing HIP RTC source file ${GPU_RTC_BIN}.src"
  )
  create_binary_resource(${GPU_RTC_BIN}.src ${GPU_RTC_BIN}.src.o)

  add_custom_command(
      OUTPUT ${GPU_RTC_BIN}.command
      COMMAND echo -n "${hip_HIPCC_EXECUTABLE} ${GPU_RTC_FLAGS_SEPARATED} ${GPU_RTC_DEFINES} --genco" > ${GPU_RTC_BIN}.command
      COMMAND_EXPAND_LISTS
      VERBATIM
      COMMENT "Preparing HIP RTC command file ${GPU_RTC_BIN}.command"
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
    PUBLIC_LINK_LIBRARIES ${TMP_BASELIB} O2::ITStrackingHIP
    PRIVATE_INCLUDE_DIRECTORIES
      ${CMAKE_SOURCE_DIR}/Detectors/Base/src
      ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
      ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
      ${GPUCA_HIP_SOURCE_DIR}
    TARGETVARNAME targetName)

  target_compile_definitions(${targetName} PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

  install(FILES ${HDRS} DESTINATION include/GPU)

#  o2_add_test(GPUsortHIP NAME test_GPUsortHIP
#                  SOURCES test/testGPUsortHIP.hip.cxx
#                  PUBLIC_LINK_LIBRARIES O2::GPUCommon hip::host hip::device hip::hipcub roc::rocthrust
#                  COMPONENT_NAME GPU
#                  LABELS gpu)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  set(targetName "Ali${MODULE}")
  set(TMP_BASELIB AliGPUTracking)
  if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 14)
    message(FATAL_ERROR "HIP support needs at least C++14")
  endif()

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("${targetName}" "" "${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIP.h" "${incdirs} .")

  # Generate the ROOT map Dependecies
  generate_rootmap("${targetName}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(${targetName} SHARED ${SRCS})
  set(SRCS_CXX ${SRCS_CXX} G__${targetName}.cxx) # CMake HIP language feature will parse hip flags to everything that is linked to hip libraries, so add the ROOT dictionary file here
  target_link_libraries(${targetName} PUBLIC ${TMP_BASELIB})

  # ROOT Cannot load this library, so we have to remove it from the list
  list(REMOVE_ITEM ALILIBSTESTED ${targetName})
  set(ALILIBSTESTED ${ALILIBSTESTED} CACHE INTERNAL "ALILIBSTESTED")

  # Installation
  install(TARGETS ${targetName} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS} DESTINATION include)
  include_directories(${GPUCA_HIP_SOURCE_DIR})
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(targetName "${MODULE}")
  set(TMP_BASELIB GPUTracking)
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} PUBLIC ${TMP_BASELIB})
  install(TARGETS GPUTrackingHIP)
  include_directories(${GPUCA_HIP_SOURCE_DIR})
endif()

add_library(${MODULE}_CXX OBJECT ${SRCS_CXX}) # Adding a C++ library for the .cxx code of the HIP library, such that it does not link to HIP libraries, and CMake HIP Language doesn't add HIP compile flags.
target_compile_definitions(${MODULE}_CXX PRIVATE $<TARGET_PROPERTY:${TMP_BASELIB},COMPILE_DEFINITIONS>)
target_include_directories(${MODULE}_CXX PRIVATE $<TARGET_PROPERTY:${TMP_BASELIB},INCLUDE_DIRECTORIES>)
target_link_libraries(${targetName} PRIVATE ${MODULE}_CXX)

if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
  add_custom_target(${MODULE}_HIPIFIED DEPENDS ${HIP_SOURCES} ${MODULE}_HIPIFIED_CHK)
  add_dependencies(${targetName} ${MODULE}_HIPIFIED)
  add_dependencies(${MODULE}_CXX ${MODULE}_HIPIFIED)
endif()

# Setting target architecture and adding GPU libraries
target_link_libraries(${targetName} PRIVATE hip::host hip::device hip::hipcub roc::rocthrust)
set_target_hip_arch(${targetName})

if(OpenMP_CXX_FOUND)
  # Must be private, depending libraries might be compiled by compiler not understanding -fopenmp
  target_compile_definitions(${MODULE}_CXX PRIVATE WITH_OPENMP)
  target_link_libraries(${MODULE}_CXX PRIVATE OpenMP::OpenMP_CXX)
endif()

# Special handling of GPU kernels in case of per-kernel compilation / RDC
if(NOT DEFINED GPUCA_HIP_COMPILE_MODE)
  if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    set(GPUCA_HIP_COMPILE_MODE "onefile")
  else()
    set(GPUCA_HIP_COMPILE_MODE "perkernel")
  endif()
endif()
if(GPUCA_HIP_COMPILE_MODE STREQUAL "onefile")
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=0)
elseif(GPUCA_HIP_COMPILE_MODE STREQUAL "perkernel")
  #add_library(GPUTrackingHIPKernels OBJECT $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_>,APPEND,.cu>, >)
  #set_property(TARGET GPUTrackingHIPKernels PROPERTY HIP_FATBIN_COMPILATION ON)
  add_subdirectory(per_kernel)
  target_compile_options(GPUTrackingHIPKernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>)
  target_link_options(GPUTrackingHIPKernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fno-gpu-rdc>)
  target_compile_definitions(${targetName} PRIVATE GPUCA_KERNEL_COMPILE_MODE=1)
  target_compile_definitions(GPUTrackingHIPKernels PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
  target_include_directories(GPUTrackingHIPKernels PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
  if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
    add_dependencies(GPUTrackingHIPKernels ${MODULE}_HIPIFIED)
  endif()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_module_fatbin)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o
    COMMAND cp -u $<TARGET_OBJECTS:GPUTrackingHIPKernels> ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_module_fatbin/
    COMMAND ${CMAKE_LINKER} --relocatable --format binary --output ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o $<PATH:RELATIVE_PATH,$<LIST:TRANSFORM,$<PATH:GET_FILENAME,$<TARGET_OBJECTS:GPUTrackingHIPKernels>>,PREPEND,${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_module_fatbin/>,${CMAKE_CURRENT_BINARY_DIR}>
    DEPENDS GPUTrackingHIPKernels $<TARGET_OBJECTS:GPUTrackingHIPKernels>
    COMMENT "Compiling fatbin kernels ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
  target_sources(${targetName} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/GPUTrackingHIPKernelModules.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)

  # Disable all non-deterministic floating point to make TPC track model encoding / decoding precise
  set_source_files_properties(${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCCompressionKernels_step0attached.hip
                              ${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCCompressionKernels_step1unattached.hip
                              ${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCDecompressionKernels_step0attached.hip
                              ${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_GPUTPCDecompressionKernels_step1unattached.hip
                              TARGET_DIRECTORY ${targetName}
                              PROPERTIES
                              COMPILE_FLAGS "${GPUCA_HIP_NO_FAST_MATH_FLAGS}"
                              COMPILE_DEFINITIONS "GPUCA_NO_FAST_MATH")
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

if(ALIGPU_BUILD_TYPE STREQUAL "O2" OR CONFIG_O2_EXTENSIONS)
  add_library(GPUTrackingHIPExternalProvider OBJECT ${GPUCA_HIP_SOURCE_DIR}/GPUReconstructionHIPExternalProvider.hip)
  add_library(O2::GPUTrackingHIPExternalProvider ALIAS GPUTrackingHIPExternalProvider)
  target_compile_options(GPUTrackingHIPExternalProvider PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
  target_link_options(GPUTrackingHIPExternalProvider PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
  target_compile_definitions(GPUTrackingHIPExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},COMPILE_DEFINITIONS>)
  target_include_directories(GPUTrackingHIPExternalProvider PRIVATE $<TARGET_PROPERTY:${targetName},INCLUDE_DIRECTORIES>)
  if(NOT DEFINED GPUCA_HIP_HIPIFY_FROM_CUDA OR "${GPUCA_HIP_HIPIFY_FROM_CUDA}")
    add_dependencies(GPUTrackingHIPExternalProvider ${MODULE}_HIPIFIED)
  endif()
endif()
