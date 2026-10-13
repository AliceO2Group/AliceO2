# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

set(MODULE GPUTrackingCPUDevice)

set(SRCS GPUReconstructionCPUDevice.cxx GPUReconstructionCPUDeviceGenRTC.cxx)
set(HDRS GPUReconstructionCPUDevice.h GPUReconstructionCPUDeviceInternals.h GPUReconstructionCPUDeviceIncludesSystem.h)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(GPUCA_CPUDEVICE_RTC ON)
else()
  set(GPUCA_CPUDEVICE_RTC OFF)
endif()
message(STATUS "Building GPUTracking with CPUDevice support (RTC: ${GPUCA_CPUDEVICE_RTC})")

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  set(defineIncludeSrc "O2::${MODULE}")
else()
  set(defineIncludeSrc "${MODULE}")
endif()

if(GPUCA_CPUDEVICE_RTC) # RTC compiles the kernels with the host compiler and the host compiler flags, and loads them via dlopen
  enable_language(ASM)
  if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
    get_filename_component(GPUDIR ${CMAKE_SOURCE_DIR}/../ ABSOLUTE)
  else()
    set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
  endif()
  string(TOUPPER "${CMAKE_BUILD_TYPE}" TMP_BUILD_TYPE_UPPER)
  set(GPU_RTC_DEFINES "-D$<JOIN:$<TARGET_PROPERTY:${defineIncludeSrc},COMPILE_DEFINITIONS>,$<SEMICOLON>-D>")
  set(GPU_RTC_INCLUDES "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:${defineIncludeSrc},INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>"
                       -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
                       -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
  )
  separate_arguments(GPU_RTC_FLAGS NATIVE_COMMAND "-std=c++${CMAKE_CXX_STANDARD} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${TMP_BUILD_TYPE_UPPER}} -w")
  set(GPU_RTC_LINK_FLAGS -fPIC -shared -fvisibility=hidden -fvisibility-inlines-hidden -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,-z,defs) # Self-contained library, only the kernels are exported (version script at runtime)
  set(GPU_RTC_SRC ${GPUDIR}/Base/cpudevice/GPUReconstructionCPUDevicertc.cxx)
  set(GPU_RTC_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionCPUDevicertc)

  add_custom_command(
      OUTPUT ${GPU_RTC_BIN}.src
      COMMAND cp ${GPUDIR}/Base/GPUStdSystemHeaders.h ${GPU_RTC_BIN}.src
      COMMAND cat ${GPUDIR}/Base/cpudevice/GPUReconstructionCPUDeviceIncludesSystem.h | grep -v GPUStdSystemHeaders.h >> ${GPU_RTC_BIN}.src
      COMMAND ${CMAKE_CXX_COMPILER} ${GPU_RTC_DEFINES} ${GPU_RTC_INCLUDES} ${GPU_RTC_FLAGS} -x c++ -nostdinc -P -E -MD -MT ${GPU_RTC_BIN}.src -MF ${GPU_RTC_BIN}.src.d ${GPU_RTC_SRC} >> ${GPU_RTC_BIN}.src
      DEPENDS ${GPU_RTC_SRC} ${GPUDIR}/Base/GPUStdSystemHeaders.h ${GPUDIR}/Base/cpudevice/GPUReconstructionCPUDeviceIncludesSystem.h
      DEPFILE ${GPU_RTC_BIN}.src.d
      COMMAND_EXPAND_LISTS
      COMMENT "Preparing CPUDevice RTC source file ${GPU_RTC_BIN}.src"
  )
  add_custom_command(
      OUTPUT ${GPU_RTC_BIN}.command
      COMMAND echo -n "${CMAKE_CXX_COMPILER} ${GPU_RTC_FLAGS} ${GPU_RTC_LINK_FLAGS}" > ${GPU_RTC_BIN}.command
      COMMAND_EXPAND_LISTS VERBATIM
      COMMENT "Preparing CPUDevice RTC command file ${GPU_RTC_BIN}.command"
  )
  add_custom_command(
      OUTPUT ${GPU_RTC_BIN}.command.no_fast_math
      COMMAND echo -n "${GPUCA_CXX_NO_FAST_MATH_FLAGS}" > ${GPU_RTC_BIN}.command.no_fast_math
      COMMAND_EXPAND_LISTS VERBATIM
      COMMENT "Preparing CPUDevice RTC NO_FAST_MATH command file ${GPU_RTC_BIN}.command.no_fast_math"
  )
  foreach(TMP_RESOURCE src command command.no_fast_math)
    create_binary_resource(${GPU_RTC_BIN}.${TMP_RESOURCE} ${GPU_RTC_BIN}.${TMP_RESOURCE}.o)
    list(APPEND SRCS ${GPU_RTC_BIN}.${TMP_RESOURCE}.o)
  endforeach()
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(
    ${MODULE}
    SOURCES ${SRCS}
    PUBLIC_LINK_LIBRARIES O2::GPUTracking
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
  install(TARGETS ${MODULE})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
endif()

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU)

target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
target_link_libraries(${targetName} PRIVATE TBB::tbb)
if(GPUCA_CPUDEVICE_RTC)
  target_compile_definitions(${targetName} PRIVATE GPUCA_CPUDEVICE_RTC)
  target_link_libraries(${targetName} PRIVATE ${CMAKE_DL_LIBS})
endif()
