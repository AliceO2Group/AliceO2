if(NOT DEFINED HIP_PATH)
  if(NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been
installed")
  else()
    set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP
has been installed")
  endif()
endif()

if(NOT EXISTS ${HIP_PATH})
        message(FATAL_ERROR "HIP requested but HIP_PATH=${HIP_PATH} does not exist")
endif()

set(hip_ROOT ${HIP_PATH})
get_filename_component(hcc_ROOT ${hip_ROOT}/../hcc ABSOLUTE)

find_package(hip REQUIRED)

set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})

message(STATUS "Building GPUTracking with HIP support")

o2_add_library(
  GPUTrackingHIP
  SOURCES GPUReconstructionHIP.hip.cpp
  PUBLIC_LINK_LIBRARIES O2::GPUTracking hip::host hip::device
  PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
                             ${CMAKE_SOURCE_DIR}/Detectors/Base/src
  TARGETVARNAME targetName)

target_compile_options(${targetName}
                       PUBLIC --amdgpu-target=gfx906
                              -Wno-unused-command-line-argument
                              -Wno-invalid-constexpr)

target_compile_definitions(
        ${targetName} PUBLIC ENABLE_HIP GPUCA_GPULIBRARY=HIP ENABLE_HIP
  $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

target_link_options(${targetName} PUBLIC --amdgpu-target=gfx906)
