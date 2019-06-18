message(STATUS "Building GPUTracking with CUDA support")

o2_add_library(
  GPUTrackingCUDA
  SOURCES GPUReconstructionCUDA.cu
  PRIVATE_INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/Detectors/Base/src
    ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
    ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include
  PUBLIC_LINK_LIBRARIES O2::GPUTracking
  TARGETVARNAME targetName)

target_compile_definitions(
  ${targetName} PUBLIC GPUCA_GPULIBRARY=CUDA
  $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

target_compile_options(${targetName} PUBLIC --expt-relaxed-constexpr)

set_target_properties(${targetName} PROPERTIES LINKER_LANGUAGE CXX)
