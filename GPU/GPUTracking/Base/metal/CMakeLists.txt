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

set(MODULE GPUTrackingMETAL)
enable_language(ASM)

message(STATUS "Building GPUTracking with Metal support")

# convenience variables
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(GPUDIR ${CMAKE_SOURCE_DIR}/../)
else()
  set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
endif()
set(METAL_SRC ${GPUDIR}/Base/metal/GPUReconstructionMetal.metal)
set(METAL_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionMetalCode)

# MSL 4.1 or later, not negotiable: the address-space macros in
# GPUCommonDefAPI.h rely on an unannotated `this` being generic, which 4.0 and
# earlier resolve as `thread` instead -- so an older toolchain would not fail
# cleanly, it would build the wrong thing. Stating -std here turns that into an
# immediate "invalid value 'metal4.1'" from the compiler.
#
# 4.1 targets macOS 27 (MSL specification, section 1.6.10); Xcode 26.6 stops at
# metal4.0, so this needs a newer toolchain than the CI builders currently have.
set(METAL_FLAGS -std=metal4.1 ${GPUCA_METAL_DENORMALS_FLAGS})
set(METAL_DEFINES "-D$<JOIN:$<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>,$<SEMICOLON>-D>"
            "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:O2::GPUTracking,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>"
            -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
            -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
            -I${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
            -DGPUCA_GPUCODE=1
)

set(SRCS GPUReconstructionMetal.mm GPUReconstructionMetalKernels.mm)
set(HDRS GPUReconstructionMetal.h GPUReconstructionMetalIncludesHost.h)

message("FOO ${SRCS} ${METAL_ENABLED} ${ALIGPU_BUILD_TYPE}")


if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ${SRCS}
                 PUBLIC_LINK_LIBRARIES O2::GPUTracking
                 TARGETVARNAME targetName)

  target_link_libraries(${targetName} PUBLIC ${METAL_FRAMEWORKS})

  target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
  # the compile_defitions are not propagated automatically on purpose (they are
  # declared PRIVATE) so we are not leaking them outside of the GPU**
  # directories
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} GPUTracking)
  install(TARGETS ${MODULE})
  set(targetName ${MODULE})
endif()

if(METAL_ENABLED) # BUILD Metal source code for runtime compilation target

  # Apple toolchain only, deliberately: the source goes .metal -> AIR with
  # `xcrun metal` and nothing else. An earlier draft routed C++ through
  # clang --target=vulkan1.4 to SPIR-V and then into Metal; that is not used and
  # is not wanted -- it drags in the LLVM-SPIRV translator, a second frontend
  # with its own dialect quirks, and a translation step Apple neither ships nor
  # supports. Kept out rather than commented out so nobody revives it by
  # accident.

  message("Metal was enabled ${METAL_SRC}")
  # executes clang to preprocess
  add_custom_command(
      OUTPUT ${METAL_BIN}.metal
      COMMAND xcrun -sdk macosx metal
              -Wno-unused-command-line-argument
              ${METAL_FLAGS}
              ${METAL_DEFINES}
              -MD -MT ${METAL_BIN}.src -MF ${METAL_BIN}.src.d
              -E -P ${METAL_SRC} > ${METAL_BIN}.metal
      DEPENDS ${METAL_SRC}
      DEPFILE ${METAL_BIN}.src.d
      COMMAND_EXPAND_LISTS
      COMMENT "Preparing Metal source file for run time compilation ${METAL_BIN}.metal")

  # Create the ir
  add_custom_command(
      OUTPUT ${METAL_BIN}.ir
      COMMAND xcrun -sdk macosx metal
              -Wno-unused-command-line-argument
              -Wno-c++17-extensions
              -ferror-limit=10000
              ${METAL_FLAGS}
              ${METAL_DEFINES}
              ${METAL_BIN}.metal
              -o ${METAL_BIN}.ir
      DEPENDS ${METAL_BIN}.metal
      COMMAND_EXPAND_LISTS
      COMMENT "Preparing Metal intermediate representation for run time compilation ${METAL_BIN}.ir")

  add_custom_target(metal_preprocessed_code ALL DEPENDS ${METAL_BIN}.metal COMMENT "Needed to inject dependency on its creation")
  add_custom_target(metal_intermediate_representation ALL DEPENDS ${METAL_BIN}.ir COMMENT "Needed to inject dependency on its creation")

  # Pack the file into __DATA,__gpu_resource during final link. This
  # way we do not need to create an intermediate object.
  target_link_options(${targetName}
                      PRIVATE
                      "-Wl,-sectcreate,__DATA,__gpu_resource,${METAL_BIN}.metal")
  add_dependencies(${targetName} metal_preprocessed_code)
  add_dependencies(${targetName} metal_intermediate_representation)
endif()

install(FILES ${HDRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU)

target_compile_definitions(${targetName} PRIVATE GPUCA_METAL_BUILD_FLAGS=$<JOIN:${METAL_FLAGS},\ > )
