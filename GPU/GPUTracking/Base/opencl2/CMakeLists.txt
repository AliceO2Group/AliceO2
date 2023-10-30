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

set(MODULE GPUTrackingOCL2)
enable_language(ASM)

message(STATUS "Building GPUTracking with OpenCL 2 support")

# convenience variables
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(GPUDIR ${CMAKE_SOURCE_DIR}/../)
else()
  set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
endif()
set(CL_SRC ${GPUDIR}/Base/opencl-common/GPUReconstructionOCL.cl)
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCL2Code)

set(OCL_FLAGS -ferror-limit=1000 -Dcl_clang_storage_class_specifiers -Wno-invalid-constexpr -Wno-unused-command-line-argument -cl-std=CLC++2021)
if(NOT DEFINED GPUCA_NO_FAST_MATH OR NOT ${GPUCA_NO_FAST_MATH})
  set(OCL_FLAGS ${OCL_FLAGS} -Xclang -fdenormal-fp-math-f32=ieee -cl-mad-enable -cl-no-signed-zeros)
endif()
set(OCL_DEFINECL "-D$<JOIN:$<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>,$<SEMICOLON>-D>"
            "-I$<JOIN:$<FILTER:$<TARGET_PROPERTY:O2::GPUTracking,INCLUDE_DIRECTORIES>,EXCLUDE,^/usr/include/?>,$<SEMICOLON>-I>"
            -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
            -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
            -I${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
            -I${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include
            -DGPUCA_GPULIBRARY=OCL2
            -D__OPENCLCPP__
)

set(SRCS GPUReconstructionOCL2.cxx)
set(HDRS GPUReconstructionOCL2.h GPUReconstructionOCL2Internals.h)

if(OPENCL2_ENABLED_SPIRV) # BUILD OpenCL2 intermediate code for SPIR-V target
  # executes clang to create llvm IL code
  # Add -fintegrated-objemitter once we switch to clang >= 17
  cmake_path(GET LLVM_SPIRV PARENT_PATH TMP_LLVM_SPIRV_PATH)
  add_custom_command(
      OUTPUT ${CL_BIN}.spirv
      COMMAND ${CMAKE_COMMAND} -E env "PATH=${TMP_LLVM_SPIRV_PATH}:\$PATH" ${LLVM_CLANG}
              -O0
              --target=spirv64
              ${OCL_FLAGS}
              ${OCL_DEFINECL}
              -o ${CL_BIN}.spirv -c ${CL_SRC}
      MAIN_DEPENDENCY ${CL_SRC}
      IMPLICIT_DEPENDS CXX ${CL_SRC}
      COMMAND_EXPAND_LISTS)

  create_binary_resource(${CL_BIN}.spirv ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.o)
  set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.o)
endif()

if(OPENCL2_ENABLED) # BUILD OpenCL2 source code for runtime compilation target
  # executes clang to preprocess
  add_custom_command(
      OUTPUT ${CL_BIN}.src
      COMMAND ${LLVM_CLANG}
              ${OCL_DEFINECL} -cl-no-stdinc
              -E ${CL_SRC} > ${CL_BIN}.src
      MAIN_DEPENDENCY ${CL_SRC}
      IMPLICIT_DEPENDS CXX ${CL_SRC}
      COMMAND_EXPAND_LISTS)

  create_binary_resource(${CL_BIN}.src ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.o)
  set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.o)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ${SRCS}
                 PUBLIC_LINK_LIBRARIES O2::GPUTrackingOpenCLCommon
                 TARGETVARNAME targetName)

  target_compile_definitions(
    ${targetName} PRIVATE GPUCA_GPULIBRARY=OCL2
    $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
  # the compile_defitions are not propagated automatically on purpose (they are
  # declared PRIVATE) so we are not leaking them outside of the GPU**
  # directories

  install(FILES ${HDRS} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_GPULIBRARY=OCL2)

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("Ali${MODULE}" "" "GPUReconstructionOCL2.h" "${incdirs} .")

  # Generate the ROOT map
  generate_rootmap("Ali${MODULE}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE} SHARED ${SRCS} G__Ali${MODULE}.cxx)
  target_link_libraries(Ali${MODULE} PUBLIC AliGPUTrackingOpenCLCommon)

  # Installation
  install(TARGETS Ali${MODULE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS} DESTINATION include)
  set(targetName Ali${MODULE})
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_definitions(-DGPUCA_GPULIBRARY=OCL2)
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} GPUTrackingOpenCLCommon)
  install(TARGETS ${MODULE})
  set(targetName ${MODULE})
endif()

if(OPENCL2_ENABLED_SPIRV)
  target_compile_definitions(${targetName} PRIVATE OPENCL2_ENABLED_SPIRV)
endif()
target_compile_definitions(${targetName} PRIVATE OCL_FLAGS=$<JOIN:${OCL_FLAGS},\ >)
