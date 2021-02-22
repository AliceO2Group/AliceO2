# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE GPUTrackingOCL2)
enable_language(ASM)

if(DEFINED OCL2_GPUTARGET)
  set(TMP_TARGET "(AMDGPU Target ${OCL2_GPUTARGET})")
endif()
message(STATUS "Building GPUTracking with OpenCL 2 support ${TMP_TARGET}")

# convenience variables
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(GPUDIR ${CMAKE_SOURCE_DIR}/../)
else()
  set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
endif()
set(CL_SRC ${GPUDIR}/Base/opencl-common/GPUReconstructionOCL.cl)
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCL2Code)

set(OCL_FLAGS -Xclang -fdenormal-fp-math-f32=ieee -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -Dcl_clang_storage_class_specifiers -Wno-invalid-constexpr -Wno-unused-command-line-argument -cl-std=clc++)
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

if(OPENCL2_ENABLED_AMD) # BUILD OpenCL2 binaries for AMD target
  if(NOT DEFINED OCL2_GPUTARGET)
    execute_process(COMMAND ${ROCM_AGENT_ENUMERATOR} COMMAND "sort" COMMAND "tail -n 1" OUTPUT_VARIABLE OCL2_GPUTARGET)
  endif()

  # executes AMD clang-ocl
  add_custom_command(
    OUTPUT ${CL_BIN}.amd
    COMMAND ${CLANG_OCL}
            -O3
            ${OCL_FLAGS}
            ${OCL_DEFINECL}
            -mcpu=${OCL2_GPUTARGET}
            -o ${CL_BIN}.amd ${CL_SRC}
    MAIN_DEPENDENCY ${CL_SRC}
    IMPLICIT_DEPENDS CXX ${CL_SRC}
    COMMAND_EXPAND_LISTS)

  # cmake-format: off
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.amd.S
      COMMAND cat ${GPUDIR}/utils/include.S | sed "s/FILENAMEMOD/_makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl_amd/g" | sed "s/FILENAMENORMAL/GPUReconstructionOCL2Code.amd/g" > ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.amd.S
      MAIN_DEPENDENCY ${GPUDIR}/utils/include.S
  )
  # cmake-format: on

  # make cmake compile the assembler file, add proper dependency on included
  # binary code
  set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.amd.S
    PROPERTIES
    LANGUAGE
    ASM
    OBJECT_DEPENDS
    "${CL_BIN}.amd;${GPUDIR}/utils/include.S")

  set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.amd.S)
endif()

if(OPENCL2_ENABLED_SPIRV) # BUILD OpenCL2 intermediate code for SPIR-V target
  # executes clang to create llvm IL code
  add_custom_command(
      OUTPUT ${CL_BIN}.bc
      COMMAND clang
              -O0
              -emit-llvm --target=spir64-unknown-unknown
              ${OCL_FLAGS}
              ${OCL_DEFINECL}
              -o ${CL_BIN}.bc -c ${CL_SRC}
      MAIN_DEPENDENCY ${CL_SRC}
      IMPLICIT_DEPENDS CXX ${CL_SRC}
      COMMAND_EXPAND_LISTS)

  # executes converter to create spirv IL code
  add_custom_command(
      OUTPUT ${CL_BIN}.spirv
      COMMAND ${LLVM_SPIRV} ${CL_BIN}.bc -o=${CL_BIN}.spirv
      MAIN_DEPENDENCY ${CL_BIN}.bc)

  # cmake-format: off
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.S
      COMMAND cat ${GPUDIR}/utils/include.S | sed "s/FILENAMEMOD/_makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl_spirv/g" | sed "s/FILENAMENORMAL/GPUReconstructionOCL2Code.spirv/g" > ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.S
      MAIN_DEPENDENCY ${GPUDIR}/utils/include.S
  )
  # cmake-format: on

  # make cmake compile the assembler file, add proper dependency on included
  # binary code
  set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.S
    PROPERTIES
    LANGUAGE
    ASM
    OBJECT_DEPENDS
    "${CL_BIN}.spirv;${GPUDIR}/utils/include.S")

  set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.spirv.S)
endif()

if(OPENCL2_ENABLED) # BUILD OpenCL2 source code for runtime compilation target
  # executes clang to preprocess
  add_custom_command(
      OUTPUT ${CL_BIN}.src
      COMMAND clang
              ${OCL_DEFINECL}
              -E ${CL_SRC} > ${CL_BIN}.src
      MAIN_DEPENDENCY ${CL_SRC}
      IMPLICIT_DEPENDS CXX ${CL_SRC}
      COMMAND_EXPAND_LISTS)

  # cmake-format: off
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.S
      COMMAND cat ${GPUDIR}/utils/include.S | sed "s/FILENAMEMOD/_makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl_src/g" | sed "s/FILENAMENORMAL/GPUReconstructionOCL2Code.src/g" > ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.S
      MAIN_DEPENDENCY ${GPUDIR}/utils/include.S
  )
  # cmake-format: on

  # make cmake compile the assembler file, add proper dependency on included
  # binary code
  set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.S
    PROPERTIES
    LANGUAGE
    ASM
    OBJECT_DEPENDS
    "${CL_BIN}.src;${GPUDIR}/utils/include.S")

  set(SRCS ${SRCS} ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.src.S)
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

if(OPENCL2_ENABLED_AMD)
  target_compile_definitions(${targetName} PRIVATE OPENCL2_ENABLED_AMD)
endif()
if(OPENCL2_ENABLED_SPIRV)
  target_compile_definitions(${targetName} PRIVATE OPENCL2_ENABLED_SPIRV)
endif()
target_compile_definitions(${targetName} PRIVATE OCL_FLAGS=$<JOIN:${OCL_FLAGS},\ >)
