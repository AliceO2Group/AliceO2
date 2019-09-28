# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE GPUTrackingOCL)

# AMD APP SDK required for OpenCL tracker as it's using specific extensions
# (currently) not provided by other vendors

if(NOT AMDAPPSDKROOT)
  message(
    FATAL_ERROR
      "AMDAPPSDKROOT not set. Please install AMD APP SDK and set $AMDAPPSDKROOT or disable ENABLE_OPENCL1."
    )
endif()

message(STATUS "Building GPUTracking with OpenCL 1.2 support")

# convenience variables
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(GPUDIR ${CMAKE_SOURCE_DIR}/../)
else()
  set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
endif()
set(CL_SRC ${GPUDIR}/Base/opencl-common/GPUReconstructionOCL.cl)
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCL1Code.bin)

# build the OpenCL compile wrapper :
#
# * checks the correct vendor implementation (AMD)
# * builds binary code (blob) for the found platform(s)
add_executable(opencl_compiler
               ${GPUDIR}/Standalone/makefiles/makefile_opencl_compiler.cxx)
target_link_libraries(opencl_compiler PUBLIC OpenCL::OpenCL)
set_property(TARGET opencl_compiler
             PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# executes OpenCL compiler wrapper to build binary object
add_custom_command(
  OUTPUT ${CL_BIN}
  COMMAND LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$<TARGET_FILE_DIR:OpenCL::OpenCL>
          $<TARGET_FILE:opencl_compiler>
          -output-file
          ${CL_BIN}
          ${CL_SRC}
          --
          "-D$<JOIN:$<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>,$<SEMICOLON>-D>"
          -I${GPUDIR}/../Common
          -I${GPUDIR}/Base
          -I${GPUDIR}/Merger
          -I${GPUDIR}/SliceTracker
          -DGPUCA_GPULIBRARY=OCL1
          -x clc++
  MAIN_DEPENDENCY ${CL_SRC}
  IMPLICIT_DEPENDS cxx ${CL_SRC}
  COMMAND_EXPAND_LISTS)

# cmake-format: off
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
    COMMAND cat ${GPUDIR}/Standalone/makefiles/include.S | sed "s/FILENAMEMOD/_makefile_opencl_program_Base_opencl_common_GPUReconstructionOCL_cl/g" | sed "s/FILENAMENORMAL/GPUReconstructionOCL1Code.bin/g" > ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
    MAIN_DEPENDENCY ${GPUDIR}/Standalone/makefiles/include.S
)
# cmake-format: on

# make cmake compile the assembler file, add proper dependency on included
# binary code
set_source_files_properties(
  ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
  PROPERTIES
  LANGUAGE
  CXX
  OBJECT_DEPENDS
  "${CL_BIN};${GPUDIR}/Standalone/makefiles/include.S")

set(SRCS GPUReconstructionOCL1.cxx
         ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S)
set(HDRS GPUReconstructionOCL1.h GPUReconstructionOCL1Internals.h)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ${SRCS}
                 PUBLIC_LINK_LIBRARIES O2::GPUTrackingOpenCLCommon
                 TARGETVARNAME targetName)

  target_compile_definitions(
    ${targetName} PRIVATE GPUCA_GPULIBRARY=OCL1
    $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
  # the compile_defitions are not propagated automatically on purpose (they are
  # declared PRIVATE) so we are not leaking them outside of the GPU**
  # directories

  install(FILES ${HDRS} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_GPULIBRARY=OCL1)

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("Ali${MODULE}" "" "GPUReconstructionOCL1.h" "${incdirs} .")

  # Generate the ROOT map
  generate_rootmap("Ali${MODULE}" "" "")

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE} SHARED ${SRCS} G__Ali${MODULE}.cxx)
  target_link_libraries(Ali${MODULE} PUBLIC AliGPUTrackingOpenCLCommon)

  # Installation
  install(TARGETS Ali${MODULE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS} DESTINATION include)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_definitions(-DGPUCA_GPULIBRARY=OCL1)
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} GPUTrackingOpenCLCommon)
  install(TARGETS ${MODULE})
endif()
