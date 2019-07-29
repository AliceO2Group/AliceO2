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

if(DEFINED OCL2_GPUTARGET)
  set(TMP_TARGET "(GPU Target ${OCL2_GPUTARGET})")
endif()
message(STATUS "Building GPUTracking with OpenCL 2 support ${TMP_TARGET}")

# convenience variables
if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(GPUDIR ${CMAKE_SOURCE_DIR}/../)
else()
  set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
endif()
set(CL_SRC ${GPUDIR}/Base/opencl-common/GPUReconstructionOCL.cl)
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCL2Code.bin)

if(NOT DEFINED OCL2_GPUTARGET)
  execute_process(COMMAND /opt/rocm/bin/rocm_agent_enumerator OUTPUT_VARIABLE OCL2_GPUTARGET)
endif()

# executes OpenCL compiler wrapper to build binary object
add_custom_command(
  OUTPUT ${CL_BIN}
  COMMAND ${CLANG_OCL}
          -o ${CL_BIN} ${CL_SRC}
          -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -ferror-limit=1000 -Xclang -finclude-default-header -Dcl_clang_storage_class_specifiers -Wno-invalid-constexpr -Wno-unused-command-line-argument
          "-D$<JOIN:$<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>, -D>"
          "-I$<JOIN:$<TARGET_PROPERTY:O2::GPUTracking,INCLUDE_DIRECTORIES>, -I>"
          -I${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src
          -I${CMAKE_SOURCE_DIR}/Detectors/Base/src
          -I${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include
          -DGPUCA_GPULIBRARY=OCL2
          -D__OPENCLCPP__
          -cl-std=c++
          -mcpu=${OCL2_GPUTARGET}
  MAIN_DEPENDENCY ${CL_SRC})

# cmake-format: off
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
    COMMAND cat ${GPUDIR}/Standalone/makefiles/include.S | sed "s/FILENAMEMOD/_makefile_opencl_program_Base_opencl_GPUReconstructionOCL2_cl/g" | sed "s/FILENAMENORMAL/GPUReconstructionOCL2Code.bin/g" > ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
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

set(SRCS GPUReconstructionOCL2.cxx
         ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S)
set(HDRS GPUReconstructionOCL2.h GPUReconstructionOCL2Internals.h)

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
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_definitions(-DGPUCA_GPULIBRARY=OCL2)
  add_library(${MODULE} SHARED ${SRCS})
  target_link_libraries(${MODULE} GPUTrackingOpenCLCommon)
  install(TARGETS ${MODULE})
endif()
