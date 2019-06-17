# AMD APP SDK required for OpenCL tracker as it's using specific extensions
# (currently) not provided by other vendors
if(NOT AMDAPPSDKROOT)
  set(AMDAPPSDKROOT $ENV{AMDAPPSDKROOT})
endif()

if(NOT AMDAPPSDKROOT)
  message(
    FATAL_ERROR
      "AMDAPPSDKROOT not set. Please install AMD APP SDK and set $AMDAPPSDKROOT or disable ENABLE_OPENCL."
    )
endif()

find_package(OpenCL REQUIRED)

message(STATUS "Building GPUTracking with OpenCL support")

# convenience variables
set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)
set(CL_SRC ${GPUDIR}/Base/opencl/GPUReconstructionOCL.cl)
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.bin)

# build the OpenCL compile wrapper :
#
# * checks the correct vendor implementation (AMD)
# * builds binary code (blob) for the found platform(s)
add_executable(opencl_compiler
               ${GPUDIR}/Standalone/makefiles/makefile_opencl_compiler.cpp)
target_link_libraries(opencl_compiler PUBLIC OpenCL::OpenCL)

# executes OpenCL compiler wrapper to build binary object
add_custom_command(
  OUTPUT ${CL_BIN}
  COMMAND LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$<TARGET_FILE_DIR:OpenCL::OpenCL>
          $<TARGET_FILE:opencl_compiler>
          -output-file
          ${CL_BIN}
          ${CL_SRC}
          --
          -I${GPUDIR}/HLTHeaders
          -I${GPUDIR}/SliceTracker
          -I${GPUDIR}/Merger
          -I${GPUDIR}/Base
          -I${GPUDIR}/TRDTracking
          -I${GPUDIR}/Merger
          -I${GPUDIR}/Standalone
          -I${CMAKE_SOURCE_DIR}/GPU/Common
          "-D$<JOIN:$<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>, -D>"
          -x
          clc++
  MAIN_DEPENDENCY ${CL_SRC})

# cmake-format: off
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
    COMMAND cat ${GPUDIR}/Standalone/makefiles/include.S | sed "s/FILENAMEMOD/_makefile_opencl_program_Base_opencl_GPUReconstructionOCL_cl/g" | sed "s/FILENAMENORMAL/GPUReconstructionOCLCode.bin/g" > ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
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

o2_add_library(GPUTrackingOpenCL
               SOURCES GPUReconstructionOCL.cxx
                       ${CMAKE_CURRENT_BINARY_DIR}/GPUReconstructionOCLCode.S
               PUBLIC_LINK_LIBRARIES OpenCL::OpenCL O2::GPUTracking
               TARGETVARNAME targetName)

target_compile_definitions(
  ${targetName} PRIVATE GPUCA_GPULIBRARY=OCL
  $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
# the compile_defitions are not propagated automatically on purpose (they are
# declared PRIVATE) so we are not leaking them outside of the GPU** directories
