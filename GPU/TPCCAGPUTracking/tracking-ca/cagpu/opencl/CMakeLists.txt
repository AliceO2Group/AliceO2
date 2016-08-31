# **************************************************************************
# * Copyright(c) 1998-2014, ALICE Experiment at CERN, All rights reserved. *
# *                                                                        *
# * Author: The ALICE Off-line Project.                                    *
# * Contributors are mentioned in the code where appropriate.              *
# *                                                                        *
# * Permission to use, copy, modify and distribute this software and its   *
# * documentation strictly for non-commercial purposes is hereby granted   *
# * without fee, provided that the above copyright notice appears in all   *
# * copies and that both the copyright notice and this permission notice   *
# * appear in the supporting documentation. The authors make no claims     *
# * about the suitability of this software for any purpose. It is          *
# * provided "as is" without express or implied warranty.                  *
# **************************************************************************

# Module
set(MODULE AliHLTTPCCAGPUOpenCL)

# AMD APP SDK required for OpenCL tracker; it's using specific
# extensions (currently) not provided by other vendors.
# either set cmake variable via "-D$AMDAPPSDKROOT=/path/to/amdappsdkroot"
# or via environment variable $AMDAPPSDKROOT
if(NOT AMDAPPSDKROOT)
    set(AMDAPPSDKROOT $ENV{AMDAPPSDKROOT})
endif()

if(NOT AMDAPPSDKROOT)
    message(FATAL_ERROR "AMDAPPSDKROOT not set. Please install AMD APP SDK and set $AMDAPPSDKROOT or disable HLT_OPENCL.")
endif()

if(NOT "$ENV{GPU_FORCE_64BIT_PTR}" STREQUAL "1")
    message(FATAL_ERROR "Please set env variable $GPU_FORCE_64BIT_PTR=1, required by HLT OpenCL tracker.")
endif()

message(STATUS "Building HLT with OpenCL support" )

#convenience variables
set(CAGPUDIR ${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca/cagpu)

#libdir path
link_directories(${AMDAPPSDKROOT}/lib/x86_64)

# build the OpenCL compile wrapper:
# -- checks the correct vendor implementation (AMD)
# -- builds binary code (blob) for the found platform(s)
add_executable(opencl_compiler ${CAGPUDIR}/makefiles/makefile_opencl_compiler.cpp)
target_link_libraries(opencl_compiler amdocl64 OpenCL)

#convenience variables
set(CL_SRC ${CAGPUDIR}/opencl/AliHLTTPCCAGPUTrackerOpenCL.cl )
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/AliHLTTPCCAGPUTrackerOpenCLCode.bin )
set(CL_INC
    -I${AliRoot_SOURCE_DIR}/HLT/BASE 
    -I${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca
    -I${CAGPUDIR}
)

# executes OpenCL compiler wrapper to build binary object
add_custom_command(
    OUTPUT ${CL_BIN}
    COMMAND opencl_compiler -output-file ${CL_BIN} ${CL_SRC} -- ${CL_INC} -x clc++
    MAIN_DEPENDENCY ${CL_SRC}
)

# make cmake compile the assembler file
set_source_files_properties(${CAGPUDIR}/makefiles/include.S PROPERTIES LANGUAGE CXX)
# add proper dependency on included binary code
set_source_files_properties(${CAGPUDIR}/makefiles/include.S PROPERTIES OBJECT_DEPENDS ${CL_BIN})

# bit of a hack, but otherwise dictionary compilation fails:
# defines __ROOT__; triggers "ifdef" in source code to circumvent broken cint behaviour with "volatile" keyword
# not sure why __ROOT__ is used here, could just as well be something else
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/G__AliHLTTPCCAGPUOpenCL.cxx PROPERTIES COMPILE_DEFINITIONS "__ROOT__")

# Module include folder
include_directories(${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca/cagpu/opencl
                    ${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca/makefiles
                    ${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca/cagpu
                    ${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca
                   )

# Additional include folders in alphabetical order except ROOT
include_directories(${ROOT_INCLUDE_DIR}
                    ${AliRoot_SOURCE_DIR}/HLT/BASE
                    ${AMDAPPSDKROOT}/include
                   )

# Sources in alphabetical order
set(SRCS
    ${CAGPUDIR}/AliHLTTPCCAGPUTrackerBase.cxx
    AliHLTTPCCAGPUTrackerOpenCL.cxx
    ${CAGPUDIR}/makefiles/include.S
   )

# Headers from sources
set(CINTHDRS
    ${CAGPUDIR}/AliHLTTPCCAGPUTrackerBase.h
    AliHLTTPCCAGPUTrackerOpenCL.h
)

set(HDRS
    ${CINTHDRS}
    ${CAGPUDIR}/AliHLTTPCCAGPUTrackerCommon.h
    AliHLTTPCCAGPUTrackerOpenCLInternals.h
  )

# Generate the dictionary
# It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
get_directory_property(incdirs INCLUDE_DIRECTORIES)
generate_dictionary("${MODULE}" "" "${CINTHDRS}" "${incdirs}")

# Generate the ROOT map
generate_rootmap("${MODULE}" "" "")

# Add a library to the project using the specified source files
add_library_tested(${MODULE} SHARED ${SRCS} G__${MODULE}.cxx)
#MAD OpenCL run-time and driver
target_link_libraries(${MODULE} amdocl64 OpenCL AliHLTTPC)

# Installation
install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES ${HDRS} DESTINATION include)
