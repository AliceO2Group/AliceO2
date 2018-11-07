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
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    set(MODULE O2TPCCAGPUTrackingOpenCL)
elseif(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    set(MODULE AliTPCCAGPUTrackingOpenCL)
endif()
add_definitions(-DHLTCA_GPULIBRARY=OPENCL)

# AMD APP SDK required for OpenCL tracker; it's using specific
# extensions (currently) not provided by other vendors.
# either set cmake variable via "-D$AMDAPPSDKROOT=/path/to/amdappsdkroot"
# or via environment variable $AMDAPPSDKROOT
if(NOT AMDAPPSDKROOT)
    set(AMDAPPSDKROOT $ENV{AMDAPPSDKROOT})
endif()

if(NOT AMDAPPSDKROOT)
    message(FATAL_ERROR "AMDAPPSDKROOT not set. Please install AMD APP SDK and set $AMDAPPSDKROOT or disable ENABLE_OPENCL.")
endif()

if(NOT "$ENV{GPU_FORCE_64BIT_PTR}" STREQUAL "1")
    message(FATAL_ERROR "Please set env variable $GPU_FORCE_64BIT_PTR=1, required by HLT OpenCL tracker.")
endif()

message(STATUS "Building CAGPUTracking with OpenCL support" )

#convenience variables
set(CAGPUDIR ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking)

#libdir path
link_directories(${AMDAPPSDKROOT}/lib/x86_64)

# build the OpenCL compile wrapper:
# -- checks the correct vendor implementation (AMD)
# -- builds binary code (blob) for the found platform(s)
add_executable(opencl_compiler ${CAGPUDIR}/Standalone/makefiles/makefile_opencl_compiler.cpp)
target_link_libraries(opencl_compiler OpenCL)

#convenience variables
set(CL_SRC ${CAGPUDIR}/GlobalTracker/opencl/AliHLTTPCCAGPUTrackerOpenCL.cl )
set(CL_BIN ${CMAKE_CURRENT_BINARY_DIR}/AliHLTTPCCAGPUTrackerOpenCLCode.bin )
set(CL_INC
    -I${CAGPUDIR}/HLTHeaders
    -I${CAGPUDIR}/SliceTracker
    -I${CAGPUDIR}/Merger
    -I${CAGPUDIR}/GlobalTracker
    -I${CAGPUDIR}/TRDTracking
    -I${ALITPCCOMMON_DIR}/sources/Common
)

# executes OpenCL compiler wrapper to build binary object
add_custom_command(
    OUTPUT ${CL_BIN}
    COMMAND opencl_compiler -output-file ${CL_BIN} ${CL_SRC} -- ${CL_INC} -x clc++
    MAIN_DEPENDENCY ${CL_SRC}
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/AliHLTTPCCAGPUTrackerOpenCLCode.S
    COMMAND cat ${CAGPUDIR}/Standalone/makefiles/include.S | sed "s/FILENAMEMOD/_makefile_opencl_program_GlobalTracker_opencl_AliHLTTPCCAGPUTrackerOpenCL_cl/g" | sed "s/FILENAMENORMAL/AliHLTTPCCAGPUTrackerOpenCLCode.bin/g" > ${CMAKE_CURRENT_BINARY_DIR}/AliHLTTPCCAGPUTrackerOpenCLCode.S
    MAIN_DEPENDENCY ${CAGPUDIR}/Standalone/makefiles/include.S
)

# make cmake compile the assembler file, add proper dependency on included binary code
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/AliHLTTPCCAGPUTrackerOpenCLCode.S PROPERTIES LANGUAGE CXX OBJECT_DEPENDS "${CL_BIN};${CAGPUDIR}/Standalone/makefiles/include.S")

# Module include folder
include_directories(${CAGPUDIR}/GlobalTracker/opencl
                    ${CAGPUDIR}/GlobalTracker
                    ${CAGPUDIR}/Standalone/makefiles
                    ${CAGPUDIR}/SliceTracker
                    ${CAGPUDIR}/TRDTracking
                    ${CAGPUDIR}/Merger
                    ${ALITPCCOMMON_DIR}/sources/Common
                   )

# Additional include folders in alphabetical order except ROOT
include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
include_directories(${AMDAPPSDKROOT}/include)
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
   include_directories(${AliRoot_SOURCE_DIR}/HLT/BASE)
else() #if we do not build for AliRoot, for the time being we still need some dummy headers for some definitions
   include_directories(${CAGPUDIR}/HLTHeaders ${CAGPUDIR}/Standalone/include)
endif()

# Sources in alphabetical order
set(SRCS
    AliHLTTPCCAGPUTrackerOpenCL.cxx
    AliGPUReconstructionOCL.cxx
    ${CMAKE_CURRENT_BINARY_DIR}/AliHLTTPCCAGPUTrackerOpenCLCode.S
   )

# Headers from sources
set(CINTHDRS
    AliHLTTPCCAGPUTrackerOpenCL.h
)

set(HDRS
    ${CINTHDRS}
    ${CAGPUDIR}/GlobalTracker/AliHLTTPCCAGPUTrackerCommon.h
    AliHLTTPCCAGPUTrackerOpenCLInternals.h
  )

#Default cmake build script for AliRoot
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    # Generate the dictionary
    # It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
    get_directory_property(incdirs INCLUDE_DIRECTORIES)
    generate_dictionary("${MODULE}" "" "${CINTHDRS}" "${incdirs}")

    # Generate the ROOT map
    generate_rootmap("${MODULE}" "" "")

    # Add a library to the project using the specified source files
    add_library_tested(${MODULE} SHARED ${SRCS} G__${MODULE}.cxx)
    # AMD OpenCL run-time and driver
    target_link_libraries(${MODULE} OpenCL AliTPCCAGPUTracking)

    # Installation
    install(TARGETS ${MODULE}
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)

    install(FILES ${HDRS} DESTINATION include)
endif()

#Default cmake build script for O2
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
#We do not need the dictionary for the standalone library files
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME TPCCAGPUTrackingOCL_bucket)

    O2_GENERATE_LIBRARY()

    target_link_libraries(${MODULE} OpenCL O2TPCCAGPUTracking)
endif()
