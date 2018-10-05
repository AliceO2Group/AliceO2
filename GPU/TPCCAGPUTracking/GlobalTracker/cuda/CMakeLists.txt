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
set(MODULE O2TPCCAGPUTrackingCUDA)
elseif(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
set(MODULE AliTPCCAGPUTrackingCUDA)
endif()
add_definitions(-DHLTCA_GPULIBRARY=CUDA)

option(CUDA_GCCBIN "GCC binary to use for compiling host part of CUDA code for systems with multiple GCC versions installed"   OFF)

find_package(CUDA)
if(NOT CUDA_FOUND)
    message( FATAL_ERROR "NVIDIA CUDA package not found. Please install or set ENABLE_CUDA=OFF" )
else()
    find_library(LIBCUDA_SO_PATH libcuda.so)
    string(FIND ${LIBCUDA_SO_PATH} "-NOTFOUND" LIBCUDA_SO_PATH_NOTFOUND )
endif(NOT CUDA_FOUND)
message( STATUS "Building CAGPUTracking with CUDA support" )

if(LIBCUDA_SO_PATH_NOTFOUND GREATER -1)
  message( FATAL_ERROR "NVIDIA CUDA libcuda.so not found. Please install or set ENABLE_CUDA=OFF" )
endif(LIBCUDA_SO_PATH_NOTFOUND GREATER -1)

#set nvcc flags
set(CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets --use_fast_math --maxrregcount 64 -O4 -Xptxas -O4 -Xcompiler -fPIC -Xptxas -O4 -gencode arch=compute_61,code=sm_61 -Xcompiler "-std=c++14")
if(CUDA_GCCBIN)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-bindir ${CUDA_GCCBIN})
endif()

#for convenience
set(CAGPUDIR ${ALITPCCOMMON_DIR}/sources/TPCCAGPUTracking)

# Module include folder
include_directories(${CAGPUDIR}/GlobalTracker
                    ${CAGPUDIR}/GlobalTracker/cuda
                    ${CAGPUDIR}/Merger
                    ${CAGPUDIR}/SliceTracker
                    ${CAGPUDIR}/TRDTracking
                    ${ALITPCCOMMON_DIR}/sources/Common
                   )

# Additional include folders in alphabetical order except ROOT
include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    include_directories(${AliRoot_SOURCE_DIR}/HLT/BASE)
    add_definitions(-DHLTCA_BUILD_ALIROOT_LIB)
else() #if we do not build for AliRoot, for the time being we still need some dummy headers for some definitions
    include_directories(${CAGPUDIR}/HLTHeaders ${CAGPUDIR}/Standalone/include)
    add_definitions(-DHLTCA_STANDALONE)
endif()
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    add_definitions(-DHLTCA_TPC_GEOMETRY_O2 -DHLTCA_BUILD_O2_LIB)
endif()

#compile CUDA object file
cuda_compile(CAGPU_O AliHLTTPCCAGPUTrackerNVCC.cu)

#end of CUDA specific stuff; from here on build usual library

# Sources in alphabetical order
set(SRCS
    ${CAGPU_O}
   )

# Headers from sources
set(CINTHDRS
    AliHLTTPCCAGPUTrackerNVCC.h
    )

set(HDRS
    ${CINTHDRS}
    ${CAGPUDIR}/GlobalTracker/AliHLTTPCCAGPUTrackerCommon.h
  )

#Default cmake build script for AliRoot
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "ALIROOT")
    # Generate the dictionary
    # It will create G_ARG1.cxx and G_ARG1.h / ARG1 = function first argument
    get_directory_property(incdirs INCLUDE_DIRECTORIES)
    generate_dictionary("${MODULE}" "" "${CINTHDRS}" "${incdirs}")

    # Generate the ROOT map
    # Dependecies
    generate_rootmap("${MODULE}" "" "")

    # Add a library to the project using the specified source files
    add_library_tested(${MODULE} SHARED ${SRCS} G__${MODULE}.cxx)
    #CUDA run-time and driver
    target_link_libraries(${MODULE} ${CUDA_LIBRARIES} ${LIBCUDA_SO_PATH} AliTPCCAGPUTracking)

    # Additional compilation flags
    set_target_properties(${MODULE} PROPERTIES COMPILE_FLAGS "")

    # Installation
    install(TARGETS ${MODULE}
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)

    install(FILES ${HDRS} DESTINATION include)
endif()

#Default cmake build script for O2
if(ALITPCCOMMON_BUILD_TYPE STREQUAL "O2")
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME TPCCAGPUTracking_bucket)

    O2_GENERATE_LIBRARY()

    target_link_libraries(${MODULE} ${CUDA_LIBRARIES} ${LIBCUDA_SO_PATH} O2TPCCAGPUTracking)

    set_target_properties(${MODULE} PROPERTIES LINKER_LANGUAGE CXX)
endif()
