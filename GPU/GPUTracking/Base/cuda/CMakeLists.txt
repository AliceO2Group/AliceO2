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

cmake_minimum_required(VERSION 3.9)

# Module
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(MODULE O2GPUTrackingCUDA)
elseif(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    set(MODULE GPUTrackingCUDA)
endif()
set(DEFINITIONS ${DEFINITIONS} GPUCA_GPULIBRARY=CUDA)

option(CUDA_GCCBIN "GCC binary to use for compiling host part of CUDA code for systems with multiple GCC versions installed"   OFF)

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if (NOT CUDA IN_LIST LANGUAGES) #We need to handle finding CUDA in AliRoot case
    if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
      set(CMAKE_CUDA_FLAGS "-Xptxas -O0 -Xcompiler -O0")
    else()
      set(CMAKE_CUDA_FLAGS "-Xptxas -O4 -Xcompiler -O4 -use_fast_math")
    endif()
    if(CUDA_GCCBIN)
      message(STATUS "Using as CUDA GCC version: ${CUDA_GCCBIN}")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-bindir ${CUDA_GCCBIN}")
    endif()
    enable_language(CUDA)
    if(CUDA_GCCBIN)
      #Ugly hack!: Otherwise CUDA includes unwanted old GCC libraries leading to version conflicts
      set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "$ENV{CUDA_PATH}/lib64")
    endif()
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -DENABLE_CUDA --compiler-options \"${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} -std=c++14\"")
endif()

#Another ugly hack!: nvcc dies due to O2 link flags, so we strip them for device linkage
set(CMAKE_CUDA_DEVICE_LINK_LIBRARY "<CMAKE_CUDA_COMPILER> <LANGUAGE_COMPILE_FLAGS> -Xcompiler=-fPIC -Wno-deprecated-gpu-targets -shared -dlink <OBJECTS> -o <TARGET>")

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
if (NOT CUDA IN_LIST LANGUAGES)
    message(FATAL_ERROR "CMake could not find CUDA (${LANGUAGES})")
endif()
message( STATUS "Building GPUTracking with CUDA support" )

#for convenience
set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)

# Module include folder
include_directories(${GPUDIR}/Base/cuda)

# Additional include folders in alphabetical order except ROOT
include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    include_directories(${CMAKE_SOURCE_DIR}/HLT/BASE)
else() #if we do not build for AliRoot, for the time being we still need some dummy headers for some definitions
    include_directories(${GPUDIR}/HLTHeaders ${GPUDIR}/Standalone/include)
endif()
if(ALIGPU_BUILD_TYPE STREQUAL "O2") #We need to add src dirs of O2 to include cxx files for CUDA compilation
    include_directories(${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src)
endif()

# Sources in alphabetical order
set(SRCS
    GPUReconstructionCUDA.cu
   )

# Headers from sources
set(CINTHDRS
    GPUReconstructionCUDA.h
    )

set(HDRS
    ${CINTHDRS}
    GPUReconstructionCUDAInternals.h
  )

foreach (def ${DEFINITIONS})
  add_definitions(-D${def})
endforeach()

#Default cmake build script for AliRoot
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
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
    target_link_libraries(${MODULE} GPUTracking)

    # Additional compilation flags
    set_target_properties(${MODULE} PROPERTIES COMPILE_FLAGS "")

    # Installation
    install(TARGETS ${MODULE}
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)

    install(FILES ${HDRS} DESTINATION include)
endif()

#Default cmake build script for O2
if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME GPUTrackingCUDA_bucket)

    O2_GENERATE_LIBRARY()
    install(FILES ${HDRS} DESTINATION include/GPU)

    target_link_libraries(${MODULE} ${CUDA_LIBRARIES} O2GPUTracking)
    set_target_properties(${MODULE} PROPERTIES LINKER_LANGUAGE CXX)
endif()

#Another hack, the above hack removes also the CUDA libraries that are actually needed...
target_link_libraries(${MODULE} "-lcudart -lcuda -lcudadevrt")
