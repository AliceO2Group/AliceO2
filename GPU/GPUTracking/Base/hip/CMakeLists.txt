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
    set(MODULE O2GPUTrackingHIP)
elseif(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    set(MODULE GPUTrackingHIP)
    if(NOT DEFINED HIP_PATH)
      if(NOT DEFINED ENV{HIP_PATH})
         set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
      else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
      endif()
    endif()
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")
    if(NOT DEFINED HCC_PATH)
      # Workaround to fix a potential FindHIP bug: find HCC_PATH ourselves
      set(_HCC_PATH "${HIP_PATH}/../hcc")
      get_filename_component(HCC_PATH ${_HCC_PATH} ABSOLUTE CACHE)
      unset(_HCC_PATH)
    endif()
    find_package(HIP REQUIRED)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
        message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
    endif()
    add_definitions(-DENABLE_HIP)
endif()
set(DEFINITIONS ${DEFINITIONS} GPUCA_GPULIBRARY=HIP)

set(CMAKE_CXX_COMPILER ${HIP_PATH}/bin/hipcc)
add_compile_options(--amdgpu-target=gfx906 -Wno-unused-command-line-argument -Wno-invalid-constexpr)
add_link_options(--amdgpu-target=gfx906)

message( STATUS "Building GPUTracking with HIP support" )

#for convenience
set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)

# Module include folder
include_directories(${GPUDIR}/Base/hip)

# Additional include folders in alphabetical order except ROOT
include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
    include_directories(${CMAKE_SOURCE_DIR}/HLT/BASE)
else() #if we do not build for AliRoot, for the time being we still need some dummy headers for some definitions
    include_directories(${GPUDIR}/HLTHeaders ${GPUDIR}/Standalone/include)
endif()
if(ALIGPU_BUILD_TYPE STREQUAL "O2") #We need to add src dirs of O2 to include cxx files for HIP compilation
    include_directories(${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src)
    include_directories(${CMAKE_SOURCE_DIR}/Detectors/Base/src)
endif()

# Sources in alphabetical order
set(SRCS
    GPUReconstructionHIP.hip.cpp
   )

# Headers from sources
set(CINTHDRS
    GPUReconstructionHIP.h
    )

set(HDRS
    ${CINTHDRS}
    GPUReconstructionHIPInternals.h
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
    #HIP run-time and driver
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
    Set(LIBRARY_NAME ${MODULE})
    set(BUCKET_NAME GPUTrackingHIP_bucket)

    O2_GENERATE_LIBRARY()
    install(FILES ${HDRS} DESTINATION include/GPU)

    target_link_libraries(${MODULE} ${HIP_LIBRARIES} O2GPUTracking)
    set_target_properties(${MODULE} PROPERTIES LINKER_LANGUAGE CXX)
endif()
