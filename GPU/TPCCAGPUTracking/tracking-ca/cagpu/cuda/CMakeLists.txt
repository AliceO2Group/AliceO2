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
set(MODULE AliHLTTPCCAGPU)

find_package(CUDA)
if(NOT CUDA_FOUND)
    message( FATAL_ERROR "NVIDIA CUDA package not found. Please install or set HLT_CUDA=OFF" )
else()
    find_library(LIBCUDA_SO_PATH libcuda.so)
    string(FIND ${LIBCUDA_SO_PATH} "-NOTFOUND" LIBCUDA_SO_PATH_NOTFOUND )
endif(NOT CUDA_FOUND)
message( STATUS "Building HLT with CUDA support" )

if(LIBCUDA_SO_PATH_NOTFOUND GREATER -1)
  message( FATAL_ERROR "NVIDIA CUDA libcuda.so not found. Please install or set HLT_CUDA=OFF" )
endif(LIBCUDA_SO_PATH_NOTFOUND GREATER -1)

#set nvcc flags
set(CUDA_NVCC_FLAGS --use_fast_math --maxrregcount 64 -O4 -Xcompiler -fPIC -Xptxas -O4 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)

#for convenience
set(CAGPUDIR ${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca/cagpu)

# Module include folder
include_directories(${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca/cagpu/cuda
                    ${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca/cagpu
                    ${AliRoot_SOURCE_DIR}/HLT/TPCLib/tracking-ca
                   )

# Additional include folders in alphabetical order except ROOT
include_directories(${ROOT_INCLUDE_DIR}
                    ${AliRoot_SOURCE_DIR}/HLT/BASE
                   )

#nvcc fails with std=c++11, remove it temporarily
if(CMAKE_CXX_FLAGS)
string(FIND ${CMAKE_CXX_FLAGS} "-std=c++11" STDCXX11FOUND )
if(STDCXX11FOUND GREATER -1)
  string(REPLACE "-std=c++11" "-std=c++98" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
endif(CMAKE_CXX_FLAGS)

#compile CUDA object file
cuda_compile(CAGPU_O AliHLTTPCCAGPUTrackerNVCC.cu)

#set it back
if(STDCXX11FOUND GREATER -1)
  string ( REPLACE "-std=c++98" "-std=c++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
endif()
#end of CUDA specific stuff; from here on build usual library


# bit of a hack, but otherwise dictionary compilation fails:
# defines cudaError_t to make it known
# defines __ROOT__; triggers "ifdef" in source code to circumvent broken cint behaviour with "volatile" keyword
# not sure why __ROOT__ is used here, could just as well be something else
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/G__AliHLTTPCCAGPU.cxx PROPERTIES COMPILE_DEFINITIONS "cudaError_t=int;__ROOT__")

# Sources in alphabetical order
set(SRCS
    ${CAGPUDIR}/AliHLTTPCCAGPUTrackerBase.cxx
    ${CAGPU_O}
   )

# Headers from sources
set(CINTHDRS
    ${CAGPUDIR}/AliHLTTPCCAGPUTrackerBase.h
    AliHLTTPCCAGPUTrackerNVCC.h
    )
  
set(HDRS
    ${CINTHDRS}
    ${CAGPUDIR}/AliHLTTPCCAGPUTrackerCommon.h
  )

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
target_link_libraries(${MODULE} ${CUDA_LIBRARIES} ${LIBCUDA_SO_PATH} AliHLTTPC)

# Additional compilation flags
set_target_properties(${MODULE} PROPERTIES COMPILE_FLAGS "")

# Installation
install(TARGETS ${MODULE}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES ${HDRS} DESTINATION include)
