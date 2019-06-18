
# * option(CUDA_GCCBIN "GCC binary to use for compiling host part of CUDA code
#   for systems with multiple GCC versions installed"   OFF)

# * get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
# * if (NOT CUDA IN_LIST LANGUAGES) #We need to handle finding CUDA in AliRoot
#   case
# * if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
# * set(CMAKE_CUDA_FLAGS "-Xptxas -O0 -Xcompiler -O0")
# * else()
# * set(CMAKE_CUDA_FLAGS "-Xptxas -O4 -Xcompiler -O4 -use_fast_math")
# * endif()
# * if(CUDA_GCCBIN)
# * message(STATUS "Using as CUDA GCC version: ${CUDA_GCCBIN}")
# * set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-bindir ${CUDA_GCCBIN}")
# * endif()
# * enable_language(CUDA)
# * if(CUDA_GCCBIN)
# * #Ugly hack!: Otherwise CUDA includes unwanted old GCC libraries leading to
#   version conflicts
# * set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "$ENV{CUDA_PATH}/lib64")
# * endif()
# * set(CMAKE_CUDA_STANDARD 14)
# * set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# * set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr
#   -DENABLE_CUDA --compiler-options \"${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}
#   -std=c++14\"")
# * endif()
#
# * #Another ugly hack!: nvcc dies due to O2 link flags, so we strip them for
#   device linkage
# * set(CMAKE_CUDA_DEVICE_LINK_LIBRARY "<CMAKE_CUDA_COMPILER>
#   <LANGUAGE_COMPILE_FLAGS> -Xcompiler=-fPIC -Wno-deprecated-gpu-targets
#   -shared -dlink <OBJECTS> -o <TARGET>")
#
# * get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
# * if (NOT CUDA IN_LIST LANGUAGES)
# * message(FATAL_ERROR "CMake could not find CUDA (${LANGUAGES})")
# * endif()

message(STATUS "Building GPUTracking with CUDA support")

# for convenience
set(GPUDIR ${CMAKE_SOURCE_DIR}/GPU/GPUTracking)

# # Module include folder include_directories(${GPUDIR}/Base/cuda)
#
# # Additional include folders in alphabetical order except ROOT
# include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
# include_directories(${GPUDIR}/HLTHeaders ${GPUDIR}/Standalone/include)
# include_directories(${CMAKE_SOURCE_DIR}/Detectors/TRD/base/src)
# include_directories(${CMAKE_SOURCE_DIR}/Detectors/Base/src)

o2_add_library(GPUTrackingCUDA
               SOURCES GPUReconstructionCUDA.cu
               PUBLIC_LINK_LIBRARIES O2::GPUTracking
               TARGETVARNAME targetName)

target_compile_definitions(
  ${targetName} PUBLIC GPUCA_GPULIBRARY=CUDA
  $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

set_target_properties(${targetName} PROPERTIES LINKER_LANGUAGE CXX)

# Another hack, the above hack removes also the CUDA libraries that are actually
# needed... target_link_libraries(GPUTrackingCUDA "-lcudart -lcuda -lcudadevrt")
