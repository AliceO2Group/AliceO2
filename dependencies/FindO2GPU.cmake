# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(NOT DEFINED ENABLE_CUDA)
  set(ENABLE_CUDA "AUTO")
endif()
if(NOT DEFINED ENABLE_OPENCL1)
  set(ENABLE_OPENCL1 "AUTO")
endif()
if(NOT DEFINED ENABLE_OPENCL2)
  set(ENABLE_OPENCL2 "AUTO")
endif()
if(NOT DEFINED ENABLE_HIP)
  set(ENABLE_HIP "AUTO")
endif()
string(TOUPPER "${ENABLE_CUDA}" ENABLE_CUDA)
string(TOUPPER "${ENABLE_OPENCL1}" ENABLE_OPENCL1)
string(TOUPPER "${ENABLE_OPENCL2}" ENABLE_OPENCL2)
string(TOUPPER "${ENABLE_HIP}" ENABLE_HIP)

# Detect and enable CUDA
if(ENABLE_CUDA)
  set(CUDA_MINIMUM_VERSION "11.0")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  include(CheckLanguage)
  check_language(CUDA)
  if(CUDA_GCCBIN)
    message(STATUS "Using as CUDA GCC version: ${CUDA_GCCBIN}")
    set(CMAKE_CUDA_HOST_COMPILER "${CUDA_GCCBIN}")
    if (NOT CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER "nvcc") #check_language does not treat the HOST_COMPILER flag correctly, we force it and will fail below if wrong.
    endif()
  endif()
  if(CMAKE_CUDA_COMPILER)
    cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
    enable_language(CUDA)
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT CUDA IN_LIST LANGUAGES)
      message(FATAL_ERROR "CUDA was found but cannot be enabled")
    endif()
    find_path(THRUST_INCLUDE_DIR thrust/version.h PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} NO_DEFAULT_PATH)
    if(THRUST_INCLUDE_DIR STREQUAL "THRUST_INCLUDE_DIR-NOTFOUND")
      message(FATAL_ERROR "CUDA found but thrust not available")
    endif()

    # Forward CXX flags to CUDA C++ Host compiler (for warnings, gdb, etc.)
    STRING(REGEX REPLACE "\-std=[^ ]*" "" CMAKE_CXX_FLAGS_NOSTD ${CMAKE_CXX_FLAGS}) # Need to strip c++17 imposed by alidist defaults
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CMAKE_CXX_FLAGS_NOSTD}\" --expt-relaxed-constexpr --extended-lambda --allow-unsupported-compiler -Xptxas -v")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo -Xcompiler \"${CMAKE_CXX_FLAGS_DEBUG}\" -Xptxas -O0 -Xcompiler -O0")
    if(NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG")
      set(CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE} "${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}} -Xcompiler \"${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}\" -Xptxas -O4 -Xcompiler -O4 -use_fast_math --ftz=true")
    endif()
    if(CMAKE_CXX_FLAGS MATCHES "(^| )-Werror( |$)")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror=cross-execution-space-call")
    endif()
    if(CUDA_COMPUTETARGET)
      set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTETARGET})
    else()
      set(CMAKE_CUDA_ARCHITECTURES 61-virtual)
    endif()

    set(CUDA_ENABLED ON)
    message(STATUS "CUDA found (Version ${CMAKE_CUDA_COMPILER_VERSION})")
  elseif(NOT ENABLE_CUDA STREQUAL "AUTO")
    message(FATAL_ERROR "CUDA not found (Compiler: ${CMAKE_CUDA_COMPILER})")
  endif()
endif()

# Detect and enable OpenCL 1.2 from AMD
if(ENABLE_OPENCL1 OR ENABLE_OPENCL2)
  find_package(OpenCL)
  if((ENABLE_OPENCL1 AND NOT ENABLE_OPENCL1 STREQUAL "AUTO")
     OR (ENABLE_OPENCL2 AND NOT ENABLE_OPENCL2 STREQUAL "AUTO"))
    set_package_properties(OpenCL PROPERTIES TYPE REQUIRED)
  else()
    set_package_properties(OpenCL PROPERTIES TYPE OPTIONAL)
  endif()
endif()
if(ENABLE_OPENCL1)
  if(NOT AMDAPPSDKROOT)
    set(AMDAPPSDKROOT "$ENV{AMDAPPSDKROOT}")
  endif()

  if(OpenCL_FOUND
     AND OpenCL_VERSION_STRING VERSION_GREATER_EQUAL 1.2
     AND AMDAPPSDKROOT
     AND EXISTS "${AMDAPPSDKROOT}")
    set(OPENCL1_ENABLED ON)
    message(STATUS "Found AMD OpenCL 1.2")
  elseif(NOT ENABLE_OPENCL1 STREQUAL "AUTO")
    message(FATAL_ERROR "AMD OpenCL 1.2 not available")
  endif()
endif()

# Detect and enable OpenCL 2.x
if(ENABLE_OPENCL2)
  find_package(LLVM)
  if(LLVM_FOUND)
    find_package(Clang)
  endif()
  find_package(ROCM PATHS /opt/rocm)
  if(NOT ROCM_DIR STREQUAL "ROCM_DIR-NOTFOUND")
    get_filename_component(ROCM_ROOT "${ROCM_DIR}/../../../" ABSOLUTE)
  else()
    set(ROCM_ROOT "/opt/rocm")
  endif()
  find_program(CLANG_OCL clang-ocl PATHS "${ROCM_ROOT}/bin")
  find_program(ROCM_AGENT_ENUMERATOR rocm_agent_enumerator PATHS "${ROCM_ROOT}/bin")
  find_program(LLVM_SPIRV llvm-spirv)
  if(Clang_FOUND
     AND LLVM_FOUND
     AND LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL 10.0)
    set(OPENCL2_COMPATIBLE_CLANG_FOUND ON)
  endif()
  if(OpenCL_VERSION_STRING VERSION_GREATER_EQUAL 2.0
     AND OPENCL2_COMPATIBLE_CLANG_FOUND
     AND NOT CLANG_OCL STREQUAL "CLANG_OCL-NOTFOUND")
    set(OPENCL2_ENABLED_AMD ON)
  endif()
  if(OpenCL_VERSION_STRING VERSION_GREATER_EQUAL 2.2
     AND OPENCL2_COMPATIBLE_CLANG_FOUND
     AND NOT LLVM_SPIRV STREQUAL "LLVM_SPIRV-NOTFOUND")
    set(OPENCL2_ENABLED_SPIRV ON)
  endif ()
  if(OPENCL2_COMPATIBLE_CLANG_FOUND AND
     (OpenCL_VERSION_STRING VERSION_GREATER_EQUAL 2.2
     OR OPENCL2_ENABLED_AMD
     OR OPENCL2_ENABLED_SPIRV))
    set(OPENCL2_ENABLED ON)
    message(
      STATUS
        "Found OpenCL 2 (${OpenCL_VERSION_STRING} ; AMD ${OPENCL2_ENABLED_AMD} ${CLANG_OCL} ; SPIR-V ${OPENCL2_ENABLED_SPIRV} ${LLVM_SPIRV} with CLANG ${LLVM_PACKAGE_VERSION})"
      )
  elseif(NOT ENABLE_OPENCL2 STREQUAL "AUTO")
    message(FATAL_ERROR "OpenCL 2.x not available")
  endif()
  if (FORCE_OPENCL2_ALL AND NOT(OPENCL2_ENABLED_AMD AND OPENCL2_ENABLED_SPIRV))
    message(FATAL_ERROR "Not all OpenCL2 backends available, but requested (AMD ${OPENCL2_ENABLED_AMD} SPIRV ${OPENCL2_ENABLED_SPIRV})")
  endif()
endif()

# Detect and enable HIP
if(ENABLE_HIP)
  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH
          "/opt/rocm/hip"
          CACHE PATH "Path to which HIP has been installed")
    else()
      set(HIP_PATH
          $ENV{HIP_PATH}
          CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()

  if(EXISTS "${HIP_PATH}")
    get_filename_component(hip_ROOT "${HIP_PATH}" ABSOLUTE)
    find_package(hip)
    find_package(hipcub)
    find_package(rocprim)
    find_package(rocthrust)
    if(ENABLE_HIP STREQUAL "AUTO")
      set_package_properties(hip PROPERTIES TYPE OPTIONAL)
      set_package_properties(hipcub PROPERTIES TYPE OPTIONAL)
      set_package_properties(rocprim PROPERTIES TYPE OPTIONAL)
      set_package_properties(rocthrust PROPERTIES TYPE OPTIONAL)
    else()
      set_package_properties(hip PROPERTIES TYPE REQUIRED)
      set_package_properties(hipcub PROPERTIES TYPE REQUIRED)
      set_package_properties(rocprim PROPERTIES TYPE REQUIRED)
      set_package_properties(rocthrust PROPERTIES TYPE REQUIRED)
    endif()
    if(hip_FOUND AND hipcub_FOUND AND rocthrust_FOUND AND rocprim_FOUND AND hip_HIPCC_EXECUTABLE)
      set(HIP_ENABLED ON)
      set_target_properties(roc::rocthrust PROPERTIES IMPORTED_GLOBAL TRUE)
      message(STATUS "HIP Found (${hip_HIPCC_EXECUTABLE})")
      set(O2_HIP_CMAKE_CXX_FLAGS "-fcuda-flush-denormals-to-zero -Wno-invalid-command-line-argument -Wno-unused-command-line-argument -Wno-invalid-constexpr -Wno-ignored-optimization-argument -Wno-unused-private-field")
      if(HIP_AMDGPUTARGET)
        set(O2_HIP_CMAKE_CXX_FLAGS "${O2_HIP_CMAKE_CXX_FLAGS} --amdgpu-target=${HIP_AMDGPUTARGET}")
      endif()
    endif()
  endif()
  if(NOT HIP_ENABLED AND NOT ENABLE_HIP STREQUAL "AUTO")
    message(
      FATAL_ERROR
        "HIP requested but HIP_PATH=${HIP_PATH} does not exist"
      )
  endif()

endif()

# if we end up here without a FATAL, it means we have found the "O2GPU" package
set(O2GPU_FOUND TRUE)
