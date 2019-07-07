# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

include_guard()

message(WARNING "Reimplement me")

if(FALSE)
  # FIXME: here also, some settings look suspicious : e.g. the setting of CUDA
  # flag based of CMAKE_BUILD_TYPE variable (should use $<CONFIG> based
  # generator expressions instead. Plus CUDA is now a language "understood" by
  # CMake, isn't ?

  set(CUDA_MINIMUM_VERSION "10.1")
  if(DEFINED ENABLE_CUDA AND NOT ENABLE_CUDA)
    message(STATUS "CUDA explicitly disabled")
  else()
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
      if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        set(CMAKE_CUDA_FLAGS "-Xptxas -O0 -Xcompiler -O0")
      else()
        set(CMAKE_CUDA_FLAGS "-Xptxas -O4 -Xcompiler -O4 -use_fast_math")
      endif()
      if(CUDA_GCCBIN)
        message(STATUS "Using as CUDA GCC version: ${CUDA_GCCBIN}")
        set(CMAKE_CUDA_FLAGS
            "${CMAKE_CUDA_FLAGS} --compiler-bindir ${CUDA_GCCBIN}")
      endif()

      enable_language(CUDA)

      get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

      if(NOT CUDA IN_LIST LANGUAGES)
        message(
          FATAL_ERROR "CUDA was found but cannot be enabled for some reason")
      endif()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "${CUDA_MINIMUM_VERSION}")
        message(
          FATAL_ERROR
            "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} found, but at least ${CUDA_MINIMUM_VERSION} required"
          )
      endif()
      set(ENABLE_CUDA ON)
      if(CUDA_GCCBIN) # Ugly hack! Otherwise CUDA includes unwanted old GCC
                      # libraries leading to # version conflicts
        set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "$ENV{CUDA_PATH}/lib64")
      endif()
      add_definitions(-DENABLE_CUDA)
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
      set(
        CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --compiler-options \"${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} -std=c++14\""
        )
    elseif(ENABLE_CUDA)
      message(FATAL_ERROR "CUDA explicitly enabled but could not be found")
    endif()
  endif()
endif()
