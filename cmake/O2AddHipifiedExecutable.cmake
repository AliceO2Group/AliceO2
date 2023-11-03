# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

include_guard()

include(O2AddExecutable)

function(o2_add_hipified_executable baseTargetName)
  # Parse arguments in the same way o2_add_executable does
  cmake_parse_arguments(PARSE_ARGV
                        1
                        A
                        "IS_TEST;NO_INSTALL;IS_BENCHMARK"
                        "COMPONENT_NAME;TARGETVARNAME"
                        "SOURCES;PUBLIC_LINK_LIBRARIES;JOB_POOL")

  # Process each .cu file to generate a .hip file
  set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")
  set(HIP_SOURCES)

  foreach(file ${A_SOURCES})
    get_filename_component(ABS_CUDA_SORUCE ${file} ABSOLUTE)
    if(file MATCHES "\\.cu$")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
      get_filename_component(CUDA_SOURCE ${file} NAME)
      string(REPLACE ".cu" ".hip" HIP_SOURCE ${CUDA_SOURCE})
      set(OUTPUT_HIP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${HIP_SOURCE}")
      list(APPEND HIP_SOURCES ${OUTPUT_HIP_FILE})

      add_custom_command(
        OUTPUT ${OUTPUT_HIP_FILE}
        COMMAND ${HIPIFY_EXECUTABLE} --quiet-warnings ${ABS_CUDA_SORUCE} | sed '1{/\#include \"hip\\/hip_runtime.h\"/d}' > ${OUTPUT_HIP_FILE}
        DEPENDS ${file}
      )
    else()
      list(APPEND HIP_SOURCES ${file})
    endif()
  endforeach()

  # This is a bit cumbersome, but it seems the only suitable since cmake_parse_arguments is not capable to filter only the SOURCE variadic values
  set(FORWARD_ARGS "")
  if(A_IS_TEST)
    list(APPEND FORWARD_ARGS "IS_TEST")
  endif()
  if(A_NO_INSTALL)
    list(APPEND FORWARD_ARGS "NO_INSTALL")
  endif()
  if(A_IS_BENCHMARK)
    list(APPEND FORWARD_ARGS "IS_BENCHMARK")
  endif()
  if(A_COMPONENT_NAME)
    list(APPEND FORWARD_ARGS "COMPONENT_NAME" ${A_COMPONENT_NAME})
  endif()
  if(A_TARGETVARNAME)
    list(APPEND FORWARD_ARGS "TARGETVARNAME" ${A_TARGETVARNAME})
  endif()
  if(A_PUBLIC_LINK_LIBRARIES)
    list(APPEND FORWARD_ARGS "PUBLIC_LINK_LIBRARIES" ${A_PUBLIC_LINK_LIBRARIES})
  endif()
  if(A_JOB_POOL)
    list(APPEND FORWARD_ARGS "JOB_POOL" ${A_JOB_POOL})
  endif()

  # Call o2_add_executable with new sources
  o2_add_executable("${baseTargetName}"
                    SOURCES ${HIP_SOURCES}
                    ${FORWARD_ARGS})
endfunction()