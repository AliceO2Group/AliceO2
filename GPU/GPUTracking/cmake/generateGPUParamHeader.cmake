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

# file generateGPUParamHeader.cmake
# author Gabriele Cimador

function(generate_gpu_param_header GPU_ARCH OUT_HEADER)
  set(GPU_PARAM_JSON ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Definitions/GPUParameters.json)
  set(TARGET_ARCH "UNKNOWN")
  if(GPU_ARCH STREQUAL "AUTO")
      detect_gpu_arch("AUTO")
  else()
    set(TARGET_ARCH ${GPU_ARCH})
  endif()
  add_custom_command(
    OUTPUT ${OUT_HEADER}
    COMMAND ${CMAKE_COMMAND}
            -DOUT_HEADER=${OUT_HEADER}
            -DGPU_PARAM_JSON=${GPU_PARAM_JSON}
            -DTARGET_ARCH_SHORT=${TARGET_ARCH}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu_param_header_generator.cmake
    DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu_param_header_generator.cmake
            ${GPU_PARAM_JSON}
    COMMENT "Generating GPU parameter header for ${TARGET_ARCH}"
    VERBATIM
  )
  add_custom_target(GPU_PARAM_HEADER_${GPU_ARCH}_ALL ALL DEPENDS ${OUT_HEADER})
endfunction()