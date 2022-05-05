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

if(HIP_ENABLED)
  # Hipify-perl to generate HIP sources
  set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")
  file(GLOB CUDA_SOURCES_FULL_PATH "../cuda/*.cu")
  foreach(file ${CUDA_SOURCES_FULL_PATH})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
    get_filename_component(CUDA_SOURCE ${file} NAME)
    string(REPLACE ".cu" "" CUDA_SOURCE_NAME ${CUDA_SOURCE})
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE_NAME}.hip.cxx
      COMMAND ${HIPIFY_EXECUTABLE} --quiet-warnings ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/${CUDA_SOURCE} | sed '1{/\#include \"hip\\/hip_runtime.h\"/d}' > ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE_NAME}.hip.cxx
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/${CUDA_SOURCE}
    )
  endforeach()

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${baseTargetName}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc")

  message(STATUS "Building ITS HIP vertexer")
  o2_add_library(ITStrackingHIP
                 SOURCES ClusterLinesGPU.hip.cxx
                         Context.hip.cxx
                         TimeFrameGPU.hip.cxx
                         Stream.hip.cxx
                         TrackerTraitsGPU.hip.cxx
                         VertexerTraitsGPU.hip.cxx
                         Utils.hip.cxx
                 PUBLIC_INCLUDE_DIRECTORIES ../
                 PUBLIC_LINK_LIBRARIES O2::ITStracking
                                       hip::host
                                       hip::device
                                       hip::hipcub
                 TARGETVARNAME targetName)

  target_compile_definitions(
    ${targetName} PRIVATE $<TARGET_PROPERTY:O2::ITStracking,COMPILE_DEFINITIONS>)

  if(HIP_AMDGPUTARGET)
    # Need to add gpu target also to link flags due to gpu-rdc option
    target_link_options(${targetName} PUBLIC --amdgpu-target=${HIP_AMDGPUTARGET})
  endif()
endif()