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

message(STATUS "Building GPU HIP benchmark")
# Hipify-perl to generate HIP sources
set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")
file(GLOB CUDA_SOURCES_FULL_PATH "../cuda/*.cu")
foreach(file ${CUDA_SOURCES_FULL_PATH})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
  get_filename_component(CUDA_SOURCE ${file} NAME)
  string(REPLACE ".cu" "" CUDA_SOURCE_NAME ${CUDA_SOURCE})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE_NAME}.hip
    COMMAND ${HIPIFY_EXECUTABLE} --quiet-warnings ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/${CUDA_SOURCE} | sed '1{/\#include \"hip\\/hip_runtime.h\"/d}' > ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE_NAME}.hip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/${CUDA_SOURCE}
  )
endforeach()
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_HIP_COMPILE_FEATURES: ${CMAKE_HIP_COMPILE_FEATURES}")
message(STATUS "CMAKE_LINKER: ${CMAKE_LINKER}")
message(STATUS "CMAKE_HIP_STANDARD_LIBRARIES: ${CMAKE_HIP_STANDARD_LIBRARIES}")
message(STATUS "CMAKE_HIP_LINK_EXECUTABLE: ${CMAKE_HIP_LINK_EXECUTABLE}")
message(STATUS "CMAKE_HIP_LINK_FLAGS: ${CMAKE_HIP_LINK_FLAGS}")
message(STATUS "CMAKE_HIP_RUNTIME_LIBRARIES_STATIC: ${CMAKE_HIP_RUNTIME_LIBRARIES_STATIC}")
message(STATUS "CMAKE_HIP_RUNTIME_LIBRARIES_SHARED: ${CMAKE_HIP_RUNTIME_LIBRARIES_SHARED}")
message(STATUS "CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
o2_add_executable(gpu-memory-benchmark-hip
                  SOURCES benchmark.hip
                          Kernels.hip
                  PUBLIC_LINK_LIBRARIES hip::host
                                        hip-lang::device
                                        Boost::program_options
                  TARGETVARNAME targetName)
# >:-)
include_directories(${Boost_INCLUDE_DIRS})
add_executable(alternative-gpu-memory-benchmark-hip benchmark.hip Kernels.hip)
target_link_libraries(alternative-gpu-memory-benchmark-hip Boost::program_options ROOT::Tree)
install(TARGETS alternative-gpu-memory-benchmark-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# if(O2_HIP_CMAKE_LINK_FLAGS)
  # Need to add gpu target also to link flags due to gpu-rdc option
  # target_link_options(${targetName} PUBLIC ${O2_HIP_CMAKE_LINK_FLAGS})
# endif()
