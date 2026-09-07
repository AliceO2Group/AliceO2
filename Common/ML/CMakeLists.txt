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

o2_add_library(ML
               SOURCES src/OrtInterface.cxx
               TARGETVARNAME targetName
               PRIVATE_LINK_LIBRARIES O2::GPUCommon onnxruntime::onnxruntime)

# Pass ORT variables as a preprocessor definition
target_compile_definitions(${targetName} PRIVATE
    $<$<BOOL:${ORT_CUDA_BUILD}>:ORT_CUDA_BUILD>
    $<$<BOOL:${ORT_MIGRAPHX_BUILD}>:ORT_MIGRAPHX_BUILD>
    $<$<BOOL:${ORT_TENSORRT_BUILD}>:ORT_TENSORRT_BUILD>)
