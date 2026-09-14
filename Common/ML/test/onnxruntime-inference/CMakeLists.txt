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

cmake_minimum_required(VERSION 3.16)
project(onnxruntime_inference_test LANGUAGES CXX)

find_package(onnxruntime CONFIG REQUIRED)

add_executable(onnxruntime-ep-inference onnxruntime_ep_inference.cxx)
target_compile_features(onnxruntime-ep-inference PRIVATE cxx_std_17)
target_link_libraries(onnxruntime-ep-inference PRIVATE onnxruntime::onnxruntime)
target_compile_definitions(onnxruntime-ep-inference PRIVATE
  $<$<BOOL:${ORT_CUDA_BUILD}>:ORT_CUDA_BUILD>
  $<$<BOOL:${ORT_MIGRAPHX_BUILD}>:ORT_MIGRAPHX_BUILD>
  $<$<BOOL:${ORT_TENSORRT_BUILD}>:ORT_TENSORRT_BUILD>)

install(TARGETS onnxruntime-ep-inference RUNTIME DESTINATION bin)
install(PROGRAMS
  run-onnxruntime-all-eps.sh
  run-onnxruntime-cpu.sh
  run-onnxruntime-cuda.sh
  run-onnxruntime-migraphx.sh
  run-onnxruntime-tensorrt.sh
  run-local-onnxruntime-inference-test.sh
  DESTINATION bin)
install(FILES net.onnx DESTINATION test/onnxruntime-inference)
