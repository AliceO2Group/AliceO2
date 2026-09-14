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
    $<$<BOOL:${ORT_ROCM_BUILD}>:ORT_ROCM_BUILD>
    $<$<BOOL:${ORT_CUDA_BUILD}>:ORT_CUDA_BUILD>
    $<$<BOOL:${ORT_MIGRAPHX_BUILD}>:ORT_MIGRAPHX_BUILD>
    $<$<BOOL:${ORT_TENSORRT_BUILD}>:ORT_TENSORRT_BUILD>)

if(BUILD_TESTING)
    set(ONNXRUNTIME_INFERENCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/onnxruntime-inference)

    o2_add_executable(onnxruntime-ep-inference
                      SOURCES ${ONNXRUNTIME_INFERENCE_TEST_DIR}/onnxruntime_ep_inference.cxx
                      COMPONENT_NAME ML
                      IS_TEST
                      NO_INSTALL
                      TARGETVARNAME onnxruntimeInferenceTestTarget
                      PUBLIC_LINK_LIBRARIES onnxruntime::onnxruntime)
    target_compile_features(${onnxruntimeInferenceTestTarget} PRIVATE cxx_std_17)
    target_compile_definitions(${onnxruntimeInferenceTestTarget} PRIVATE
        $<$<BOOL:${ORT_CUDA_BUILD}>:ORT_CUDA_BUILD>
        $<$<BOOL:${ORT_MIGRAPHX_BUILD}>:ORT_MIGRAPHX_BUILD>
        $<$<BOOL:${ORT_TENSORRT_BUILD}>:ORT_TENSORRT_BUILD>)

    foreach(provider CUDA MIGRAPHX TENSORRT)
        set(ONNXRUNTIME_INFERENCE_TEST_${provider} 0)
        if(ORT_${provider}_BUILD)
            set(ONNXRUNTIME_INFERENCE_TEST_${provider} 1)
        endif()
    endforeach()

    o2_add_test_command(NAME Common/ML/onnxruntime-inference
                        COMMAND ${ONNXRUNTIME_INFERENCE_TEST_DIR}/run-onnxruntime-all-eps.sh
                        COMMAND_LINE_ARGS ${ONNXRUNTIME_INFERENCE_TEST_DIR}/net.onnx
                        ENVIRONMENT
                          "ONNXRUNTIME_INFERENCE_TEST_BINARY=$<TARGET_FILE:${onnxruntimeInferenceTestTarget}>"
                          "ORT_CUDA_BUILD=${ONNXRUNTIME_INFERENCE_TEST_CUDA}"
                          "ORT_MIGRAPHX_BUILD=${ONNXRUNTIME_INFERENCE_TEST_MIGRAPHX}"
                          "ORT_TENSORRT_BUILD=${ONNXRUNTIME_INFERENCE_TEST_TENSORRT}"
                        LABELS ml onnxruntime
                        TIMEOUT 300)
endif()
