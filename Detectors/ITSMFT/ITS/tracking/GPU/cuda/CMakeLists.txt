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

# CUDA
if(CUDA_ENABLED)
    find_package(CUDAToolkit)
    message(STATUS "Building ITS CUDA tracker")
    o2_add_library(ITStrackingCUDA
                   SOURCES TrackerTraitsGPU.cxx
                           TimeFrameGPU.cu
                           TrackingKernels.cu
                   PUBLIC_INCLUDE_DIRECTORIES ../
                   PUBLIC_LINK_LIBRARIES O2::ITStracking
                                         O2::MathUtils
                                         O2::SimConfig
                                         O2::SimulationDataFormat
                                         O2::ReconstructionDataFormats
                                         O2::GPUCommon
                   PRIVATE_LINK_LIBRARIES O2::GPUTrackingCUDAExternalProvider
                   TARGETVARNAME targetName)

    set_target_gpu_arch("CUDA" ${targetName})
    # Enable relocatable device code (needed for separable compilation + debugging)
    set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${targetName} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-error=20014>
    #     $<$<COMPILE_LANGUAGE:CUDA>:-G;-O0;-Xptxas=-O0>
    #     $<$<COMPILE_LANGUAGE:CXX>:-O0;-g>
    )
    # target_compile_definitions(${targetName} PRIVATE ITS_MEASURE_GPU_TIME ITS_GPU_LOG)
    target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::ITStracking,COMPILE_DEFINITIONS>)
endif()
