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

o2_add_library(ALICE3GlobalReconstructionWorkflow
               TARGETVARNAME targetName
               SOURCES src/TrackerSpec.cxx
                       src/TrackWriterSpec.cxx
                       src/RecoWorkflow.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::GPUWorkflow
                                     O2::SimConfig
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsTRK
                                     O2::SimulationDataFormat
                                     O2::DPLUtils
                                     O2::TRKBase
                                     O2::TRKSimulation
                                     O2::ALICE3GlobalReconstruction
                                     O2::CommonUtils
                                     nlohmann_json::nlohmann_json)

if(CUDA_ENABLED OR HIP_ENABLED)
  target_compile_definitions(${targetName} PUBLIC TRK_HAS_GPU_TRACKING)
endif()

if(CUDA_ENABLED)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(${targetName} PUBLIC TRK_HAS_CUDA_TRACKING)
  o2_add_library(ALICE3GlobalReconstructionWorkflowCUDA
                 TARGETVARNAME cudaTargetName
                 SOURCES src/TrackerSpecGPU.cxx
                         ../reconstruction/src/TimeFrameGPU.cxx
                         ../reconstruction/src/GPUExternalAllocator.cu
                 PUBLIC_LINK_LIBRARIES
                         O2::ALICE3GlobalReconstructionWorkflow
                         O2::ITStrackingCUDA
                 PRIVATE_LINK_LIBRARIES
                         CUDA::cudart)
  target_include_directories(${cudaTargetName} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(HIP_ENABLED)
  target_compile_definitions(${targetName} PUBLIC TRK_HAS_HIP_TRACKING)
  o2_add_hipified_library(ALICE3GlobalReconstructionWorkflowHIP
                          SOURCES src/TrackerSpecGPU.cxx
                                  ../reconstruction/src/TimeFrameGPU.cxx
                                  ../reconstruction/src/GPUExternalAllocator.cu
                          PUBLIC_LINK_LIBRARIES
                                  O2::ALICE3GlobalReconstructionWorkflow
                                  O2::ITStrackingHIP
                          PRIVATE_LINK_LIBRARIES
                                  hip::host)
endif()

o2_add_executable(reco-workflow
                  SOURCES src/alice3-global-reconstruction-workflow.cxx
                  COMPONENT_NAME alice3-global-reconstruction
                  PUBLIC_LINK_LIBRARIES O2::ALICE3GlobalReconstructionWorkflow
                                        O2::TRKSimulation
                                        O2::ALICE3GlobalReconstruction
                                        O2::ITStracking)
