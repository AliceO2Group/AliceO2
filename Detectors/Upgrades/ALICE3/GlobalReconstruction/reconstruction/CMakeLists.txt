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

if(Acts_FOUND)
  set(actsTarget Acts::Core)
endif()

set(alice3GlobalRecoGpuSources "")
set(alice3GlobalRecoGpuTargets "")
set(alice3GlobalRecoGpuPrivateTargets "")
if(CUDA_ENABLED)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND alice3GlobalRecoGpuSources src/TimeFrameGPU.cxx src/GPUExternalAllocator.cxx)
  list(APPEND alice3GlobalRecoGpuTargets O2::ITStrackingCUDA)
  list(APPEND alice3GlobalRecoGpuPrivateTargets CUDA::cudart)
elseif(HIP_ENABLED)
  list(APPEND alice3GlobalRecoGpuSources src/TimeFrameGPU.cxx src/GPUExternalAllocator.cxx)
  list(APPEND alice3GlobalRecoGpuTargets O2::ITStrackingHIP)
  list(APPEND alice3GlobalRecoGpuPrivateTargets hip::host)
endif()

o2_add_library(ALICE3GlobalReconstruction
               TARGETVARNAME targetName
               SOURCES src/TimeFrame.cxx
                       ${alice3GlobalRecoGpuSources}
                       $<$<BOOL:${Acts_FOUND}>:src/TrackerACTS.cxx>
               PUBLIC_LINK_LIBRARIES
                       O2::ITStracking
                       O2::GPUCommon
                       Microsoft.GSL::GSL
                       O2::CommonConstants
                       O2::DataFormatsITSMFT
                       O2::DataFormatsTRK
                       O2::SimulationDataFormat
                       O2::ITSBase
                       O2::ITSReconstruction
                       O2::ITSMFTReconstruction
                       O2::DataFormatsITS
                       O2::TRKBase
                       O2::TRKReconstruction
                       O2::TRKSimulation
                       nlohmann_json::nlohmann_json
                       ${alice3GlobalRecoGpuTargets}
                       ${actsTarget}
               PRIVATE_LINK_LIBRARIES
                       O2::Steer
                       TBB::tbb
                       ${alice3GlobalRecoGpuPrivateTargets})

if(alice3GlobalRecoGpuTargets)
  target_compile_definitions(${targetName} PUBLIC TRK_HAS_GPU_TRACKING)
endif()

if(CUDA_ENABLED)
  target_include_directories(${targetName} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(CUDA_ENABLED)
  target_compile_definitions(${targetName} PUBLIC TRK_HAS_CUDA_TRACKING)
elseif(HIP_ENABLED)
  target_compile_definitions(${targetName} PUBLIC TRK_HAS_HIP_TRACKING)
endif()

if(Acts_FOUND)
  target_compile_definitions(${targetName} PUBLIC O2_WITH_ACTS)
endif()
