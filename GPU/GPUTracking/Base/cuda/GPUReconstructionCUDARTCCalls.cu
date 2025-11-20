// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDARTCCalls.cu
/// \author David Rohr

#define GPUCA_GPUCODE_HOSTONLY
#define GPUCA_GPUCODE_NO_LAUNCH_BOUNDS

#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_STRIP(args))

#include "GPUReconstructionCUDAIncludesSystem.h"
#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionCUDA.h"

using namespace o2::gpu;

void GPUReconstructionCUDA::getRTCKernelCalls(std::vector<std::string>& kernels)
{
#undef GPUCA_KRNL
#define GPUCA_KRNL(...) kernels.emplace_back(GPUCA_M_STR(GPUCA_KRNLGPU(__VA_ARGS__)));
#undef __launch_bounds__
#include "GPUReconstructionKernelList.h"
}
