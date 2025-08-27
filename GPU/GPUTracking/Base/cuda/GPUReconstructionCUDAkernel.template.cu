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

/// \file GPUReconstructionCUDAkernel.cu
/// \author David Rohr

#define GPUCA_GPUCODE_COMPILEKERNELS
#include "GPUReconstructionCUDAIncludesSystem.h"
#define GPUCA_KRNL_REG(...) GPUCA_KRNL_REG_DEFAULT(__VA_ARGS__)
#define GPUCA_KRNL(...) GPUCA_KRNLGPU(__VA_ARGS__);
#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionKernelMacros.h"

// clang-format off
@O2_GPU_KERNEL_TEMPLATE_FILES@
// clang-format on

extern "C" {
// clang-format off
@O2_GPU_KERNEL_TEMPLATE_REPLACE@
// clang-format on
}
