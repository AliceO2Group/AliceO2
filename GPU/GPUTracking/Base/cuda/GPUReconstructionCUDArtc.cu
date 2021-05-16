// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDArtc.cu
/// \author David Rohr

#include "GPUReconstructionCUDADef.h"
#include "GPUReconstructionIncludesDevice.h"

#ifndef GPUCA_GPUCODE_DEVICE
#error RTC Preprocessing must run on device code
#endif

extern "C" {
#undef GPUCA_KRNL_REG
#define GPUCA_KRNL_REG(args) __launch_bounds__(GPUCA_M_MAX2_3(GPUCA_M_STRIP(args)))
#define GPUCA_KRNL(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNL_WRAP(GPUCA_KRNL_LOAD_, x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_single(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNLGPU_SINGLE(x_class, x_attributes, x_arguments, x_forward)
#define GPUCA_KRNL_LOAD_multi(x_class, x_attributes, x_arguments, x_forward) GPUCA_KRNLGPU_MULTI(x_class, x_attributes, x_arguments, x_forward)
#include "GPUReconstructionKernels.h"
#undef GPUCA_KRNL
}
