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

#include <cuda_runtime.h>
#include <cuda.h>

#include "GPUReconstructionCUDArtcPre.h"

#define GPUCA_GPUTYPE_TURING
#include "GPUReconstructionIncludesDevice.h"

extern "C" __global__ void foo()
{
  if (threadIdx.x || blockIdx.x)
    return;
  printf("test\n");
}
