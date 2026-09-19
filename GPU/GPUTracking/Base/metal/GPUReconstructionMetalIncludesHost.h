// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef GPURECONSTRUCTIONOMETALINCLUDESHOST_H
#define GPURECONSTRUCTIONOMETALINCLUDESHOST_H

#define GPUCA_GPUTYPE_METAL

#import <Metal/Metal.h>
#ifndef __METAL_VERSION__
// __METAL_VERSION__ is only defined in device code.
#define __METAL_HOST__
#endif

#import <Foundation/Foundation.h>

#include <vector>
#include <string>
#include <memory>
#include "GPULogging.h"

#include "GPUReconstructionMetal.h"
#include "GPUReconstructionIncludes.h"
#include "GPUCommonHelpers.h"

using namespace o2::gpu;

#include <cstring>
#include <unistd.h>
#include <typeinfo>
#include <cstdlib>

namespace o2::gpu
{

struct GPUReconstructionMetalInternals {
  id<MTLDevice> device;

  std::array<id<MTLCommandQueue>, constants::GPU_MAX_STREAMS> commandQueues; // ~ cl_command_queue[]
  std::array<id<MTLCommandBuffer>, constants::GPU_MAX_STREAMS> commandBuffers;

  std::vector<id<MTLFunction>> functions;             // ~ cl_kernel (symbols)
  std::vector<id<MTLComputePipelineState>> pipelines; // compiled kernels

  id<MTLBuffer> mem_gpu;      // ~ cl_mem (device/global)
  id<MTLBuffer> mem_constant; // ~ cl_mem (constant-like)
  id<MTLBuffer> mem_host;     // ~ cl_mem (host-visible)

  id<MTLLibrary> library; // ~ cl_program
};
} // namespace o2::gpu

#endif // GPURECONSTRUCTIONOMETALINCLUDESHOST_H
