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

#ifdef GPUCOMMONDEF_H
#error "this header establishes GPUCA_GPUTYPE_METAL and __METAL_HOST__, so it must be included before any other GPU header"
#endif

#define GPUCA_GPUTYPE_METAL

#import <Metal/Metal.h>
#ifndef __METAL_VERSION__
// __METAL_VERSION__ is only defined in device code.
#define __METAL_HOST__
// The parameter table picks MSL's `half` for the device layouts this translation
// unit has to agree with; _Float16 is the same IEEE binary16 type.
using half = _Float16;
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

  std::array<id<MTLCommandQueue>, constants::GPU_MAX_STREAMS> commandQueues;
  // The buffer currently open for encoding, plus the ones already committed and
  // not yet waited for. A Metal command buffer takes one commit and cannot be
  // encoded into afterwards, so the two cannot be the same object.
  std::array<id<MTLCommandBuffer>, constants::GPU_MAX_STREAMS> commandBuffers;
  std::array<std::vector<id<MTLCommandBuffer>>, constants::GPU_MAX_STREAMS> submittedBuffers;

  std::vector<id<MTLFunction>> functions;
  std::vector<id<MTLComputePipelineState>> pipelines;
  std::vector<uint32_t> maxThreadsPerGroup;

  id<MTLBuffer> mem_gpu;
  id<MTLBuffer> mem_constant;
  id<MTLBuffer> mem_host;

  id<MTLLibrary> library;
};
} // namespace o2::gpu

#endif // GPURECONSTRUCTIONOMETALINCLUDESHOST_H
