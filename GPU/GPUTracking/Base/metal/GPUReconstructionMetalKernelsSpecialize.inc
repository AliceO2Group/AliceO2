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

/// \file GPUReconstructionMetalKernelsSpecialize.inc

template <>
inline void GPUReconstructionMetal::runKernelBackend<GPUMemClean16, 0>(const krnlSetupTime& _xyz, void* const& ptr, uint64_t const& size)
{
  const uint64_t offset = static_cast<const uint8_t*>(ptr) - static_cast<const uint8_t*>(mDeviceMemoryBase);
  const uint64_t length = (size + 15ull) & ~15ull;

  id<MTLCommandBuffer> cb = mInternals->commandBuffers[_xyz.x.stream];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

  [blit fillBuffer:mInternals->mem_gpu range:NSMakeRange(offset, length) value:0];
  [blit endEncoding];

  [cb commit];
}
