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

#define GPUCA_GPUCODE_HOSTONLY

#include <cuda_runtime.h>

#include "ALICE3GlobalReconstruction/GPUExternalAllocator.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace
{
void checkGpuError(cudaError_t error, const char* call)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
  }
}
} // namespace

namespace o2::trk
{

GPUExternalAllocator::~GPUExternalAllocator()
{
  releaseAll();
}

void* GPUExternalAllocator::allocate(size_t size)
{
  const auto type = static_cast<MemoryType>(getType());
  const bool useHost = (type & static_cast<MemoryType>(o2::gpu::GPUMemoryResource::MEMORY_HOST)) != 0;
  const bool useStack = (type & static_cast<MemoryType>(o2::gpu::GPUMemoryResource::MEMORY_STACK)) != 0;

  void* ptr = useHost ? allocateHost(size) : allocateDevice(size);

  std::lock_guard<std::mutex> guard(mMutex);
  const uint64_t tag = (useStack && !mTagStack.empty()) ? mTagStack.back() : 0;
  mAllocations.emplace(ptr, AllocationMeta{useHost ? AllocationSpace::Host : AllocationSpace::Device, tag, useStack});
  if (useStack) {
    mTaggedAllocations[tag].push_back(ptr);
  }

  return ptr;
}

void GPUExternalAllocator::deallocate(char* ptr, size_t)
{
  if (!ptr) {
    return;
  }

  AllocationMeta meta;
  {
    std::lock_guard<std::mutex> guard(mMutex);
    const auto found = mAllocations.find(ptr);
    if (found == mAllocations.end()) {
      return;
    }
    meta = found->second;
    mAllocations.erase(found);
    if (meta.stacked) {
      removeFromTagLocked(meta.tag, ptr);
    }
  }

  freeAllocation(ptr, meta.space);
}

void GPUExternalAllocator::pushTagOnStack(uint64_t tag)
{
  std::lock_guard<std::mutex> guard(mMutex);
  mTagStack.push_back(tag);
}

void GPUExternalAllocator::popTagOffStack(uint64_t tag)
{
  std::vector<std::pair<void*, AllocationSpace>> toFree;
  {
    std::lock_guard<std::mutex> guard(mMutex);
    if (mTagStack.empty() || mTagStack.back() != tag) {
      throw std::runtime_error("GPUExternalAllocator tag stack mismatch");
    }

    const auto tagged = mTaggedAllocations.find(tag);
    if (tagged != mTaggedAllocations.end()) {
      toFree.reserve(tagged->second.size());
      for (void* ptr : tagged->second) {
        const auto found = mAllocations.find(ptr);
        if (found != mAllocations.end()) {
          toFree.emplace_back(ptr, found->second.space);
          mAllocations.erase(found);
        }
      }
      mTaggedAllocations.erase(tagged);
    }

    mTagStack.pop_back();
  }

  for (const auto& [ptr, space] : toFree) {
    freeAllocation(ptr, space);
  }
}

void GPUExternalAllocator::releaseAll()
{
  std::vector<std::pair<void*, AllocationSpace>> toFree;
  {
    std::lock_guard<std::mutex> guard(mMutex);
    toFree.reserve(mAllocations.size());
    for (const auto& [ptr, meta] : mAllocations) {
      toFree.emplace_back(ptr, meta.space);
    }
    mAllocations.clear();
    mTaggedAllocations.clear();
    mTagStack.clear();
  }

  for (const auto& [ptr, space] : toFree) {
    freeAllocation(ptr, space);
  }
}

void* GPUExternalAllocator::allocateHost(size_t size)
{
  void* ptr = nullptr;
  checkGpuError(cudaHostAlloc(&ptr, size, cudaHostAllocPortable), "cudaHostAlloc");
  return ptr;
}

void* GPUExternalAllocator::allocateDevice(size_t size)
{
  void* ptr = nullptr;
  checkGpuError(cudaMalloc(&ptr, size), "cudaMalloc");
  return ptr;
}

void GPUExternalAllocator::freeAllocation(void* ptr, AllocationSpace space)
{
  if (!ptr) {
    return;
  }

  if (space == AllocationSpace::Host) {
    checkGpuError(cudaFreeHost(ptr), "cudaFreeHost");
  } else {
    checkGpuError(cudaFree(ptr), "cudaFree");
  }
}

void GPUExternalAllocator::removeFromTagLocked(uint64_t tag, void* ptr)
{
  const auto tagged = mTaggedAllocations.find(tag);
  if (tagged == mTaggedAllocations.end()) {
    return;
  }

  auto& entries = tagged->second;
  entries.erase(std::remove(entries.begin(), entries.end(), ptr), entries.end());
  if (entries.empty()) {
    mTaggedAllocations.erase(tagged);
  }
}

} // namespace o2::trk
