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

#ifndef ALICEO2_ALICE3GLOBALRECONSTRUCTION_GPUEXTERNALALLOCATOR_H
#define ALICEO2_ALICE3GLOBALRECONSTRUCTION_GPUEXTERNALALLOCATOR_H

#include "ITStracking/ExternalAllocator.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace o2::trk
{

class GPUExternalAllocator final : public o2::its::ExternalAllocator
{
 public:
  GPUExternalAllocator() = default;
  ~GPUExternalAllocator();

  void* allocate(size_t size) override;
  void deallocate(char* ptr, size_t size) override;
  void pushTagOnStack(uint64_t tag) override;
  void popTagOffStack(uint64_t tag) override;

  void releaseAll();

 private:
  enum class AllocationSpace { Host,
                               Device };

  struct AllocationMeta {
    AllocationSpace space;
    uint64_t tag;
    bool stacked;
  };

  using MemoryType = std::underlying_type_t<o2::gpu::GPUMemoryResource::MemoryType>;

  void* allocateHost(size_t size);
  void* allocateDevice(size_t size);
  void freeAllocation(void* ptr, AllocationSpace space);
  void removeFromTagLocked(uint64_t tag, void* ptr);

  std::mutex mMutex;
  std::vector<uint64_t> mTagStack;
  std::unordered_map<uint64_t, std::vector<void*>> mTaggedAllocations;
  std::unordered_map<void*, AllocationMeta> mAllocations;
};

} // namespace o2::trk

#endif
