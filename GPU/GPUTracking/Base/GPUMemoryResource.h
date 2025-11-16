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

/// \file GPUMemoryResource.h
/// \author David Rohr

#ifndef GPUMEMORYRESOURCE_H
#define GPUMEMORYRESOURCE_H

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstddef>
#endif

namespace o2::gpu
{

class GPUProcessor;
struct GPUMemoryReuse {
  enum Type : int32_t {
    NONE = 0,
    REUSE_1TO1 = 1
  };
  enum Group : uint16_t {
    ClustererScratch,
    NNClusterer,
    ClustererZS,
    TrackerScratch,
    TrackerDataLinks,
    TrackerDataWeights
  };
  using ID = uint32_t;

  GPUMemoryReuse(Type t, Group g, uint16_t i) : type(t), id(((uint32_t)g << 16) | ((uint32_t)i & 0xFFFF)) {}
  GPUMemoryReuse(bool condition, Type t, Group g, uint16_t i) : GPUMemoryReuse()
  {
    if (condition) {
      *this = GPUMemoryReuse{t, g, i};
    }
  }
  constexpr GPUMemoryReuse() = default;

  Type type = NONE;
  ID id = 0;
};

class GPUMemoryResource
{
  friend class GPUReconstruction;
  friend class GPUReconstructionCPU;

 public:
  enum MemoryType {
    MEMORY_HOST = 1,              // Memory allocated on host (irrespective of other flags)
    MEMORY_GPU = 2,               // Memory allocated on GPU (irrespective of other flags)
    MEMORY_INPUT_FLAG = 4,        // Flag to signal this memory is copied to GPU with TransferMemoryResourcesToGPU, and alike
    MEMORY_INPUT = 7,             // Input data for GPU has the MEMORY_INPUT_FLAG flat and is allocated on host and GPU
    MEMORY_OUTPUT_FLAG = 8,       // Flag to signal this memory is copied to Host with TransferMemoryResourcesToHost, and alike
    MEMORY_OUTPUT = 11,           // Output data for GPU has the MEMORY_OUTPUT_FLAG flat and is allocated on host and GPU
    MEMORY_INOUT = 15,            // Combination if MEMORY_INPUT and MEMORY_OUTPUT
    MEMORY_SCRATCH = 16,          // Scratch memory, is allocated only on GPU by default if running on GPU, only on host otherwise, if MEMORY_HOST and MEMORY_GPU flags not set.
    MEMORY_SCRATCH_HOST = 17,     // Scratch memory only on host
    MEMORY_EXTERNAL = 32,         // Special flag to signal that memory on host shall not be allocated, but will be provided externally and manually
    MEMORY_PERMANENT = 64,        // Permanent memory, registered once with AllocateRegisteredPermanentMemory, not per time frame. Only for small sizes!
    MEMORY_CUSTOM = 128,          // Memory is not allocated automatically with AllocateRegisteredMemory(GPUProcessor), but must be allocated manually via AllocateRegisteredMemory(memoryId)
    MEMORY_CUSTOM_TRANSFER = 256, // Memory is not transfered automatically with TransferMemoryResourcesTo, but must be transferred manually with TransferMemoryTo...(memoryId)
    MEMORY_STACK = 512            // Use memory from non-persistent stack at the end of the global memory region. Not persistent for full TF. Use PushNonPersistentMemory and PopNonPersistentMemory to release memory from the stack
  };
  enum AllocationType { ALLOCATION_AUTO = 0,       // --> GLOBAL if GPU is used, INDIVIDUAL otherwise
                        ALLOCATION_INDIVIDUAL = 1, // Individual memory allocations with malloc (host only)
                        ALLOCATION_GLOBAL = 2 };   // Allocate memory blocks from large preallocated memory range with internal allocator (host and GPU)

  GPUMemoryResource(GPUProcessor* proc, void* (GPUProcessor::*setPtr)(void*), MemoryType type, const char* name = "") : mProcessor(proc), mPtr(nullptr), mPtrDevice(nullptr), mSetPointers(setPtr), mName(name), mSize(0), mOverrideSize(0), mReuse(-1), mType(type)
  {
  }
  GPUMemoryResource(const GPUMemoryResource&) = default;

  void* SetPointers(void* ptr) const;
  void* SetDevicePointers(void* ptr) const;
  void* Ptr() const { return mPtr; }
  void* PtrDevice() const { return mPtrDevice; }
  size_t Size() const { return mSize; }
  const char* Name() const { return mName; }
  MemoryType Type() const { return mType; }

 private:
  GPUProcessor* mProcessor;
  void* mPtr;
  void* mPtrDevice;
  void* (GPUProcessor::*mSetPointers)(void*);
  const char* mName;
  size_t mSize;
  size_t mOverrideSize;
  int32_t mReuse;
  MemoryType mType;
};
} // namespace o2::gpu

#endif
