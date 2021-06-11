// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUMemoryResource.h
/// \author David Rohr

#ifndef GPUMEMORYRESOURCE_H
#define GPUMEMORYRESOURCE_H

#include "GPUCommonDef.h"
#include "GPUProcessor.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#ifdef GPUCA_NOCOMPAT_ALLOPENCL
struct GPUMemoryReuse {
  enum Type : int {
    NONE = 0,
    REUSE_1TO1 = 1
  };
  enum Group : unsigned short {
    ClustererScratch,
    ClustererZS,
    TrackerScratch,
    TrackerDataLinks,
    TrackerDataWeights
  };
  using ID = unsigned int;

  GPUMemoryReuse(Type t, Group g, unsigned short i) : type(t), id(((unsigned int)g << 16) | ((unsigned int)i & 0xFFFF)) {}
  GPUMemoryReuse(bool condition, Type t, Group g, unsigned short i) : GPUMemoryReuse()
  {
    if (condition) {
      *this = GPUMemoryReuse{t, g, i};
    }
  }
  constexpr GPUMemoryReuse() = default;

  Type type = NONE;
  ID id = 0;
};
#endif

class GPUMemoryResource
{
  friend class GPUReconstruction;
  friend class GPUReconstructionCPU;

 public:
  enum MemoryType {
    MEMORY_HOST = 1,
    MEMORY_GPU = 2,
    MEMORY_INPUT_FLAG = 4,
    MEMORY_INPUT = 7,
    MEMORY_OUTPUT_FLAG = 8,
    MEMORY_OUTPUT = 11,
    MEMORY_INOUT = 15,
    MEMORY_SCRATCH = 16,
    MEMORY_SCRATCH_HOST = 17,
    MEMORY_EXTERNAL = 32,
    MEMORY_PERMANENT = 64,
    MEMORY_CUSTOM = 128,
    MEMORY_CUSTOM_TRANSFER = 256,
    MEMORY_STACK = 512
  };
  enum AllocationType { ALLOCATION_AUTO = 0,
                        ALLOCATION_INDIVIDUAL = 1,
                        ALLOCATION_GLOBAL = 2 };

#ifndef GPUCA_GPUCODE
  GPUMemoryResource(GPUProcessor* proc, void* (GPUProcessor::*setPtr)(void*), MemoryType type, const char* name = "") : mProcessor(proc), mPtr(nullptr), mPtrDevice(nullptr), mSetPointers(setPtr), mName(name), mSize(0), mOverrideSize(0), mReuse(-1), mType(type)
  {
  }
  GPUMemoryResource(const GPUMemoryResource&) CON_DEFAULT;
#endif

  void* SetPointers(void* ptr)
  {
    return (mProcessor->*mSetPointers)(ptr);
  }
  void* SetDevicePointers(void* ptr) { return (mProcessor->mLinkedProcessor->*mSetPointers)(ptr); }
  void* Ptr() { return mPtr; }
  void* PtrDevice() { return mPtrDevice; }
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
  int mReuse;
  MemoryType mType;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
