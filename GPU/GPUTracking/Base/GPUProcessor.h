// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUProcessor.h
/// \author David Rohr

#ifndef GPUPROCESSOR_H
#define GPUPROCESSOR_H

#include "GPUCommonDef.h"
#include "GPUDef.h"

#ifndef GPUCA_GPUCODE
#include <cstddef>
#include <algorithm>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTrackingInOutPointers;
class GPUReconstruction;
MEM_CLASS_PRE()
struct GPUParam;
MEM_CLASS_PRE()
struct GPUConstantMem;

class GPUProcessor
{
  friend class GPUReconstruction;
  friend class GPUReconstructionCPU;
  friend class GPUMemoryResource;

 public:
  enum ProcessorType { PROCESSOR_TYPE_CPU = 0,
                       PROCESSOR_TYPE_DEVICE = 1,
                       PROCESSOR_TYPE_SLAVE = 2 };

#ifndef GPUCA_GPUCODE
  GPUProcessor();
  ~GPUProcessor();
  GPUProcessor(const GPUProcessor&) CON_DELETE;
  GPUProcessor& operator=(const GPUProcessor&) CON_DELETE;
#endif

  GPUd() GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * GetConstantMem() const; // Body in GPUConstantMem.h to avoid circular headers
  GPUd() GPUconstantref() const MEM_CONSTANT(GPUParam) & Param() const;                // ...
  GPUd() void raiseError(unsigned int code, unsigned int param1 = 0, unsigned int param2 = 0, unsigned int param3 = 0) const;
  const GPUReconstruction& GetRec() const { return *mRec; }

#ifndef __OPENCL__
  void InitGPUProcessor(GPUReconstruction* rec, ProcessorType type = PROCESSOR_TYPE_CPU, GPUProcessor* slaveProcessor = nullptr);
  void Clear();
  template <class T>
  T& HostProcessor(T*)
  {
    return *(T*)(mGPUProcessorType == PROCESSOR_TYPE_DEVICE ? mLinkedProcessor : this);
  }

  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT>
  static inline size_t getAlignmentMod(size_t addr)
  {
    static_assert((alignment & (alignment - 1)) == 0, "Invalid alignment, not power of 2");
    if (alignment <= 1) {
      return 0;
    }
    return addr & (alignment - 1);
  }
  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT>
  static inline size_t getAlignment(size_t addr)
  {
    size_t mod = getAlignmentMod<alignment>(addr);
    if (mod == 0) {
      return 0;
    }
    return (alignment - mod);
  }
  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT>
  static inline size_t nextMultipleOf(size_t size)
  {
    return size + getAlignment<alignment>(size);
  }
  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT>
  static inline void* alignPointer(void* ptr)
  {
    return (reinterpret_cast<void*>(nextMultipleOf<alignment>(reinterpret_cast<size_t>(ptr))));
  }
  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT>
  static inline size_t getAlignmentMod(void* addr)
  {
    return (getAlignmentMod<alignment>(reinterpret_cast<size_t>(addr)));
  }
  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT>
  static inline size_t getAlignment(void* addr)
  {
    return (getAlignment<alignment>(reinterpret_cast<size_t>(addr)));
  }
  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT, class S>
  static inline S* getPointerWithAlignment(size_t& basePtr, size_t nEntries = 1)
  {
    if (basePtr == 0) {
      basePtr = 1;
    }
    CONSTEXPR size_t maxAlign = (alignof(S) > alignment) ? alignof(S) : alignment;
    basePtr += getAlignment<maxAlign>(basePtr);
    S* retVal = (S*)(basePtr);
    basePtr += nEntries * sizeof(S);
    return retVal;
  }

  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT, class S>
  static inline S* getPointerWithAlignment(void*& basePtr, size_t nEntries = 1)
  {
    return getPointerWithAlignment<alignment, S>(reinterpret_cast<size_t&>(basePtr), nEntries);
  }

  template <size_t alignment = GPUCA_BUFFER_ALIGNMENT, class T, class S>
  static inline void computePointerWithAlignment(T*& basePtr, S*& objPtr, size_t nEntries = 1)
  {
    objPtr = getPointerWithAlignment<alignment, S>(reinterpret_cast<size_t&>(basePtr), nEntries);
  }

  template <class T, class S>
  static inline void computePointerWithoutAlignment(T*& basePtr, S*& objPtr, size_t nEntries = 1)
  {
    if ((size_t)basePtr < GPUCA_BUFFER_ALIGNMENT) {
      reinterpret_cast<size_t&>(basePtr) = GPUCA_BUFFER_ALIGNMENT;
    }
    objPtr = reinterpret_cast<S*>(getPointerWithAlignment<1, char>(reinterpret_cast<size_t&>(basePtr), nEntries * sizeof(S)));
  }
#endif

 protected:
  void AllocateAndInitializeLate() { mAllocateAndInitializeLate = true; }

  GPUReconstruction* mRec;
  ProcessorType mGPUProcessorType;
  GPUProcessor* mLinkedProcessor;
  GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mConstantMem;

 private:
  bool mAllocateAndInitializeLate;

  friend class GPUTPCNeighboursFinder;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
