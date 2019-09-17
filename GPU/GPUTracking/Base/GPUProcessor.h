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

  GPUd() GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * GetConstantMem() const
  {
    return mConstantMem;
  }

  // The template is a trick to forward the member even though it is forward-declared, to avoid cyclic header dependency
  template <class X = const MEM_CONSTANT(GPUConstantMem)>
    GPUd() GPUconstantref() const MEM_CONSTANT(GPUParam) & Param() const
  {
    GPUconstantref() X* v = mConstantMem;
    return v->param;
  }

  const GPUReconstruction& GetRec() const { return *mRec; }

#ifndef __OPENCL__
  void InitGPUProcessor(GPUReconstruction* rec, ProcessorType type = PROCESSOR_TYPE_CPU, GPUProcessor* slaveProcessor = nullptr);
  void Clear();

  // Helpers for memory allocation
  CONSTEXPR static size_t MIN_ALIGNMENT = 64;

  template <size_t alignment = MIN_ALIGNMENT>
  static inline size_t getAlignment(size_t addr)
  {
    static_assert((alignment & (alignment - 1)) == 0, "Invalid alignment, not power of 2");
    if (alignment <= 1) {
      return 0;
    }
    size_t mod = addr & (alignment - 1);
    if (mod == 0) {
      return 0;
    }
    return (alignment - mod);
  }
  template <size_t alignment = MIN_ALIGNMENT>
  static inline size_t nextMultipleOf(size_t size)
  {
    return size + getAlignment<alignment>(size);
  }
  template <size_t alignment = MIN_ALIGNMENT>
  static inline void* alignPointer(void* ptr)
  {
    return (reinterpret_cast<void*>(nextMultipleOf<alignment>(reinterpret_cast<size_t>(ptr))));
  }
  template <size_t alignment = MIN_ALIGNMENT>
  static inline size_t getAlignment(void* addr)
  {
    return (getAlignment<alignment>(reinterpret_cast<size_t>(addr)));
  }
  template <size_t alignment = MIN_ALIGNMENT, class S>
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
  template <size_t alignment = MIN_ALIGNMENT, class S>
  static inline S* getPointerWithAlignment(void*& basePtr, size_t nEntries = 1)
  {
    return getPointerWithAlignment<alignment, S>(reinterpret_cast<size_t&>(basePtr), nEntries);
  }

  template <size_t alignment = MIN_ALIGNMENT, class T, class S>
  static inline void computePointerWithAlignment(T*& basePtr, S*& objPtr, size_t nEntries = 1, bool runConstructor = false)
  {
    objPtr = getPointerWithAlignment<alignment, S>(reinterpret_cast<size_t&>(basePtr), nEntries);
    if (runConstructor) {
      for (size_t i = 0; i < nEntries; i++) {
        new (objPtr + i) S;
      }
    }
  }
#endif

 protected:
  void AllocateAndInitializeLate() { mAllocateAndInitializeLate = true; }

  GPUReconstruction* mRec;
  ProcessorType mGPUProcessorType;
  GPUProcessor* mDeviceProcessor;
  GPUconstantref() const MEM_CONSTANT(GPUConstantMem) * mConstantMem;

 private:
  bool mAllocateAndInitializeLate;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
