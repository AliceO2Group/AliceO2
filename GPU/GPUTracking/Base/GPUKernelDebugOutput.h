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

/// \file GPUGeneralKernels.h
/// \author David Rohr

#ifndef O2_GPU_KERNELDEBUGOUTPUT_H
#define O2_GPU_KERNELDEBUGOUTPUT_H

#include "GPUDef.h"
#include "GPUProcessor.h"
#ifdef GPUCA_KERNEL_DEBUGGER_OUTPUT

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUKernelDebugOutput : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersMemory(void* mem);

  void Print()
  {
    printf("------ Kernel Debug Output\n");
    for (int32_t i = 0; i < 100 * 1024; i++) {
      int32_t* pos = mDebugOutMemory + i * 1024;
      int32_t count = *(pos++);
      if (count) {
        printf("Thread %d: ", i);
        for (int32_t j = 0; j < count; j++) {
          printf("%d, ", pos[j]);
        }
        printf("\n");
      }
    }
    printf("------ End of Kernel Debug Output\n");
  }
#endif
  GPUdi() int32_t* memory()
  {
    return mDebugOutMemory;
  }
  GPUdi() static size_t memorySize() { return 100 * 1024 * 1024; }

  GPUd() void Add(uint32_t id, int32_t val) const
  {
    printf("Filling debug: id %d, val %d, current count %d\n", id, val, *(mDebugOutMemory + id * 1024));
    if (id > 100 * 1024) {
      return;
    }
    int32_t* pos = mDebugOutMemory + id * 1024;
    if (*pos >= 1023) {
      return;
    }
    pos += ++(*pos);
    *pos = val;
  }

 private:
  mutable int32_t* mDebugOutMemory;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif
