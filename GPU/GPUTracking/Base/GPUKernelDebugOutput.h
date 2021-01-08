// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
    for (int i = 0; i < 100 * 1024; i++) {
      int* pos = mDebugOutMemory + i * 1024;
      int count = *(pos++);
      if (count) {
        printf("Thread %d: ", i);
        for (int j = 0; j < count; j++) {
          printf("%d, ", pos[j]);
        }
        printf("\n");
      }
    }
    printf("------ End of Kernel Debug Output\n");
  }
#endif
  GPUdi() int* memory()
  {
    return mDebugOutMemory;
  }
  GPUdi() static size_t memorySize() { return 100 * 1024 * 1024; }

  GPUd() void Add(unsigned int id, int val) const
  {
    printf("Filling debug: id %d, val %d, current count %d\n", id, val, *(mDebugOutMemory + id * 1024));
    if (id > 100 * 1024) {
      return;
    }
    int* pos = mDebugOutMemory + id * 1024;
    if (*pos >= 1023) {
      return;
    }
    pos += ++(*pos);
    *pos = val;
  }

 private:
  mutable int* mDebugOutMemory;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif
