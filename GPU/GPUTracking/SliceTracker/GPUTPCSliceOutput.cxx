// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceOutput.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUOutputControl.h"
#include "GPUTPCSliceOutput.h"
#include "GPUCommonMath.h"
#include <atomic>

using namespace GPUCA_NAMESPACE::gpu;

unsigned int GPUTPCSliceOutput::EstimateSize(unsigned int nOfTracks, unsigned int nOfTrackClusters)
{
  // calculate the amount of memory [bytes] needed for the event
  return sizeof(GPUTPCSliceOutput) + sizeof(GPUTPCTrack) * nOfTracks + sizeof(GPUTPCSliceOutCluster) * nOfTrackClusters;
}

#ifndef GPUCA_GPUCODE
void GPUTPCSliceOutput::Allocate(GPUTPCSliceOutput*& ptrOutput, int nTracks, int nTrackHits, GPUOutputControl* outputControl, void*& internalMemory)
{
  // Allocate All memory needed for slice output
  const size_t memsize = EstimateSize(nTracks, nTrackHits);

  if (outputControl && outputControl->useExternal()) {
    static std::atomic_flag lock = ATOMIC_FLAG_INIT;
    while (lock.test_and_set(std::memory_order_acquire)) {
    }
    outputControl->checkCurrent();
    if (outputControl->size - ((char*)outputControl->ptrCurrent - (char*)outputControl->ptrBase) < memsize) {
      outputControl->size = 1;
      ptrOutput = nullptr;
      lock.clear(std::memory_order_release);
      return;
    }
    ptrOutput = reinterpret_cast<GPUTPCSliceOutput*>(outputControl->ptrCurrent);
    outputControl->ptrCurrent = (char*)outputControl->ptrCurrent + memsize;
    lock.clear(std::memory_order_release);
  } else {
    if (internalMemory) {
      free(internalMemory);
    }
    internalMemory = malloc(memsize);
    ptrOutput = reinterpret_cast<GPUTPCSliceOutput*>(internalMemory);
  }
  ptrOutput->SetMemorySize(memsize);
}
#endif
