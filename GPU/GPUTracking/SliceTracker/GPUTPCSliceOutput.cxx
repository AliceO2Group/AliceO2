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

using namespace GPUCA_NAMESPACE::gpu;

unsigned int GPUTPCSliceOutput::EstimateSize(unsigned int nOfTracks, unsigned int nOfTrackClusters)
{
  // calculate the amount of memory [bytes] needed for the event
  return sizeof(GPUTPCSliceOutput) + sizeof(GPUTPCSliceOutTrack) * nOfTracks + sizeof(GPUTPCSliceOutCluster) * nOfTrackClusters;
}

#ifndef GPUCA_GPUCODE
void GPUTPCSliceOutput::Allocate(GPUTPCSliceOutput*& ptrOutput, int nTracks, int nTrackHits, GPUOutputControl* outputControl, void*& internalMemory)
{
  // Allocate All memory needed for slice output
  const size_t memsize = EstimateSize(nTracks, nTrackHits);

  if (outputControl && outputControl->OutputType != GPUOutputControl::AllocateInternal) {
    if (outputControl->OutputMaxSize - ((char*)outputControl->OutputPtr - (char*)outputControl->OutputBase) < memsize) {
      outputControl->EndOfSpace = 1;
      ptrOutput = nullptr;
      return;
    }
    ptrOutput = reinterpret_cast<GPUTPCSliceOutput*>(outputControl->OutputPtr);
    outputControl->OutputPtr = (char*)outputControl->OutputPtr + memsize;
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
