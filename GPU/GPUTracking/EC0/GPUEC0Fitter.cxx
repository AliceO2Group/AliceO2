// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUEC0Fitter.cxx
/// \author David Rohr, Maximiliano Puccio

#include "GPUEC0Fitter.h"

#include "EC0tracking/Road.h"
#include "EC0tracking/Cluster.h"
#include "GPUEC0Track.h"
#include "GPUReconstruction.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_GPUCODE
void GPUEC0Fitter::InitializeProcessor()
{
}

void* GPUEC0Fitter::SetPointersInput(void* mem)
{
  computePointerWithAlignment(mem, mRoads, mNumberOfRoads);
  for (int i = 0; i < 7; i++) {
    computePointerWithAlignment(mem, mTF[i], mNTF[i]);
  }
  return mem;
}

void* GPUEC0Fitter::SetPointersTracks(void* mem)
{
  computePointerWithAlignment(mem, mTracks, mNMaxTracks);
  return mem;
}

void* GPUEC0Fitter::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory, 1);
  return mem;
}

void GPUEC0Fitter::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mMemoryResInput = mRec->RegisterMemoryAllocation(this, &GPUEC0Fitter::SetPointersInput, GPUMemoryResource::MEMORY_INPUT, "EC0Input");
  mMemoryResTracks = mRec->RegisterMemoryAllocation(this, &GPUEC0Fitter::SetPointersTracks, GPUMemoryResource::MEMORY_OUTPUT, "EC0Tracks");
  mMemoryResMemory = mRec->RegisterMemoryAllocation(this, &GPUEC0Fitter::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "EC0Memory");
}

void GPUEC0Fitter::SetMaxData(const GPUTrackingInOutPointers& io) { mNMaxTracks = mNumberOfRoads; }
#endif

void GPUEC0Fitter::clearMemory()
{
  new (mMemory) Memory;
}
