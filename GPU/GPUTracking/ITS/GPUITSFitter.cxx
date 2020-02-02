// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUITSFitter.cxx
/// \author David Rohr, Maximiliano Puccio

#include "GPUITSFitter.h"

#include "ITStracking/Road.h"
#include "ITStracking/Cluster.h"
#include "GPUITSTrack.h"
#include "GPUReconstruction.h"

using namespace GPUCA_NAMESPACE::gpu;

#ifndef GPUCA_GPUCODE
void GPUITSFitter::InitializeProcessor()
{
}

void* GPUITSFitter::SetPointersInput(void* mem)
{
  computePointerWithAlignment(mem, mRoads, mNumberOfRoads);
  for (int i = 0; i < 7; i++) {
    computePointerWithAlignment(mem, mTF[i], mNTF[i]);
  }
  return mem;
}

void* GPUITSFitter::SetPointersTracks(void* mem)
{
  computePointerWithAlignment(mem, mTracks, mNMaxTracks);
  return mem;
}

void* GPUITSFitter::SetPointersMemory(void* mem)
{
  computePointerWithAlignment(mem, mMemory, 1);
  return mem;
}

void GPUITSFitter::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mMemoryResInput = mRec->RegisterMemoryAllocation(this, &GPUITSFitter::SetPointersInput, GPUMemoryResource::MEMORY_INPUT, "ITSInput");
  mMemoryResTracks = mRec->RegisterMemoryAllocation(this, &GPUITSFitter::SetPointersTracks, GPUMemoryResource::MEMORY_OUTPUT, "ITSTracks");
  mMemoryResMemory = mRec->RegisterMemoryAllocation(this, &GPUITSFitter::SetPointersMemory, GPUMemoryResource::MEMORY_PERMANENT, "ITSMemory");
}

void GPUITSFitter::SetMaxData(const GPUTrackingInOutPointers& io) { mNMaxTracks = mNumberOfRoads; }
#endif

void GPUITSFitter::clearMemory()
{
  new (mMemory) Memory;
}
