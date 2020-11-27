// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUITSFitter.h
/// \author David Rohr, Maximiliano Puccio

#ifndef GPUITSFITTER_H
#define GPUITSFITTER_H

#include "GPUProcessor.h"
#include "GPUITSTrack.h"

namespace o2::its
{
class Road;
struct TrackingFrameInfo;
struct Cluster;
class Cell;
} // namespace o2::its

namespace GPUCA_NAMESPACE::gpu
{
class GPUITSTrack;

class GPUITSFitter : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersInput(void* mem);
  void* SetPointersTracks(void* mem);
  void* SetPointersMemory(void* mem);
#endif

  GPUd() o2::its::Road* roads()
  {
    return mRoads;
  }
  GPUd() void SetNumberOfRoads(int v) { mNumberOfRoads = v; }
  GPUd() int NumberOfRoads() { return mNumberOfRoads; }
  GPUd() GPUITSTrack* tracks()
  {
    return mTracks;
  }
  GPUd() GPUAtomic(unsigned int) & NumberOfTracks()
  {
    return mMemory->mNumberOfTracks;
  }
  GPUd() void SetNumberOfLayers(int i) { mNumberOfLayers = i; }
  GPUd() int NumberOfLayers() { return mNumberOfLayers; }
  GPUd() void SetNumberTF(int i, int v) { mNTF[i] = v; }
  GPUd() o2::its::TrackingFrameInfo** trackingFrame()
  {
    return mTF;
  }
  GPUd() const o2::its::Cluster** clusters()
  {
    return mClusterPtrs;
  }
  GPUd() const o2::its::Cell** cells()
  {
    return mCellPtrs;
  }

  void clearMemory();

  struct Memory {
    GPUAtomic(unsigned int) mNumberOfTracks = 0;
  };

 protected:
  int mNumberOfLayers;
  int mNumberOfRoads = 0;
  int mNMaxTracks = 0;
  int* mNTF = nullptr;
  Memory* mMemory = nullptr;
  o2::its::Road* mRoads = nullptr;
  o2::its::TrackingFrameInfo** mTF = {nullptr};
  GPUITSTrack* mTracks = nullptr;

  const o2::its::Cluster** mClusterPtrs;
  const o2::its::Cell** mCellPtrs;

  short mMemoryResInput = -1;
  short mMemoryResTracks = -1;
  short mMemoryResMemory = -1;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
