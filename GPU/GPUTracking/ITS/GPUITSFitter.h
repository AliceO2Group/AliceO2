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

/// \file GPUITSFitter.h
/// \author David Rohr, Maximiliano Puccio

#ifndef GPUITSFITTER_H
#define GPUITSFITTER_H

#include "GPUProcessor.h"
#include "GPUITSTrack.h"

namespace o2::its
{
template <uint8_t N>
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

  GPUd() o2::its::Road<5>* roads()
  {
    return mRoads;
  }
  GPUd() void SetNumberOfRoads(int32_t v) { mNumberOfRoads = v; }
  GPUd() int32_t NumberOfRoads() { return mNumberOfRoads; }
  GPUd() GPUITSTrack* tracks()
  {
    return mTracks;
  }
  GPUd() GPUAtomic(uint32_t) & NumberOfTracks()
  {
    return mMemory->mNumberOfTracks;
  }
  GPUd() void SetNumberOfLayers(int32_t i) { mNumberOfLayers = i; }
  GPUd() int32_t NumberOfLayers() { return mNumberOfLayers; }
  GPUd() void SetNumberTF(int32_t i, int32_t v) { mNTF[i] = v; }
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
    GPUAtomic(uint32_t) mNumberOfTracks = 0;
  };

 protected:
  int32_t mNumberOfLayers;
  int32_t mNumberOfRoads = 0;
  int32_t mNMaxTracks = 0;
  int32_t* mNTF = nullptr;
  Memory* mMemory = nullptr;
  o2::its::Road<5>* mRoads = nullptr;
  o2::its::TrackingFrameInfo** mTF = {nullptr};
  GPUITSTrack* mTracks = nullptr;

  const o2::its::Cluster** mClusterPtrs;
  const o2::its::Cell** mCellPtrs;

  int16_t mMemoryResInput = -1;
  int16_t mMemoryResTracks = -1;
  int16_t mMemoryResMemory = -1;
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
