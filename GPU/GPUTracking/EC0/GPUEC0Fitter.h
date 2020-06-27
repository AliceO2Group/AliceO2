// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUEC0Fitter.h
/// \author David Rohr, Maximiliano Puccio

#ifndef GPUEC0FITTER_H
#define GPUEC0FITTER_H

#include "GPUProcessor.h"
#include "GPUEC0Track.h"

namespace o2
{
namespace ecl
{
class Road;
struct TrackingFrameInfo;
struct Cluster;
class Cell;
} // namespace ecl
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUEC0Track;

class GPUEC0Fitter : public GPUProcessor
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

  GPUd() o2::ecl::Road* roads()
  {
    return mRoads;
  }
  GPUd() void SetNumberOfRoads(int v) { mNumberOfRoads = v; }
  GPUd() int NumberOfRoads() { return mNumberOfRoads; }
  GPUd() GPUEC0Track* tracks()
  {
    return mTracks;
  }
  GPUd() GPUAtomic(unsigned int) & NumberOfTracks()
  {
    return mMemory->mNumberOfTracks;
  }
  GPUd() void SetNumberTF(int i, int v) { mNTF[i] = v; }
  GPUd() o2::ecl::TrackingFrameInfo** trackingFrame()
  {
    return mTF;
  }
  GPUd() const o2::ecl::Cluster** clusters()
  {
    return mClusterPtrs;
  }
  GPUd() const o2::ecl::Cell** cells()
  {
    return mCellPtrs;
  }

  void clearMemory();

  struct Memory {
    GPUAtomic(unsigned int) mNumberOfTracks = 0;
  };

 protected:
  int mNumberOfRoads = 0;
  int mNMaxTracks = 0;
  int mNTF[7] = {};
  Memory* mMemory = nullptr;
  o2::ecl::Road* mRoads = nullptr;
  o2::ecl::TrackingFrameInfo* mTF[7] = {};
  GPUEC0Track* mTracks = nullptr;

  const o2::ecl::Cluster* mClusterPtrs[7];
  const o2::ecl::Cell* mCellPtrs[5];

  short mMemoryResInput = -1;
  short mMemoryResTracks = -1;
  short mMemoryResMemory = -1;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
