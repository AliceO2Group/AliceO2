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

/// \file GPUTPCTrackletConstructor.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCTRACKLETCONSTRUCTOR_H
#define GPUTPCTRACKLETCONSTRUCTOR_H

#include "GPUTPCDef.h"
#include "GPUTPCTrackParam.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCTrackletConstructor
 *
 */
MEM_CLASS_PRE()
class GPUTPCTracker;

class GPUTPCTrackletConstructor
{
 public:
  enum K {
    singleSlice = 0,
    allSlices = 1
  };

  class GPUTPCThreadMemory
  {
    friend class GPUTPCTrackletConstructor; //! friend class
   public:
#if !defined(GPUCA_GPUCODE)
    GPUTPCThreadMemory() : mISH(0), mFirstRow(0), mLastRow(0), mStartRow(0), mEndRow(0), mCurrIH(0), mGo(0), mStage(0), mNHits(0), mNHitsEndRow(0), mNMissed(0), mLastY(0), mLastZ(0)
    {
    }

    GPUTPCThreadMemory(const GPUTPCThreadMemory& /*dummy*/) : mISH(0), mFirstRow(0), mLastRow(0), mStartRow(0), mEndRow(0), mCurrIH(0), mGo(0), mStage(0), mNHits(0), mNHitsEndRow(0), mNMissed(0), mLastY(0), mLastZ(0) {}
    GPUTPCThreadMemory& operator=(const GPUTPCThreadMemory& /*dummy*/) { return *this; }
#endif //! GPUCA_GPUCODE

   protected:
    // WARNING: This data is copied element by element in CopyTrackletTempData. Changes to members of this class must be reflected in CopyTrackletTempData!!!
    int mISH;         // track index
    int mFirstRow;    // first row index
    int mLastRow;     // last row index
    int mStartRow;    // row index of first hit in seed
    int mEndRow;      // row index of last hit in seed
    calink mCurrIH;   // indef of the current hit
    char mGo;         // do fit/searching flag
    int mStage;       // reco stage
    int mNHits;       // n track hits
    int mNHitsEndRow; // n hits at end row
    int mNMissed;     // n missed hits during search
    float mLastY;     // Y of the last fitted cluster
    float mLastZ;     // Z of the last fitted cluster
  };

  MEM_CLASS_PRE()
  struct GPUSharedMemory {
    CA_SHARED_STORAGE(MEM_LG(GPUTPCRow) mRows[GPUCA_ROW_COUNT]); // rows
    int mNextStartHitFirst;                                      // First start hit to be processed by CUDA block during next iteration
    int mNextStartHitCount;                                      // Number of start hits to be processed by CUDA block during next iteration
    int mNextStartHitFirstRun;                                   // First run for dynamic scheduler?
    int mNStartHits;                                             // Total number of start hits

#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
    int fMaxSync; // temporary shared variable during profile creation
#endif            // GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
  };

  MEM_CLASS_PRE2()
  GPUd() static void InitTracklet(MEM_LG2(GPUTPCTrackParam) & tParam);

  MEM_CLASS_PRE2_TEMPLATE(class T)
  GPUd() static void UpdateTracklet(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() T& s, GPUTPCThreadMemory& r, GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, MEM_LG2(GPUTPCTrackParam) & tParam, int iRow, calink& rowHit, calink* rowHits);

  MEM_CLASS_PRE23()
  GPUd() static void StoreTracklet(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & s, GPUTPCThreadMemory& r, GPUconstantref() MEM_LG2(GPUTPCTracker) & tracker, MEM_LG3(GPUTPCTrackParam) & tParam, calink* rowHits);

  MEM_CLASS_PRE2()
  GPUd() static bool CheckCov(MEM_LG2(GPUTPCTrackParam) & tParam);

  GPUd() static void DoTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, GPUsharedref() GPUTPCTrackletConstructor::MEM_LOCAL(GPUSharedMemory) & sMem, GPUTPCThreadMemory& rMem);

#ifdef GPUCA_GPUCODE
  GPUd() static int FetchTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & sMem);
#endif // GPUCA_GPUCODE

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  template <class T>
  GPUd() static int GPUTPCTrackletConstructorGlobalTracking(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, GPUsharedref() T& sMem, GPUTPCTrackParam& tParam, int startrow, int increment, int iTracklet, calink* rowHits);
#endif

  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  MEM_TEMPLATE()
  GPUhdi() static processorType* Processor(MEM_TYPE(GPUConstantMem) & processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& tracker);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKLETCONSTRUCTOR_H
