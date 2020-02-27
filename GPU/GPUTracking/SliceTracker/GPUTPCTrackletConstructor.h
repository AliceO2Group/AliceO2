// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
    GPUTPCThreadMemory() : mItr(0), mFirstRow(0), mLastRow(0), mStartRow(0), mEndRow(0), mCurrIH(0), mGo(0), mStage(0), mNHits(0), mNHitsEndRow(0), mNMissed(0), mLastY(0), mLastZ(0)
    {
    }

    GPUTPCThreadMemory(const GPUTPCThreadMemory& /*dummy*/) : mItr(0), mFirstRow(0), mLastRow(0), mStartRow(0), mEndRow(0), mCurrIH(0), mGo(0), mStage(0), mNHits(0), mNHitsEndRow(0), mNMissed(0), mLastY(0), mLastZ(0) {}
    GPUTPCThreadMemory& operator=(const GPUTPCThreadMemory& /*dummy*/) { return *this; }
#endif //! GPUCA_GPUCODE

   protected:
    // WARNING: This data is copied element by element in CopyTrackletTempData. Changes to members of this class must be reflected in CopyTrackletTempData!!!
    int mItr;         // track index
    int mFirstRow;    // first row index
    int mLastRow;     // last row index
    int mStartRow;    // first row index
    int mEndRow;      // first row index
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
    int mNextTrackletFirst;                                      // First tracklet to be processed by CUDA block during next iteration
    int mNextTrackletCount;                                      // Number of Tracklets to be processed by CUDA block during next iteration
    int mNextTrackletFirstRun;                                   // First run for dynamic scheduler?
    int mNTracklets;                                             // Total number of tracklets

#ifdef GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
    int fMaxSync; // temporary shared variable during profile creation
#endif            // GPUCA_TRACKLET_CONSTRUCTOR_DO_PROFILE
  };

  MEM_CLASS_PRE2()
  GPUd() static void InitTracklet(MEM_LG2(GPUTPCTrackParam) & tParam);

  MEM_CLASS_PRE2()
  GPUd() static void UpdateTracklet(int nBlocks, int nThreads, int iBlock, int iThread, MEM_LOCAL(GPUsharedref() GPUSharedMemory) & s, GPUTPCThreadMemory& r, GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, MEM_LG2(GPUTPCTrackParam) & tParam, int iRow);

  MEM_CLASS_PRE23()
  GPUd() static void StoreTracklet(int nBlocks, int nThreads, int iBlock, int iThread, MEM_LOCAL(GPUsharedref() GPUSharedMemory) & s, GPUTPCThreadMemory& r, GPUconstantref() MEM_LG2(GPUTPCTracker) & tracker, MEM_LG3(GPUTPCTrackParam) & tParam);

  MEM_CLASS_PRE2()
  GPUd() static bool CheckCov(MEM_LG2(GPUTPCTrackParam) & tParam);

  GPUd() static void DoTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, GPUsharedref() GPUTPCTrackletConstructor::MEM_LOCAL(GPUSharedMemory) & sMem, GPUTPCThreadMemory& rMem);

#ifdef GPUCA_GPUCODE
  GPUd() static int FetchTracklet(GPUconstantref() MEM_GLOBAL(GPUTPCTracker) & tracker, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & sMem);
#else
  static int GPUTPCTrackletConstructorGlobalTracking(GPUTPCTracker& tracker, GPUTPCTrackParam& tParam, int startrow, int increment, int iTracklet);
#endif // GPUCA_GPUCODE

  typedef GPUconstantref() MEM_GLOBAL(GPUTPCTracker) processorType;
  GPUhdi() CONSTEXPRRET static GPUDataTypes::RecoStep GetRecoStep() { return GPUCA_RECO_STEP::TPCSliceTracking; }
  MEM_TEMPLATE()
  GPUhdi() static processorType* Processor(MEM_TYPE(GPUConstantMem) & processors)
  {
    return processors.tpcTrackers;
  }
  template <int iKernel = GPUKernelTemplate::defaultKernel>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& tracker);
};

class GPUTPCGlobalTracking
{
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCTRACKLETCONSTRUCTOR_H
