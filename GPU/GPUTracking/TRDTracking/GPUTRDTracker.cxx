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

/// \file GPUTRDTracker.cxx
/// \author Ole Schmidt

//#define ENABLE_GPUTRDDEBUG
#define ENABLE_WARNING 0
#define ENABLE_INFO 0
#ifdef GPUCA_ALIROOT_LIB
#define ENABLE_GPUMC
#endif

#include "GPUTRDTracker.h"
#include "GPUTRDTrackletWord.h"
#include "GPUTRDGeometry.h"
#include "GPUTRDTrackerDebug.h"
#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"

using namespace GPUCA_NAMESPACE::gpu;

class GPUTPCGMPolynomialField;

#ifndef GPUCA_GPUCODE
#include "GPUMemoryResource.h"
#include "GPUReconstruction.h"
#ifdef WITH_OPENMP
#include <omp.h>
#endif // WITH_OPENMP
#include <chrono>
#include <vector>
#ifdef GPUCA_ALIROOT_LIB
#include "TDatabasePDG.h"
#include "AliMCParticle.h"
#include "AliMCEvent.h"
#endif // GPUCA_ALIROOT_LIB

#include "GPUChainTracking.h"

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mNMaxTracks = std::max(std::max(io.nOutputTracksTPCO2, io.nTracksTPCITSO2), std::max(io.nMergedTracks, io.nOutputTracksTPCO2)); // TODO: This is a bit stupid, we should just take the correct number, not the max of all
  mNMaxSpacePoints = io.nTRDTracklets;
  mNMaxCollisions = io.nTRDTriggerRecords;
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
  mMemoryPermanent = mRec->RegisterMemoryAllocation(this, &GPUTRDTracker_t<TRDTRK, PROP>::SetPointersBase, GPUMemoryResource::MEMORY_PERMANENT, "TRDInitialize");
  mMemoryTracklets = mRec->RegisterMemoryAllocation(this, &GPUTRDTracker_t<TRDTRK, PROP>::SetPointersTracklets, GPUMemoryResource::MEMORY_INPUT, "TRDTracklets");
  mMemoryTracks = mRec->RegisterMemoryAllocation(this, &GPUTRDTracker_t<TRDTRK, PROP>::SetPointersTracks, GPUMemoryResource::MEMORY_INOUT, "TRDTracks");
}

template <class TRDTRK, class PROP>
void* GPUTRDTracker_t<TRDTRK, PROP>::SetPointersBase(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for fixed size objects (needs to be done only once)
  //--------------------------------------------------------------------
  mMaxThreads = mRec->GetMaxThreads();
  computePointerWithAlignment(base, mR, kNChambers);
  computePointerWithAlignment(base, mHypothesis, mNCandidates * mMaxThreads);
  computePointerWithAlignment(base, mCandidates, mNCandidates * 2 * mMaxThreads);
  return base;
}

template <class TRDTRK, class PROP>
void* GPUTRDTracker_t<TRDTRK, PROP>::SetPointersTracklets(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for tracklets and space points
  // (size might change for different events)
  //--------------------------------------------------------------------
  if (mGenerateSpacePoints) {
    computePointerWithAlignment(base, mSpacePoints, mNMaxSpacePoints);
  }
  computePointerWithAlignment(base, mTrackletIndexArray, (kNChambers + 1) * mNMaxCollisions);
  return base;
}

template <class TRDTRK, class PROP>
void* GPUTRDTracker_t<TRDTRK, PROP>::SetPointersTracks(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for tracks (this is done once per event)
  //--------------------------------------------------------------------
  computePointerWithAlignment(base, mTracks, mNMaxTracks);
  computePointerWithAlignment(base, mTrackAttribs, mNMaxTracks);
  return base;
}

template <class TRDTRK, class PROP>
GPUTRDTracker_t<TRDTRK, PROP>::GPUTRDTracker_t() : mR(nullptr), mIsInitialized(false), mGenerateSpacePoints(false), mProcessPerTimeFrame(false), mNAngleHistogramBins(25), mAngleHistogramRange(50), mMemoryPermanent(-1), mMemoryTracklets(-1), mMemoryTracks(-1), mNMaxCollisions(0), mNMaxTracks(0), mNMaxSpacePoints(0), mTracks(nullptr), mTrackAttribs(nullptr), mNCandidates(1), mNTracks(0), mNEvents(0), mMaxThreads(100), mTrackletIndexArray(nullptr), mHypothesis(nullptr), mCandidates(nullptr), mSpacePoints(nullptr), mGeo(nullptr), mRPhiA2(0), mRPhiB(0), mRPhiC2(0), mDyA2(0), mDyB(0), mDyC2(0), mAngleToDyA(0), mAngleToDyB(0), mAngleToDyC(0), mDebugOutput(false), mMaxEta(0.84f), mRoadZ(18.f), mZCorrCoefNRC(1.4f), mTPCVdrift(2.58f), mDebug(new GPUTRDTrackerDebug<TRDTRK>())
{
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
}

template <class TRDTRK, class PROP>
GPUTRDTracker_t<TRDTRK, PROP>::~GPUTRDTracker_t()
{
  //--------------------------------------------------------------------
  // Destructor
  //--------------------------------------------------------------------
  delete mDebug;
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::InitializeProcessor()
{
  //--------------------------------------------------------------------
  // Initialise tracker
  //--------------------------------------------------------------------
  mGeo = (TRD_GEOMETRY_CONST GPUTRDGeometry*)GetConstantMem()->calibObjects.trdGeometry;
  if (!mGeo) {
    GPUError("TRD geometry must be provided externally");
  }

  float Bz = Param().bzkG;
  float resRPhiIdeal2 = Param().rec.trd.trkltResRPhiIdeal * Param().rec.trd.trkltResRPhiIdeal;
  GPUInfo("Initializing with B-field: %f kG", Bz);
  if (CAMath::Abs(CAMath::Abs(Bz) - 2) < 0.1) {
    // magnetic field +-0.2 T
    if (Bz > 0) {
      GPUInfo("Loading error parameterization for Bz = +2 kG");
      mRPhiA2 = resRPhiIdeal2, mRPhiB = -1.43e-2f, mRPhiC2 = 4.55e-2f;
      mDyA2 = 1.225e-3f, mDyB = -9.8e-3f, mDyC2 = 3.88e-2f;
      mAngleToDyA = -0.1f, mAngleToDyB = 1.89f, mAngleToDyC = -0.4f;
    } else {
      GPUInfo("Loading error parameterization for Bz = -2 kG");
      mRPhiA2 = resRPhiIdeal2, mRPhiB = 1.43e-2f, mRPhiC2 = 4.55e-2f;
      mDyA2 = 1.225e-3f, mDyB = 9.8e-3f, mDyC2 = 3.88e-2f;
      mAngleToDyA = 0.1f, mAngleToDyB = 1.89f, mAngleToDyC = 0.4f;
    }
  } else if (CAMath::Abs(CAMath::Abs(Bz) - 5) < 0.1) {
    // magnetic field +-0.5 T
    if (Bz > 0) {
      GPUInfo("Loading error parameterization for Bz = +5 kG");
      mRPhiA2 = resRPhiIdeal2, mRPhiB = 0.125f, mRPhiC2 = 0.0961f;
      mDyA2 = 1.681e-3f, mDyB = 0.15f, mDyC2 = 0.1849f;
      mAngleToDyA = 0.13f, mAngleToDyB = 2.43f, mAngleToDyC = -0.58f;
    } else {
      GPUInfo("Loading error parameterization for Bz = -5 kG");
      mRPhiA2 = resRPhiIdeal2, mRPhiB = -0.14f, mRPhiC2 = 0.1156f;
      mDyA2 = 2.209e-3f, mDyB = -0.15f, mDyC2 = 0.2025f;
      mAngleToDyA = -0.15f, mAngleToDyB = 2.34f, mAngleToDyC = 0.56f;
    }
  } else {
    // magnetic field 0 T or another value which is not covered by the error parameterizations
    GPUError("No error parameterization available for Bz = %.2f kG", Bz);
  }

#ifdef GPUCA_ALIROOT_LIB
  for (int iCandidate = 0; iCandidate < mNCandidates * 2 * mMaxThreads; ++iCandidate) {
    new (&mCandidates[iCandidate]) TRDTRK;
  }
#endif

  // obtain average radius of TRD chambers
  float x0[kNLayers] = {300.2f, 312.8f, 325.4f, 338.0f, 350.6f, 363.2f}; // used as default value in case no transformation matrix can be obtained
  auto* matrix = mGeo->GetClusterMatrix(0);
  My_Float loc[3] = {mGeo->AnodePos(), 0.f, 0.f};
  My_Float glb[3] = {0.f, 0.f, 0.f};
  for (int iDet = 0; iDet < kNChambers; ++iDet) {
    matrix = mGeo->GetClusterMatrix(iDet);
    if (!matrix) {
      mR[iDet] = x0[mGeo->GetLayer(iDet)];
      continue;
    }
    matrix->LocalToMaster(loc, glb);
    mR[iDet] = glb[0];
  }

  mDebug->ExpandVectors();
  mIsInitialized = true;
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::Reset()
{
  //--------------------------------------------------------------------
  // Reset tracker
  //--------------------------------------------------------------------
  mNTracks = 0;
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::PrepareTracking(GPUChainTracking* chainTracking)
{
  //--------------------------------------------------------------------
  // Prepare tracklet index array and if requested calculate space points
  // in part duplicated from DoTracking() method to allow for calling
  // this function on the host prior to GPU processing
  //--------------------------------------------------------------------
  for (unsigned int iColl = 0; iColl < GetConstantMem()->ioPtrs.nTRDTriggerRecords; ++iColl) {
    if (GetConstantMem()->ioPtrs.trdTrigRecMask && GetConstantMem()->ioPtrs.trdTrigRecMask[iColl] == 0) {
      // this trigger is masked as there is no ITS information available for it
      continue;
    }
    int nTrklts = 0;
    int idxOffset = 0;
    if (mProcessPerTimeFrame) {
      idxOffset = GetConstantMem()->ioPtrs.trdTrackletIdxFirst[iColl];
      nTrklts = (iColl < GetConstantMem()->ioPtrs.nTRDTriggerRecords - 1) ? GetConstantMem()->ioPtrs.trdTrackletIdxFirst[iColl + 1] - GetConstantMem()->ioPtrs.trdTrackletIdxFirst[iColl] : GetConstantMem()->ioPtrs.nTRDTracklets - GetConstantMem()->ioPtrs.trdTrackletIdxFirst[iColl];
    } else {
      nTrklts = GetConstantMem()->ioPtrs.nTRDTracklets;
    }
    const GPUTRDTrackletWord* tracklets = &((GetConstantMem()->ioPtrs.trdTracklets)[idxOffset]);
    int* trkltIndexArray = &mTrackletIndexArray[iColl * (kNChambers + 1) + 1];
    trkltIndexArray[-1] = 0;
    int currDet = 0;
    int nextDet = 0;
    int trkltCounter = 0;
    for (int iTrklt = 0; iTrklt < nTrklts; ++iTrklt) {
      if (tracklets[iTrklt].GetDetector() > currDet) {
        nextDet = tracklets[iTrklt].GetDetector();
        for (int iDet = currDet; iDet < nextDet; ++iDet) {
          trkltIndexArray[iDet] = trkltCounter;
        }
        currDet = nextDet;
      }
      ++trkltCounter;
    }
    for (int iDet = currDet; iDet <= kNChambers; ++iDet) {
      trkltIndexArray[iDet] = trkltCounter;
    }
    if (mGenerateSpacePoints) {
      if (!CalculateSpacePoints(iColl)) {
        GPUError("Space points for at least one chamber could not be calculated (for interaction %i)", iColl);
        break;
      }
    }
  }
  if (mGenerateSpacePoints) {
    chainTracking->mIOPtrs.trdSpacePoints = mSpacePoints;
  }
  mNEvents++;
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::SetNCandidates(int n)
{
  //--------------------------------------------------------------------
  // set the number of candidates to be used
  //--------------------------------------------------------------------
  if (!mIsInitialized) {
    mNCandidates = n;
  } else {
    GPUError("Cannot change mNCandidates after initialization");
  }
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::PrintSettings() const
{
  //--------------------------------------------------------------------
  // print current settings to screen
  //--------------------------------------------------------------------
  GPUInfo("##############################################################");
  GPUInfo("Current settings for GPU TRD tracker:");
  GPUInfo(" maxChi2(%.2f), chi2Penalty(%.2f), nCandidates(%i), maxMissingLayers(%i)", Param().rec.trd.maxChi2, Param().rec.trd.penaltyChi2, mNCandidates, Param().rec.trd.stopTrkAfterNMissLy);
  GPUInfo(" ptCut = %.2f GeV, abs(eta) < %.2f", Param().rec.trd.minTrackPt, mMaxEta);
  GPUInfo("##############################################################");
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::StartDebugging()
{
  mDebug->CreateStreamer();
}



#endif //! GPUCA_GPUCODE

template <>
GPUdi() const GPUTRDPropagatorGPU::propagatorParam* GPUTRDTracker_t<GPUTRDTrackGPU, GPUTRDPropagatorGPU>::getPropagatorParam(bool externalDefaultO2Propagator)
{
  return &Param().polynomialField;
}

template <class TRDTRK, class PROP>
GPUdi() const typename PROP::propagatorParam* GPUTRDTracker_t<TRDTRK, PROP>::getPropagatorParam(bool externalDefaultO2Propagator)
{
#ifdef GPUCA_GPUCODE
  return GetConstantMem()->calibObjects.o2Propagator;
#elif defined GPUCA_ALIROOT_LIB
  return nullptr;
#else
  if (externalDefaultO2Propagator) {
    return o2::base::Propagator::Instance();
  } else {
    return GetConstantMem()->calibObjects.o2Propagator;
  }
#endif
}

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::CheckTrackTRDCandidate(const TRDTRK& trk) const
{
  if (!trk.CheckNumericalQuality()) {
    return false;
  }
  if (CAMath::Abs(trk.getEta()) > mMaxEta) {
    return false;
  }
  if (trk.getPt() < Param().rec.trd.minTrackPt) {
    return false;
  }
  return true;
}

template <class TRDTRK, class PROP>
GPUd() int GPUTRDTracker_t<TRDTRK, PROP>::LoadTrack(const TRDTRK& trk, unsigned int tpcTrackId, bool checkTrack, HelperTrackAttributes* attribs)
{
  if (mNTracks >= mNMaxTracks) {
#ifndef GPUCA_GPUCODE
    GPUError("Error: Track dropped (no memory available) -> must not happen");
#endif
    return (1);
  }
  if (checkTrack && !CheckTrackTRDCandidate(trk)) {
    return 2;
  }
#ifdef GPUCA_ALIROOT_LIB
  new (&mTracks[mNTracks]) TRDTRK(trk); // We need placement new, since the class is virtual
#else
  mTracks[mNTracks] = trk;
#endif
  mTracks[mNTracks].setRefGlobalTrackIdRaw(tpcTrackId);
  if (attribs) {
    mTrackAttribs[mNTracks] = *attribs;
  }
  mNTracks++;
  return (0);
}


template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::DumpTracks()
{
  //--------------------------------------------------------------------
  // helper function (only for debugging purposes)
  //--------------------------------------------------------------------
  GPUInfo("There are in total %i tracklets loaded", GetConstantMem()->ioPtrs.nTRDTracklets);
  GPUInfo("There are %i tracks loaded. mNMaxTracks(%i)", mNTracks, mNMaxTracks);
  for (int i = 0; i < mNTracks; ++i) {
    auto* trk = &(mTracks[i]);
    GPUInfo("track %i: x=%f, alpha=%f, nTracklets=%i, pt=%f, time=%f", i, trk->getX(), trk->getAlpha(), trk->getNtracklets(), trk->getPt(), mTrackAttribs[i].mTime);
  }
}

template <class TRDTRK, class PROP>
GPUd() int GPUTRDTracker_t<TRDTRK, PROP>::GetCollisionIDs(int iTrk, int* collisionIds) const
{
  //--------------------------------------------------------------------
  // Check which TRD trigger times possibly match given input track.
  // If ITS-TPC matches or CE-crossing TPC tracks the time is precisely
  // known and max 1 trigger time can be assigned.
  // For TPC-only tracks the collision IDs are stored in collisionIds array
  // and the number of valid entries in the array is returned
  //--------------------------------------------------------------------
  int nColls = 0;
  for (unsigned int iColl = 0; iColl < GetConstantMem()->ioPtrs.nTRDTriggerRecords; ++iColl) {
    if (GetConstantMem()->ioPtrs.trdTrigRecMask && GetConstantMem()->ioPtrs.trdTrigRecMask[iColl] == 0) {
      continue;
    }
    if (GetConstantMem()->ioPtrs.trdTriggerTimes[iColl] > mTrackAttribs[iTrk].GetTimeMin() && GetConstantMem()->ioPtrs.trdTriggerTimes[iColl] < mTrackAttribs[iTrk].GetTimeMax()) {
      if (nColls == 20) {
        GPUError("Found too many collision candidates for track with tMin(%f) and tMax(%f)", mTrackAttribs[iTrk].GetTimeMin(), mTrackAttribs[iTrk].GetTimeMax());
        return nColls;
      }
      collisionIds[nColls++] = iColl;
    }
  }
  return nColls;
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::DoTrackingThread(int iTrk, int threadId)
{
  //--------------------------------------------------------------------
  // perform the tracking for one track (must be threadsafe)
  //--------------------------------------------------------------------
  int collisionIds[20] = {0}; // due to the dead time there will never exist more possible TRD triggers for a single track
  int nCollisionIds = 1;      // initialize with 1 for AliRoot compatibility
  if (mProcessPerTimeFrame) {
    nCollisionIds = GetCollisionIDs(iTrk, collisionIds);
    if (nCollisionIds == 0) {
      if (ENABLE_INFO) {
        GPUInfo("Did not find TRD data for track %i with t=%f. tMin(%f), tMax(%f)", iTrk, mTrackAttribs[iTrk].mTime, mTrackAttribs[iTrk].GetTimeMin(), mTrackAttribs[iTrk].GetTimeMax());
      }
      // no TRD data available for the bunch crossing this track originates from
      return;
    }
  }
  PROP prop(getPropagatorParam(Param().rec.trd.useExternalO2DefaultPropagator));
  mTracks[iTrk].setChi2(Param().rec.trd.penaltyChi2); // TODO check if this should not be higher
  auto trkStart = mTracks[iTrk];
  for (int iColl = 0; iColl < nCollisionIds; ++iColl) {
    // do track following for each collision candidate and keep best track
    auto trkCopy = trkStart;
    prop.setTrack(&trkCopy);
    prop.setFitInProjections(true);
    if (!FollowProlongation(&prop, &trkCopy, iTrk, threadId, collisionIds[iColl])) {
      // track following failed
      continue;
    }
    if (trkCopy.getReducedChi2() < mTracks[iTrk].getReducedChi2()) {
      mTracks[iTrk] = trkCopy; // copy back the resulting track
    }
  }
}

#ifndef GPUCA_ALIROOT_LIB // AliRoot TRD geometry functions are non-const, and cannot work with a const geometry
template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::ConvertTrkltToSpacePoint(const GPUTRDGeometry& geo, GPUTRDTrackletWord& trklt, GPUTRDSpacePoint& sp)
{
  // converts a single GPUTRDTrackletWord into GPUTRDSpacePoint
  // returns true if successfull
  int det = trklt.GetDetector();
  if (!geo.ChamberInGeometry(det)) {
    return false;
  }
  auto* matrix = geo.GetClusterMatrix(det);
  if (!matrix) {
    return false;
  }
  const GPUTRDpadPlane* pp = geo.GetPadPlane(det);
  int trkltZbin = trklt.GetZbin();
  My_Float xTrkltDet[3] = {0.f}; // trklt position in chamber coordinates
  My_Float xTrkltSec[3] = {0.f}; // trklt position in sector coordinates
  xTrkltDet[0] = geo.AnodePos() - sRadialOffset;
  xTrkltDet[1] = trklt.GetY();
  xTrkltDet[2] = pp->GetRowPos(trkltZbin) - pp->GetRowSize(trkltZbin) / 2.f - pp->GetRowPos(pp->GetNrows() / 2);
  matrix->LocalToMaster(xTrkltDet, xTrkltSec);
  sp.setX(xTrkltSec[0]);
  sp.setY(xTrkltSec[1]);
  sp.setZ(xTrkltSec[2]);
  sp.setDy(trklt.GetdY());

  return true;
}
#endif

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::CalculateSpacePoints(int iCollision)
{
  //--------------------------------------------------------------------
  // Calculates TRD space points in sector tracking coordinates
  // from online tracklets
  //--------------------------------------------------------------------

  bool result = true;
  int idxOffset = iCollision * (kNChambers + 1); // offset for accessing mTrackletIndexArray for collision iCollision

  const GPUTRDTrackletWord* tracklets = GetConstantMem()->ioPtrs.trdTracklets;

  for (int iDet = 0; iDet < kNChambers; ++iDet) {
    int iFirstTrackletInDet = mTrackletIndexArray[idxOffset + iDet];
    int iFirstTrackletInNextDet = mTrackletIndexArray[idxOffset + iDet + 1];
    int nTrackletsInDet = iFirstTrackletInNextDet - iFirstTrackletInDet;
    if (nTrackletsInDet == 0) {
      continue;
    }
    if (!mGeo->ChamberInGeometry(iDet)) {
      GPUError("Found TRD tracklets in chamber %i which is not included in the geometry", iDet);
      return false;
    }
    auto* matrix = mGeo->GetClusterMatrix(iDet);
    if (!matrix) {
      GPUError("No cluster matrix available for chamber %i. Skipping it...", iDet);
      result = false;
      continue;
    }
    const GPUTRDpadPlane* pp = mGeo->GetPadPlane(iDet);

    int trkltIdxOffset = (mProcessPerTimeFrame) ? GetConstantMem()->ioPtrs.trdTrackletIdxFirst[iCollision] : 0; // global index of first tracklet in iCollision
    int trkltIdxStart = trkltIdxOffset + iFirstTrackletInDet;
    for (int trkltIdx = trkltIdxStart; trkltIdx < trkltIdxStart + nTrackletsInDet; ++trkltIdx) {
      int trkltZbin = tracklets[trkltIdx].GetZbin();
      My_Float xTrkltDet[3] = {0.f};                                            // trklt position in chamber coordinates
      My_Float xTrkltSec[3] = {0.f};                                            // trklt position in sector coordinates
      xTrkltDet[0] = mGeo->AnodePos() + sRadialOffset;
      xTrkltDet[1] = tracklets[trkltIdx].GetY();
      xTrkltDet[2] = pp->GetRowPos(trkltZbin) - pp->GetRowSize(trkltZbin) / 2.f - pp->GetRowPos(pp->GetNrows() / 2);
      //GPUInfo("Space point local %i: x=%f, y=%f, z=%f", trkltIdx, xTrkltDet[0], xTrkltDet[1], xTrkltDet[2]);
      matrix->LocalToMaster(xTrkltDet, xTrkltSec);
      mSpacePoints[trkltIdx].setX(xTrkltSec[0]);
      mSpacePoints[trkltIdx].setY(xTrkltSec[1]);
      mSpacePoints[trkltIdx].setZ(xTrkltSec[2]);
      mSpacePoints[trkltIdx].setDy(tracklets[trkltIdx].GetdY());

      //GPUInfo("Space point global %i: x=%f, y=%f, z=%f", trkltIdx, mSpacePoints[trkltIdx].getX(), mSpacePoints[trkltIdx].getY(), mSpacePoints[trkltIdx].getZ());
    }
  }
  return result;
}

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::FollowProlongation(PROP* prop, TRDTRK* t, int iTrk, int threadId, int collisionId)
{
  //--------------------------------------------------------------------
  // Propagate TPC track layerwise through TRD and pick up closest
  // tracklet(s) on the way
  // -> returns false if prolongation could not be executed fully
  //    or track does not fullfill threshold conditions
  //--------------------------------------------------------------------

  if (ENABLE_INFO) {
    GPUInfo("Start track following for track %i at x=%f with pt=%f", t->getRefGlobalTrackIdRaw(), t->getX(), t->getPt());
  }
  mDebug->Reset();
  t->setChi2(0.f);
  float zShiftTrk = 0.f;
  if (mProcessPerTimeFrame) {
    zShiftTrk = (mTrackAttribs[iTrk].mTime - GetConstantMem()->ioPtrs.trdTriggerTimes[collisionId]) * mTPCVdrift * mTrackAttribs[iTrk].mSide;
    //float addZerr = (mTrackAttribs[iTrk].mTimeAddMax + mTrackAttribs[iTrk].mTimeSubMax) * .5f * mTPCVdrift;
    // increase Z error based on time window
    // -> this is here since it was done before, but the efficiency seems to be better if the covariance is not updated (more tracklets are attached)
    //t->updateCovZ2(addZerr * addZerr); // TODO check again once detailed performance study tools are available, maybe this can be tuned
  }
  const GPUTRDpadPlane* pad = nullptr;
  const GPUTRDTrackletWord* tracklets = GetConstantMem()->ioPtrs.trdTracklets;
  const GPUTRDSpacePoint* spacePoints = GetConstantMem()->ioPtrs.trdSpacePoints;

#ifdef ENABLE_GPUTRDDEBUG
  TRDTRK trackNoUp(*t);
#endif

  int candidateIdxOffset = threadId * 2 * mNCandidates;
  int hypothesisIdxOffset = threadId * mNCandidates;
  int trkltIdxOffset = collisionId * (kNChambers + 1);                                                            // offset for accessing mTrackletIndexArray for given collision
  int glbTrkltIdxOffset = (mProcessPerTimeFrame) ? GetConstantMem()->ioPtrs.trdTrackletIdxFirst[collisionId] : 0; // offset of first tracklet in given collision in global tracklet array

  auto trkWork = t;
  if (mNCandidates > 1) {
    // copy input track to first candidate
    mCandidates[candidateIdxOffset] = *t;
  }

  int nCandidates = 1;     // we always start with one candidate
  int nCurrHypothesis = 0; // the number of track hypothesis in given iLayer

  // search window
  float roadY = 0.f;
  float roadZ = 0.f;
  const int nMaxChambersToSearch = 4;

  mDebug->SetGeneralInfo(mNEvents, mNTracks, iTrk, t->getPt());

  for (int iLayer = 0; iLayer < kNLayers; ++iLayer) {
    nCurrHypothesis = 0;
    bool isOK = false; // if at least one candidate could be propagated or the track was stopped this becomes true
    int currIdx = candidateIdxOffset + iLayer % 2;
    int nextIdx = candidateIdxOffset + (iLayer + 1) % 2;
    pad = mGeo->GetPadPlane(iLayer, 0);
    float tilt = CAMath::Tan(CAMath::Pi() / 180.f * pad->GetTiltingAngle()); // tilt is signed!
    const float zMaxTRD = pad->GetRow0();

    // --------------------------------------------------------------------------------
    //
    // for each candidate, propagate to layer radius and look for close tracklets
    //
    // --------------------------------------------------------------------------------
    for (int iCandidate = 0; iCandidate < nCandidates; iCandidate++) {

      int det[nMaxChambersToSearch] = {-1, -1, -1, -1}; // TRD chambers to be searched for tracklets

      if (mNCandidates > 1) {
        trkWork = &mCandidates[2 * iCandidate + currIdx];
        prop->setTrack(trkWork);
      }

      if (trkWork->getIsStopped()) {
        Hypothesis hypo(trkWork->getNlayersFindable(), iCandidate, -1, trkWork->getChi2());
        InsertHypothesis(hypo, nCurrHypothesis, hypothesisIdxOffset);
        isOK = true;
        continue;
      }

      // propagate track to average radius of TRD layer iLayer (sector 0, stack 2 is chosen as a reference)
      if (!prop->propagateToX(mR[2 * kNLayers + iLayer], .8f, 2.f)) {
        if (ENABLE_INFO) {
          GPUInfo("Track propagation failed for track %i candidate %i in layer %i (pt=%f, x=%f, mR[layer]=%f)", iTrk, iCandidate, iLayer, trkWork->getPt(), trkWork->getX(), mR[2 * kNLayers + iLayer]);
        }
        continue;
      }

      // rotate track in new sector in case of sector crossing
      if (!AdjustSector(prop, trkWork)) {
        if (ENABLE_INFO) {
          GPUInfo("Adjusting sector failed for track %i candidate %i in layer %i", iTrk, iCandidate, iLayer);
        }
        continue;
      }

      // check if track is findable
      if (IsGeoFindable(trkWork, iLayer, prop->getAlpha(), zShiftTrk)) {
        trkWork->setIsFindable(iLayer);
      }

      // define search window
      roadY = 7.f * CAMath::Sqrt(trkWork->getSigmaY2() + 0.1f * 0.1f) + Param().rec.trd.extraRoadY; // add constant to the road to account for uncertainty due to radial deviations (few mm)
      // roadZ = 7.f * CAMath::Sqrt(trkWork->getSigmaZ2() + 9.f * 9.f / 12.f); // take longest pad length
      roadZ = mRoadZ + Param().rec.trd.extraRoadZ; // simply twice the longest pad length -> efficiency 99.996%
      //
      if (CAMath::Abs(trkWork->getZ() + zShiftTrk) - roadZ >= zMaxTRD) {
        if (ENABLE_INFO) {
          GPUInfo("Track out of TRD acceptance with z=%f in layer %i (eta=%f)", trkWork->getZ() + zShiftTrk, iLayer, trkWork->getEta());
        }
        continue;
      }

      // determine chamber(s) to be searched for tracklets
      FindChambersInRoad(trkWork, roadY, roadZ, iLayer, det, zMaxTRD, prop->getAlpha(), zShiftTrk);

      // track debug information to be stored in case no matching tracklet can be found
      mDebug->SetTrackParameter(*trkWork, iLayer);

      // look for tracklets in chamber(s)
      for (int iDet = 0; iDet < nMaxChambersToSearch; iDet++) {
        int currDet = det[iDet];
        if (currDet == -1) {
          continue;
        }
        pad = mGeo->GetPadPlane(currDet);
        int currSec = mGeo->GetSector(currDet);
        if (currSec != GetSector(prop->getAlpha())) {
          if (!prop->rotate(GetAlphaOfSector(currSec))) {
            if (ENABLE_WARNING) {
              GPUWarning("Track could not be rotated in tracklet coordinate system");
            }
            break;
          }
        }
        if (currSec != GetSector(prop->getAlpha())) {
          GPUError("Track is in sector %i and sector %i is searched for tracklets", GetSector(prop->getAlpha()), currSec);
          continue;
        }
        // propagate track to radius of chamber
        if (!prop->propagateToX(mR[currDet], .8f, .2f)) {
          if (ENABLE_WARNING) {
            GPUWarning("Track parameter for track %i, x=%f at chamber %i x=%f in layer %i cannot be retrieved", iTrk, trkWork->getX(), currDet, mR[currDet], iLayer);
          }
        }
        // first propagate track to x of tracklet
        for (int trkltIdx = glbTrkltIdxOffset + mTrackletIndexArray[trkltIdxOffset + currDet]; trkltIdx < glbTrkltIdxOffset + mTrackletIndexArray[trkltIdxOffset + currDet + 1]; ++trkltIdx) {
          if (CAMath::Abs(trkWork->getY() - spacePoints[trkltIdx].getY()) > roadY || CAMath::Abs(trkWork->getZ() + zShiftTrk - spacePoints[trkltIdx].getZ()) > roadZ) {
            // skip tracklets which are too far away
            // although the radii of space points and tracks may differ by ~ few mm the roads are large enough to allow no efficiency loss by this cut
            continue;
          }
          float projY, projZ;
          prop->getPropagatedYZ(spacePoints[trkltIdx].getX(), projY, projZ);
          // correction for tilted pads (only applied if deltaZ < lPad && track z err << lPad)
          float tiltCorr = tilt * (spacePoints[trkltIdx].getZ() - projZ);
          float lPad = pad->GetRowSize(tracklets[trkltIdx].GetZbin());
          if (!((CAMath::Abs(spacePoints[trkltIdx].getZ() - projZ) < lPad) && (trkWork->getSigmaZ2() < (lPad * lPad / 12.f)))) {
            tiltCorr = 0.f; // will be zero also for TPC tracks which are shifted in z
          }
          // correction for mean z position of tracklet (is not the center of the pad if track eta != 0)
          float zPosCorr = spacePoints[trkltIdx].getZ() + mZCorrCoefNRC * trkWork->getTgl();
          float yPosCorr = spacePoints[trkltIdx].getY() - tiltCorr;
          zPosCorr -= zShiftTrk; // shift tracklet instead of track in order to avoid having to do a re-fit for each collision
          float deltaY = yPosCorr - projY;
          float deltaZ = zPosCorr - projZ;
          My_Float trkltPosTmpYZ[2] = {yPosCorr, zPosCorr};
          My_Float trkltCovTmp[3] = {0.f};
          if ((CAMath::Abs(deltaY) < roadY) && (CAMath::Abs(deltaZ) < roadZ)) { // TODO: check if this is still necessary after the cut before propagation of track
            // tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            RecalcTrkltCov(tilt, trkWork->getSnp(), pad->GetRowSize(tracklets[trkltIdx].GetZbin()), trkltCovTmp);
            float chi2 = prop->getPredictedChi2(trkltPosTmpYZ, trkltCovTmp);
            // TODO cut on angular pull should be made stricter when proper v-drift calibration for the TRD tracklets is implemented
            if ((chi2 > Param().rec.trd.maxChi2) || (Param().rec.trd.applyDeflectionCut && CAMath::Abs(GetAngularPull(spacePoints[trkltIdx].getDy(), trkWork->getSnp())) > 4)) {
              continue;
            }
            Hypothesis hypo(trkWork->getNlayersFindable(), iCandidate, trkltIdx, trkWork->getChi2() + chi2);
            InsertHypothesis(hypo, nCurrHypothesis, hypothesisIdxOffset);
          }   // end tracklet in window
        }     // tracklet loop
      }       // chamber loop

      // add no update to hypothesis list
      Hypothesis hypoNoUpdate(trkWork->getNlayersFindable(), iCandidate, -1, trkWork->getChi2() + Param().rec.trd.penaltyChi2);
      InsertHypothesis(hypoNoUpdate, nCurrHypothesis, hypothesisIdxOffset);
      isOK = true;
    } // end candidate loop

    mDebug->SetChi2Update(mHypothesis[0 + hypothesisIdxOffset].mChi2 - t->getChi2(), iLayer); // only meaningful for ONE candidate!!!
    mDebug->SetRoad(roadY, roadZ, iLayer);                                                    // only meaningful for ONE candidate
    bool wasTrackStored = false;
    // --------------------------------------------------------------------------------
    //
    // loop over the best N_candidates hypothesis
    //
    // --------------------------------------------------------------------------------
    // GPUInfo("nCurrHypothesis=%i, nCandidates=%i", nCurrHypothesis, nCandidates);
    // for (int idx=0; idx<10; ++idx) { GPUInfo("mHypothesis[%i]: candidateId=%i, nLayers=%i, trackletId=%i, chi2=%f", idx, mHypothesis[idx].mCandidateId,  mHypothesis[idx].mLayers, mHypothesis[idx].mTrackletId, mHypothesis[idx].mChi2); }
    for (int iUpdate = 0; iUpdate < nCurrHypothesis && iUpdate < mNCandidates; iUpdate++) {
      if (mHypothesis[iUpdate + hypothesisIdxOffset].mCandidateId == -1) {
        // no more candidates
        if (iUpdate == 0) {
          return false; // no valid candidates for this track (probably propagation failed)
        }
        break; // go to next layer
      }
      nCandidates = iUpdate + 1;
      if (mNCandidates > 1) {
        mCandidates[2 * iUpdate + nextIdx] = mCandidates[2 * mHypothesis[iUpdate + hypothesisIdxOffset].mCandidateId + currIdx];
        trkWork = &mCandidates[2 * iUpdate + nextIdx];
      }
      if (mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId == -1) {
        // no matching tracklet found
        if (trkWork->getIsFindable(iLayer)) {
          if (trkWork->getNmissingConsecLayers(iLayer) > Param().rec.trd.stopTrkAfterNMissLy) {
            trkWork->setIsStopped();
          }
          trkWork->setChi2(trkWork->getChi2() + Param().rec.trd.penaltyChi2);
        }
        if (iUpdate == 0 && mNCandidates > 1) { // TODO: is thie really necessary????? CHECK!
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }
      // matching tracklet found
      if (mNCandidates > 1) {
        prop->setTrack(trkWork);
      }
      int trkltSec = mGeo->GetSector(tracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector());
      if (trkltSec != GetSector(prop->getAlpha())) {
        // if after a matching tracklet was found another sector was searched for tracklets the track needs to be rotated back
        prop->rotate(GetAlphaOfSector(trkltSec));
      }
      if (!prop->propagateToX(spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getX(), .8f, 2.f)) {
        if (ENABLE_WARNING) {
          GPUWarning("Final track propagation for track %i update %i in layer %i failed", iTrk, iUpdate, iLayer);
        }
        trkWork->setChi2(trkWork->getChi2() + Param().rec.trd.penaltyChi2);
        if (trkWork->getIsFindable(iLayer)) {
          if (trkWork->getNmissingConsecLayers(iLayer) >= Param().rec.trd.stopTrkAfterNMissLy) {
            trkWork->setIsStopped();
          }
        }
        if (iUpdate == 0 && mNCandidates > 1) {
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }

      pad = mGeo->GetPadPlane(tracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector());
      float tiltCorrUp = tilt * (spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getZ() - trkWork->getZ());
      float zPosCorrUp = spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getZ() + mZCorrCoefNRC * trkWork->getTgl();
      zPosCorrUp -= zShiftTrk;
      float padLength = pad->GetRowSize(tracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetZbin());
      if (!((trkWork->getSigmaZ2() < (padLength * padLength / 12.f)) && (CAMath::Abs(spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getZ() - trkWork->getZ()) < padLength))) {
        tiltCorrUp = 0.f;
      }
      My_Float trkltPosUp[2] = {spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getY() - tiltCorrUp, zPosCorrUp};
      My_Float trkltCovUp[3] = {0.f};
      RecalcTrkltCov(tilt, trkWork->getSnp(), pad->GetRowSize(tracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetZbin()), trkltCovUp);

#ifdef ENABLE_GPUTRDDEBUG
      prop->setTrack(&trackNoUp);
      prop->rotate(GetAlphaOfSector(trkltSec));
      //prop->propagateToX(spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getX(), .8f, 2.f);
      prop->propagateToX(mR[tracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector()], .8f, 2.f);
      prop->setTrack(trkWork);
#endif

      if (!wasTrackStored) {
#ifdef ENABLE_GPUTRDDEBUG
        mDebug->SetTrackParameterNoUp(trackNoUp, iLayer);
#endif
        mDebug->SetTrackParameter(*trkWork, iLayer);
        mDebug->SetRawTrackletPosition(spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getX(), spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getY(), spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getZ(), iLayer);
        mDebug->SetCorrectedTrackletPosition(trkltPosUp, iLayer);
        mDebug->SetTrackletCovariance(trkltCovUp, iLayer);
        mDebug->SetTrackletProperties(spacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].getDy(), tracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector(), iLayer);
        wasTrackStored = true;
      }

      if (!prop->update(trkltPosUp, trkltCovUp)) {
        if (ENABLE_WARNING) {
          GPUWarning("Failed to update track %i with space point in layer %i", iTrk, iLayer);
        }
        trkWork->setChi2(trkWork->getChi2() + Param().rec.trd.penaltyChi2);
        if (trkWork->getIsFindable(iLayer)) {
          if (trkWork->getNmissingConsecLayers(iLayer) >= Param().rec.trd.stopTrkAfterNMissLy) {
            trkWork->setIsStopped();
          }
        }
        if (iUpdate == 0 && mNCandidates > 1) {
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }
      if (!trkWork->CheckNumericalQuality()) {
        if (ENABLE_INFO) {
          GPUInfo("Track %i has invalid covariance matrix. Aborting track following\n", iTrk);
        }
        return false;
      }
      trkWork->addTracklet(iLayer, mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId);
      trkWork->setChi2(mHypothesis[iUpdate + hypothesisIdxOffset].mChi2);
      trkWork->setIsFindable(iLayer);
      trkWork->setCollisionId(collisionId);
      if (iUpdate == 0 && mNCandidates > 1) {
        *t = mCandidates[2 * iUpdate + nextIdx];
      }
    } // end update loop

    if (!isOK) {
      if (ENABLE_INFO) {
        GPUInfo("Track %i cannot be followed. Stopped in layer %i", iTrk, iLayer);
      }
      return false;
    }
  } // end layer loop

  // --------------------------------------------------------------------------------
  // add some debug information (compare labels of attached tracklets to track label)
  // and store full track information
  // --------------------------------------------------------------------------------
  if (mDebugOutput) {
    mDebug->SetTrack(*t);
    mDebug->Output();
  }
  if (ENABLE_INFO) {
    GPUInfo("Ended track following for track %i at x=%f with pt=%f. Attached %i tracklets", t->getRefGlobalTrackIdRaw(), t->getX(), t->getPt(), t->getNtracklets());
  }
  if (nCurrHypothesis > 1) {
    if (CAMath::Abs(mHypothesis[hypothesisIdxOffset + 1].GetReducedChi2() - mHypothesis[hypothesisIdxOffset].GetReducedChi2()) < Param().rec.trd.chi2SeparationCut) {
      t->setIsAmbiguous();
    }
  }
  return true;
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::InsertHypothesis(Hypothesis hypo, int& nCurrHypothesis, int idxOffset)
{
  // Insert hypothesis into the array. If the array is full and the reduced chi2 is worse
  // than the worst hypothesis in the array it is dropped.
  // The hypothesis array is always sorted.

  if (nCurrHypothesis == 0) {
    // this is the first hypothesis in the array
    mHypothesis[idxOffset] = hypo;
    ++nCurrHypothesis;
  } else if (nCurrHypothesis > 0 && nCurrHypothesis < mNCandidates) {
    // insert the hypothesis into the right position and shift all worse hypothesis to the right
    for (int i = idxOffset; i < nCurrHypothesis + idxOffset; ++i) {
      if (hypo.GetReducedChi2() < mHypothesis[i].GetReducedChi2()) {
        for (int k = nCurrHypothesis + idxOffset; k > i; --k) {
          mHypothesis[k] = mHypothesis[k - 1];
        }
        mHypothesis[i] = hypo;
        ++nCurrHypothesis;
        return;
      }
    }
    mHypothesis[nCurrHypothesis + idxOffset] = hypo;
    ++nCurrHypothesis;
    return;
  } else {
    // array is already full, check if new hypothesis should be inserted
    int i = nCurrHypothesis + idxOffset - 1;
    for (; i >= idxOffset; --i) {
      if (mHypothesis[i].GetReducedChi2() < hypo.GetReducedChi2()) {
        break;
      }
    }
    if (i < (nCurrHypothesis + idxOffset - 1)) {
      // new hypothesis should be inserted into the array
      for (int k = nCurrHypothesis + idxOffset - 1; k > i + 1; --k) {
        mHypothesis[k] = mHypothesis[k - 1];
      }
      mHypothesis[i + 1] = hypo;
    }
  }
}

template <class TRDTRK, class PROP>
GPUd() int GPUTRDTracker_t<TRDTRK, PROP>::GetDetectorNumber(const float zPos, const float alpha, const int layer) const
{
  //--------------------------------------------------------------------
  // if track position is within chamber return the chamber number
  // otherwise return -1
  //--------------------------------------------------------------------
  int stack = mGeo->GetStack(zPos, layer);
  if (stack < 0) {
    return -1;
  }
  int sector = GetSector(alpha);

  return mGeo->GetDetector(layer, stack, sector);
}

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::AdjustSector(PROP* prop, TRDTRK* t) const
{
  //--------------------------------------------------------------------
  // rotate track in new sector if necessary and
  // propagate to previous x afterwards
  // cancel if track crosses two sector boundaries
  //--------------------------------------------------------------------
  float alpha = mGeo->GetAlpha();
  float xTmp = t->getX();
  float y = t->getY();
  float yMax = t->getX() * CAMath::Tan(0.5f * alpha);
  float alphaCurr = t->getAlpha();

  if (CAMath::Abs(y) > 2.f * yMax) {
    if (ENABLE_INFO) {
      GPUInfo("AdjustSector: Track %i with pT = %f crossing two sector boundaries at x = %f", t->getRefGlobalTrackIdRaw(), t->getPt(), t->getX());
    }
    return false;
  }

  int nTries = 0;
  while (CAMath::Abs(y) > yMax) {
    if (nTries >= 2) {
      return false;
    }
    int sign = (y > 0) ? 1 : -1;
    float alphaNew = alphaCurr + alpha * sign;
    if (alphaNew > CAMath::Pi()) {
      alphaNew -= 2 * CAMath::Pi();
    } else if (alphaNew < -CAMath::Pi()) {
      alphaNew += 2 * CAMath::Pi();
    }
    if (!prop->rotate(alphaNew)) {
      return false;
    }
    if (!prop->propagateToX(xTmp, .8f, 2.f)) {
      return false;
    }
    y = t->getY();
    ++nTries;
  }
  return true;
}

template <class TRDTRK, class PROP>
GPUd() int GPUTRDTracker_t<TRDTRK, PROP>::GetSector(float alpha) const
{
  //--------------------------------------------------------------------
  // TRD sector number for reference system alpha
  //--------------------------------------------------------------------
  if (alpha < 0) {
    alpha += 2.f * CAMath::Pi();
  } else if (alpha >= 2.f * CAMath::Pi()) {
    alpha -= 2.f * CAMath::Pi();
  }
  return (int)(alpha * kNSectors / (2.f * CAMath::Pi()));
}

template <class TRDTRK, class PROP>
GPUd() float GPUTRDTracker_t<TRDTRK, PROP>::GetAlphaOfSector(const int sec) const
{
  //--------------------------------------------------------------------
  // rotation angle for TRD sector sec
  //--------------------------------------------------------------------
  float alpha = 2.0f * CAMath::Pi() / (float)kNSectors * ((float)sec + 0.5f);
  if (alpha > CAMath::Pi()) {
    alpha -= 2 * CAMath::Pi();
  }
  return alpha;
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3])
{
  //--------------------------------------------------------------------
  // recalculate tracklet covariance taking track phi angle into account
  // store the new values in cov
  //--------------------------------------------------------------------
  float t2 = tilt * tilt;      // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = GetRPhiRes(snp);
  float sz2 = rowSize * rowSize / 12.f;
  cov[0] = c2 * (sy2 + t2 * sz2);
  cov[1] = c2 * tilt * (sz2 - sy2);
  cov[2] = c2 * (t2 * sy2 + sz2);
}

template <class TRDTRK, class PROP>
GPUd() float GPUTRDTracker_t<TRDTRK, PROP>::GetAngularPull(float dYtracklet, float snp) const
{
  float dYtrack = ConvertAngleToDy(snp);
  float dYresolution = GetAngularResolution(snp);
  if (dYresolution < 1e-6f) {
    return 999.f;
  }
  return (dYtracklet - dYtrack) / CAMath::Sqrt(dYresolution);
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::FindChambersInRoad(const TRDTRK* t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha, const float zShiftTrk) const
{
  //--------------------------------------------------------------------
  // determine initial chamber where the track ends up
  // add more chambers of the same sector or (and) neighbouring
  // stack if track is close the edge(s) of the chamber
  //--------------------------------------------------------------------

  const float yMax = CAMath::Abs(mGeo->GetCol0(iLayer));
  float zTrk = t->getZ() + zShiftTrk;

  int currStack = mGeo->GetStack(zTrk, iLayer);
  int currSec = GetSector(alpha);
  int currDet;

  int nDets = 0;


  if (currStack > -1) {
    // chamber unambiguous
    currDet = mGeo->GetDetector(iLayer, currStack, currSec);
    det[nDets++] = currDet;
    const GPUTRDpadPlane* pp = mGeo->GetPadPlane(iLayer, currStack);
    int lastPadRow = mGeo->GetRowMax(iLayer, currStack, 0);
    float zCenter = pp->GetRowPos(lastPadRow / 2);
    if ((zTrk + roadZ) > pp->GetRow0() || (zTrk - roadZ) < pp->GetRowEnd()) {
      int addStack = zTrk > zCenter ? currStack - 1 : currStack + 1;
      if (addStack < kNStacks && addStack > -1) {
        det[nDets++] = mGeo->GetDetector(iLayer, addStack, currSec);
      }
    }
  } else {
    if (CAMath::Abs(zTrk) > zMax) {
      // shift track in z so it is in the TRD acceptance
      if (zTrk > 0) {
        currDet = mGeo->GetDetector(iLayer, 0, currSec);
      } else {
        currDet = mGeo->GetDetector(iLayer, kNStacks - 1, currSec);
      }
      det[nDets++] = currDet;
      currStack = mGeo->GetStack(currDet);
    } else {
      // track in between two stacks, add both surrounding chambers
      // gap between two stacks is 4 cm wide
      currDet = GetDetectorNumber(zTrk + 4.0f, alpha, iLayer);
      if (currDet != -1) {
        det[nDets++] = currDet;
      }
      currDet = GetDetectorNumber(zTrk - 4.0f, alpha, iLayer);
      if (currDet != -1) {
        det[nDets++] = currDet;
      }
    }
  }
  // add chamber(s) from neighbouring sector in case the track is close to the boundary
  if ((CAMath::Abs(t->getY()) + roadY) > yMax) {
    const int nStacksToSearch = nDets;
    int newSec;
    if (t->getY() > 0) {
      newSec = (currSec + 1) % kNSectors;
    } else {
      newSec = (currSec > 0) ? currSec - 1 : kNSectors - 1;
    }
    for (int idx = 0; idx < nStacksToSearch; ++idx) {
      currStack = mGeo->GetStack(det[idx]);
      det[nDets++] = mGeo->GetDetector(iLayer, currStack, newSec);
    }
  }
  // skip PHOS hole and non-existing chamber 17_4_4
  for (int iDet = 0; iDet < nDets; iDet++) {
    if (!mGeo->ChamberInGeometry(det[iDet])) {
      det[iDet] = -1;
    }
  }
}

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::IsGeoFindable(const TRDTRK* t, const int layer, const float alpha, const float zShiftTrk) const
{
  //--------------------------------------------------------------------
  // returns true if track position inside active area of the TRD
  // and not too close to the boundaries
  //--------------------------------------------------------------------

  float zTrk = t->getZ() + zShiftTrk;

  int det = GetDetectorNumber(zTrk, alpha, layer);

  // reject tracks between stacks
  if (det < 0) {
    return false;
  }

  // reject tracks in PHOS hole and for non existent chamber 17_4_4
  if (!mGeo->ChamberInGeometry(det)) {
    return false;
  }

  const GPUTRDpadPlane* pp = mGeo->GetPadPlane(det);
  float yMax = pp->GetColEnd();
  float zMax = pp->GetRow0();
  float zMin = pp->GetRowEnd();

  float epsY = 5.f;
  float epsZ = 5.f;

  // reject tracks closer than epsY cm to pad plane boundary
  if (yMax - CAMath::Abs(t->getY()) < epsY) {
    return false;
  }
  // reject tracks closer than epsZ cm to stack boundary
  if (!((zTrk > zMin + epsZ) && (zTrk < zMax - epsZ))) {
    return false;
  }

  return true;
}


#ifndef GPUCA_GPUCODE
namespace GPUCA_NAMESPACE
{
namespace gpu
{
// instantiate version for AliExternalTrackParam / o2::TrackParCov data types
template class GPUTRDTracker_t<GPUTRDTrack, GPUTRDPropagator>;
// always instantiate version for GPU Track Model
template class GPUTRDTracker_t<GPUTRDTrackGPU, GPUTRDPropagatorGPU>;
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#endif
