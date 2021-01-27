// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#endif
#include <chrono>
#include <vector>
#ifdef GPUCA_ALIROOT_LIB
#include "TDatabasePDG.h"
#include "AliMCParticle.h"
#include "AliMCEvent.h"
//static const float piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass();
#else
//static const float piMass = 0.139f;
#endif

#include "GPUChainTracking.h"

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::SetMaxData(const GPUTrackingInOutPointers& io)
{
  mNMaxTracks = io.nMergedTracks;
  mNMaxSpacePoints = io.nTRDTracklets;
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
  computePointerWithAlignment(base, mTrackletIndexArray, (kNChambers + 1) * mNMaxCollisions);
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
  computePointerWithAlignment(base, mTracklets, mNMaxSpacePoints * mNMaxCollisions);
  computePointerWithAlignment(base, mSpacePoints, mNMaxSpacePoints * mNMaxCollisions);
  computePointerWithAlignment(base, mTrackletLabels, 3 * mNMaxSpacePoints * mNMaxCollisions);
  return base;
}

template <class TRDTRK, class PROP>
void* GPUTRDTracker_t<TRDTRK, PROP>::SetPointersTracks(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for tracks (this is done once per event)
  //--------------------------------------------------------------------
  computePointerWithAlignment(base, mTracks, mNMaxTracks);
  return base;
}

template <class TRDTRK, class PROP>
GPUTRDTracker_t<TRDTRK, PROP>::GPUTRDTracker_t() : mR(nullptr), mIsInitialized(false), mProcessPerTimeFrame(false), mMemoryPermanent(-1), mMemoryTracklets(-1), mMemoryTracks(-1), mNMaxCollisions(1), mNMaxTracks(0), mNMaxSpacePoints(0), mTracks(nullptr), mNCandidates(1), mNCollisions(1), mNTracks(0), mNEvents(0), mTriggerRecordIndices(nullptr), mTriggerRecordTimes(nullptr), mTracklets(nullptr), mMaxThreads(100), mNTracklets(0), mTrackletIndexArray(nullptr), mHypothesis(nullptr), mCandidates(nullptr), mSpacePoints(nullptr), mTrackletLabels(nullptr), mGeo(nullptr), mRPhiA2(0), mRPhiB(0), mRPhiC2(0), mDyA2(0), mDyB(0), mDyC2(0), mAngleToDyA(0), mAngleToDyB(0), mAngleToDyC(0), mDebugOutput(false), mTimeWindow(.1f), mRadialOffset(-0.1), mMaxEta(0.84f), mExtraRoadY(2.f), mRoadZ(18.f), mZCorrCoefNRC(1.4f), mMCEvent(nullptr), mDebug(new GPUTRDTrackerDebug<TRDTRK>())
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
    Error("Init", "TRD geometry must be provided externally");
  }

  float Bz = Param().par.BzkG;
  GPUInfo("Initializing with B-field: %f kG", Bz);
  if (abs(abs(Bz) - 2) < 0.1) {
    // magnetic field +-0.2 T
    if (Bz > 0) {
      GPUInfo("Loading error parameterization for Bz = +2 kG");
      mRPhiA2 = 1.6e-3f, mRPhiB = -1.43e-2f, mRPhiC2 = 4.55e-2f;
      mDyA2 = 1.225e-3f, mDyB = -9.8e-3f, mDyC2 = 3.88e-2f;
      mAngleToDyA = -0.1f, mAngleToDyB = 1.89f, mAngleToDyC = -0.4f;
    } else {
      GPUInfo("Loading error parameterization for Bz = -2 kG");
      mRPhiA2 = 1.6e-3f, mRPhiB = 1.43e-2f, mRPhiC2 = 4.55e-2f;
      mDyA2 = 1.225e-3f, mDyB = 9.8e-3f, mDyC2 = 3.88e-2f;
      mAngleToDyA = 0.1f, mAngleToDyB = 1.89f, mAngleToDyC = 0.4f;
    }
  } else if (abs(abs(Bz) - 5) < 0.1) {
    // magnetic field +-0.5 T
    if (Bz > 0) {
      GPUInfo("Loading error parameterization for Bz = +5 kG");
      mRPhiA2 = 1.6e-3f, mRPhiB = 0.125f, mRPhiC2 = 0.0961f;
      mDyA2 = 1.681e-3f, mDyB = 0.15f, mDyC2 = 0.1849f;
      mAngleToDyA = 0.13f, mAngleToDyB = 2.43f, mAngleToDyC = -0.58f;
    } else {
      GPUInfo("Loading error parameterization for Bz = -5 kG");
      mRPhiA2 = 1.6e-3f, mRPhiB = -0.14f, mRPhiC2 = 0.1156f;
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
  mNTracklets = 0;
  mNTracks = 0;
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::DoTracking(GPUChainTracking* chainTracking)
{
  //--------------------------------------------------------------------
  // Steering function for the tracking
  //--------------------------------------------------------------------

  // sort tracklets and fill index array
  for (int iColl = 0; iColl < mNCollisions; ++iColl) {
    int nTrklts = 0;
    if (mProcessPerTimeFrame) {
      // FIXME maybe two nested if statements are not so good in terms of performance?
      nTrklts = (iColl < mNCollisions - 1) ? mTriggerRecordIndices[iColl + 1] - mTriggerRecordIndices[iColl] : mNTracklets - mTriggerRecordIndices[iColl];
    } else {
      nTrklts = mNTracklets;
    }
    GPUTRDTrackletWord* tracklets = (mProcessPerTimeFrame) ? &(mTracklets[mTriggerRecordIndices[iColl]]) : mTracklets;
    CAAlgo::sort(tracklets, tracklets + nTrklts); // tracklets are sorted by HCId
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

    if (!CalculateSpacePoints(iColl)) {
      GPUError("Space points for at least one chamber could not be calculated (for interaction %i)", iColl);
      break;
    }
  }

  auto timeStart = std::chrono::high_resolution_clock::now();

  if (mRec->GetRecoStepsGPU() & GPUDataTypes::RecoStep::TRDTracking) {
    chainTracking->DoTRDGPUTracking();
  } else {
#ifdef WITH_OPENMP
#pragma omp parallel for
    for (int iTrk = 0; iTrk < mNTracks; ++iTrk) {
      if (omp_get_num_threads() > mMaxThreads) {
        GPUError("Number of parallel threads too high, aborting tracking");
        // break statement not possible in OpenMP for loop
        iTrk = mNTracks;
        continue;
      }
      DoTrackingThread(iTrk, omp_get_thread_num());
    }
#else
    for (int iTrk = 0; iTrk < mNTracks; ++iTrk) {
      DoTrackingThread(iTrk);
    }
#endif
  }

  auto duration = std::chrono::high_resolution_clock::now() - timeStart;
  (void)duration; // suppress warning about unused variable
  /*
  std::cout << "--->  -----> -------> ---------> ";
  std::cout << "Time for event " << mNEvents << ": " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << " us ";
  std::cout << "nTracks: " << mNTracks;
  std::cout << " nTracklets: " << mNTracklets;
  std::cout << std::endl;
  */
  //DumpTracks();
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
    Error("SetNCandidates", "Cannot change mNCandidates after initialization");
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
  GPUInfo(" maxChi2(%.2f), chi2Penalty(%.2f), nCandidates(%i), maxMissingLayers(%i)", Param().rec.trdMaxChi2, Param().rec.trdPenaltyChi2, mNCandidates, Param().rec.trdStopTrkAfterNMissLy);
  GPUInfo(" ptCut = %.2f GeV, abs(eta) < %.2f", Param().rec.trdMinTrackPt, mMaxEta);
  GPUInfo("##############################################################");
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::StartDebugging()
{
  mDebug->CreateStreamer();
}

template <class TRDTRK, class PROP>
void GPUTRDTracker_t<TRDTRK, PROP>::CountMatches(const int trackID, std::vector<int>* matches) const
{
//--------------------------------------------------------------------
// search in all TRD chambers for matching tracklets
// including all tracklets created by the track and its daughters
// important: tracklets far away / pointing in different direction of
// the track should be rejected (or this has to be done afterwards in analysis)
//--------------------------------------------------------------------
#ifndef GPUCA_GPUCODE
#ifdef ENABLE_GPUMC
  for (int k = 0; k < kNChambers; k++) {
    int layer = mGeo->GetLayer(k);
    for (int trkltIdx = mTrackletIndexArray[k]; trkltIdx < mTrackletIndexArray[k + 1]; trkltIdx++) {
      bool trkltStored = false;
      for (int il = 0; il < 3; il++) {
        int lb = mSpacePoints[trkltIdx].mLabel[il];
        if (lb < 0) {
          // no more valid labels
          break;
        }
        if (lb == CAMath::Abs(trackID)) {
          matches[layer].push_back(trkltIdx);
          break;
        }
        if (!mMCEvent) {
          continue;
        }
        //continue; //FIXME uncomment to count only exact matches
        AliMCParticle* mcPart = (AliMCParticle*)mMCEvent->GetTrack(lb);
        while (mcPart) {
          lb = mcPart->GetMother();
          if (lb == CAMath::Abs(trackID)) {
            matches[layer].push_back(trkltIdx);
            trkltStored = true;
            break;
          }
          mcPart = lb >= 0 ? (AliMCParticle*)mMCEvent->GetTrack(lb) : 0;
        }
        if (trkltStored) {
          break;
        }
      }
    }
  }
#endif
#endif
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::CheckTrackRefs(const int trackID, bool* findableMC) const
{
#ifdef ENABLE_GPUMC
  //--------------------------------------------------------------------
  // loop over all track references for the input trackID and set
  // findableMC to true for each layer in which a track hit  both
  // entering and exiting the TRD chamber exists
  // (in debug mode)
  //--------------------------------------------------------------------
  TParticle* particle;
  TClonesArray* trackRefs;

  int nHits = mMCEvent->GetParticleAndTR(trackID, particle, trackRefs);
  if (nHits < 1) {
    return;
  }
  bool isFindable[2 * kNLayers] = {false};
  int nHitsTrd = 0;
  for (int iHit = 0; iHit < nHits; ++iHit) {
    AliTrackReference* trackReference = static_cast<AliTrackReference*>(trackRefs->UncheckedAt(iHit));
    if (trackReference->DetectorId() != AliTrackReference::kTRD) {
      continue;
    }
    nHitsTrd++;
    float xLoc = trackReference->LocalX();
    if (!((trackReference->TestBits(0x1 << 18)) || (trackReference->TestBits(0x1 << 17)))) {
      //if (!trackReference->TestBits(0x1 << 18)) {
      // bit 17 - entering; bit 18 - exiting
      continue;
    }
    int layer = -1;
    if (xLoc < 304.f) {
      layer = 0;
    } else if (xLoc < 317.f) {
      layer = 1;
    } else if (xLoc < 330.f) {
      layer = 2;
    } else if (xLoc < 343.f) {
      layer = 3;
    } else if (xLoc < 356.f) {
      layer = 4;
    } else if (xLoc < 369.f) {
      layer = 5;
    }
    if (layer < 0) {
      GPUError("No layer can be determined for x=%f, y=%f, z=%f, layer=%i", xLoc, trackReference->LocalY(), trackReference->Z(), layer);
      continue;
    }
    if (trackReference->TestBits(0x1 << 18)) {
      isFindable[layer * 2] = true;
    }
    if (trackReference->TestBits(0x1 << 17)) {
      isFindable[layer * 2 + 1] = true;
    }
  }
  for (int iLayer = 0; iLayer < kNLayers; ++iLayer) {
    if (isFindable[iLayer * 2] && isFindable[iLayer * 2 + 1]) {
      findableMC[iLayer] = true;
    } else {
      findableMC[iLayer] = false;
    }
  }
#endif
}
#endif //! GPUCA_GPUCODE

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::CheckTrackTRDCandidate(const TRDTRK& trk) const
{
  if (!trk.CheckNumericalQuality()) {
    return false;
  }
  if (CAMath::Abs(trk.getEta()) > mMaxEta) {
    return false;
  }
  if (trk.getPt() < Param().rec.trdMinTrackPt) {
    return false;
  }
  return true;
}

template <class TRDTRK, class PROP>
GPUd() int GPUTRDTracker_t<TRDTRK, PROP>::LoadTrack(const TRDTRK& trk, const int label, const int* nTrkltsOffline, const int labelOffline, int tpcTrackId, bool checkTrack)
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
  mTracks[mNTracks].SetTPCtrackId(tpcTrackId >= 0 ? tpcTrackId : mNTracks);
  if (label >= 0) {
    mTracks[mNTracks].SetLabel(label);
  }
  if (nTrkltsOffline) {
    for (int i = 0; i < 4; ++i) {
      mTracks[mNTracks].SetNtrackletsOffline(i, nTrkltsOffline[i]); // see GPUTRDTrack.h for information on the index
    }
  }
  mTracks[mNTracks].SetLabelOffline(labelOffline);
  mNTracks++;
  return (0);
}

template <class TRDTRK, class PROP>
GPUd() int GPUTRDTracker_t<TRDTRK, PROP>::LoadTracklet(const GPUTRDTrackletWord& tracklet, const int* labels)
{
  //--------------------------------------------------------------------
  // Add single tracklet to tracker
  //--------------------------------------------------------------------
  if (mNTracklets >= mNMaxSpacePoints * mNMaxCollisions) {
    Error("LoadTracklet", "Running out of memory for tracklets, skipping tracklet(s). This should actually never happen.");
    return 1;
  }
  if (labels) {
    for (int i = 0; i < 3; ++i) {
      mTrackletLabels[3 * mNTracklets + i] = labels[i];
    }
  }
  mTracklets[mNTracklets++] = tracklet;
  return 0;
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::DumpTracks()
{
  //--------------------------------------------------------------------
  // helper function (only for debugging purposes)
  //--------------------------------------------------------------------
  GPUInfo("There are %i tracks loaded. mNMaxTracks(%i)\n", mNTracks, mNMaxTracks);
  for (int i = 0; i < mNTracks; ++i) {
    auto* trk = &(mTracks[i]);
    GPUInfo("track %i: x=%f, alpha=%f, nTracklets=%i, pt=%f", i, trk->getX(), trk->getAlpha(), trk->GetNtracklets(), trk->getPt());
  }
}

template <class TRDTRK, class PROP>
GPUd() int GPUTRDTracker_t<TRDTRK, PROP>::GetCollisionID(float trkTime) const
{
  for (int iColl = 0; iColl < mNCollisions; ++iColl) {
    if (CAMath::Abs(trkTime - mTriggerRecordTimes[iColl]) < mTimeWindow) {
      if (ENABLE_INFO) {
        GPUInfo("TRD info found from interaction %i at %f for track with time %f", iColl, mTriggerRecordTimes[iColl], trkTime);
      }
      return iColl;
    }
  }
  return -1;
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::DoTrackingThread(int iTrk, int threadId)
{
  //--------------------------------------------------------------------
  // perform the tracking for one track (must be threadsafe)
  //--------------------------------------------------------------------
  int collisionId = 0;
  if (mProcessPerTimeFrame) {
    collisionId = GetCollisionID(mTracks[iTrk].getTime());
    if (collisionId < 0) {
      if (ENABLE_INFO) {
        GPUInfo("Did not find TRD data for track with t=%f", mTracks[iTrk].getTime());
      }
      // no TRD data available for the bunch crossing this track originates from
      return;
    }
  }
  PROP prop(&Param().polynomialField);
  auto trkCopy = mTracks[iTrk];
  prop.setTrack(&trkCopy);
  prop.setFitInProjections(true);
  FollowProlongation(&prop, &trkCopy, threadId, collisionId);
  mTracks[iTrk] = trkCopy; // copy back the resulting track
}

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::CalculateSpacePoints(int iCollision)
{
  //--------------------------------------------------------------------
  // Calculates TRD space points in sector tracking coordinates
  // from online tracklets
  //--------------------------------------------------------------------

  bool result = true;
  int idxOffset = iCollision * (kNChambers + 1);

  for (int iDet = 0; iDet < kNChambers; ++iDet) {
    int nTracklets = mTrackletIndexArray[idxOffset + iDet + 1] - mTrackletIndexArray[idxOffset + iDet];
    if (nTracklets == 0) {
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
    float tilt = CAMath::Tan(M_PI / 180.f * pp->GetTiltingAngle());
    float t2 = tilt * tilt;      // tan^2 (tilt)
    float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
    float sy2 = 0.1f * 0.1f;     // sigma_rphi^2, currently assume sigma_rphi = 1 mm

    for (int trkltIdx = mTrackletIndexArray[idxOffset + iDet]; trkltIdx < mTrackletIndexArray[idxOffset + iDet + 1]; ++trkltIdx) {
      int trkltZbin = mTracklets[trkltIdx].GetZbin();
      float sz2 = pp->GetRowSize(trkltZbin) * pp->GetRowSize(trkltZbin) / 12.f; // sigma_z = l_pad/sqrt(12) TODO try a larger z error
      My_Float xTrkltDet[3] = {0.f};                                            // trklt position in chamber coordinates
      My_Float xTrkltSec[3] = {0.f};                                            // trklt position in sector coordinates
      xTrkltDet[0] = mGeo->AnodePos() + mRadialOffset;
      xTrkltDet[1] = mTracklets[trkltIdx].GetY();
      xTrkltDet[2] = pp->GetRowPos(trkltZbin) - pp->GetRowSize(trkltZbin) / 2.f - pp->GetRowPos(pp->GetNrows() / 2);
      //GPUInfo("Space point local %i: x=%f, y=%f, z=%f", trkltIdx, xTrkltDet[0], xTrkltDet[1], xTrkltDet[2]);
      matrix->LocalToMaster(xTrkltDet, xTrkltSec);
      mSpacePoints[trkltIdx].mR = xTrkltSec[0];
      mSpacePoints[trkltIdx].mX[0] = xTrkltSec[1];
      mSpacePoints[trkltIdx].mX[1] = xTrkltSec[2];
      mSpacePoints[trkltIdx].mId = mTracklets[trkltIdx].GetId();
      for (int i = 0; i < 3; i++) {
        mSpacePoints[trkltIdx].mLabel[i] = mTrackletLabels[3 * mTracklets[trkltIdx].GetId() + i];
      }
      mSpacePoints[trkltIdx].mCov[0] = c2 * (sy2 + t2 * sz2);
      mSpacePoints[trkltIdx].mCov[1] = c2 * tilt * (sz2 - sy2);
      mSpacePoints[trkltIdx].mCov[2] = c2 * (t2 * sy2 + sz2);
      mSpacePoints[trkltIdx].mDy = 0.014f * mTracklets[trkltIdx].GetdY();

      int modId = mGeo->GetSector(iDet) * GPUTRDGeometry::kNstack + mGeo->GetStack(iDet); // global TRD stack number
      unsigned short volId = mGeo->GetGeomManagerVolUID(iDet, modId);
      mSpacePoints[trkltIdx].mVolumeId = volId;
      //GPUInfo("Space point global %i: x=%f, y=%f, z=%f", trkltIdx, mSpacePoints[trkltIdx].mR, mSpacePoints[trkltIdx].mX[0], mSpacePoints[trkltIdx].mX[1]);
    }
  }
  return result;
}

template <class TRDTRK, class PROP>
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::FollowProlongation(PROP* prop, TRDTRK* t, int threadId, int collisionId)
{
  //--------------------------------------------------------------------
  // Propagate TPC track layerwise through TRD and pick up closest
  // tracklet(s) on the way
  // -> returns false if prolongation could not be executed fully
  //    or track does not fullfill threshold conditions
  //--------------------------------------------------------------------
  //GPUInfo("Start track following for track %i at x=%f with pt=%f", t->GetTPCtrackId(), t->getX(), t->getPt());
  mDebug->Reset();
  int iTrack = t->GetTPCtrackId();
  t->SetChi2(0.f);
  const GPUTRDpadPlane* pad = nullptr;

#ifdef ENABLE_GPUTRDDEBUG
  TRDTRK trackNoUp(*t);
#endif

  // look for matching tracklets via MC label
  int trackID = t->GetLabel();

#ifdef ENABLE_GPUMC
  std::vector<int> matchAvailableAll[kNLayers]; // all available MC tracklet matches for this track
  if (mDebugOutput && trackID > 0 && mMCEvent) {
    CountMatches(trackID, matchAvailableAll);
    bool findableMC[kNLayers] = {false};
    CheckTrackRefs(trackID, findableMC);
    mDebug->SetFindableMC(findableMC);
  }
#endif

  int candidateIdxOffset = threadId * 2 * mNCandidates;
  int hypothesisIdxOffset = threadId * mNCandidates;
  int trkltIdxOffset = collisionId * (kNChambers + 1);

  auto trkWork = t;
  if (mNCandidates > 1) {
    // copy input track to first candidate
    mCandidates[candidateIdxOffset] = *t;
  }

  int nCandidates = 1;

  // search window
  float roadY = 0.f;
  float roadZ = 0.f;
  const int nMaxChambersToSearch = 4;

  mDebug->SetGeneralInfo(mNEvents, mNTracks, iTrack, trackID, t->getPt());

  for (int iLayer = 0; iLayer < kNLayers; ++iLayer) {
    int nCurrHypothesis = 0;
    bool isOK = false; // if at least one candidate could be propagated or the track was stopped this becomes true
    int currIdx = candidateIdxOffset + iLayer % 2;
    int nextIdx = candidateIdxOffset + (iLayer + 1) % 2;
    pad = mGeo->GetPadPlane(iLayer, 0);
    float tilt = CAMath::Tan(M_PI / 180.f * pad->GetTiltingAngle()); // tilt is signed!
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

      if (trkWork->GetIsStopped()) {
        Hypothesis hypo(trkWork->GetNlayers(), iCandidate, -1, trkWork->GetChi2());
        InsertHypothesis(hypo, nCurrHypothesis, hypothesisIdxOffset);
        isOK = true;
        continue;
      }

      // propagate track to average radius of TRD layer iLayer (sector 0, stack 2 is chosen as a reference)
      if (!prop->propagateToX(mR[2 * kNLayers + iLayer], .8f, 2.f)) {
        if (ENABLE_INFO) {
          GPUInfo("Track propagation failed for track %i candidate %i in layer %i (pt=%f, x=%f, mR[layer]=%f)", iTrack, iCandidate, iLayer, trkWork->getPt(), trkWork->getX(), mR[2 * kNLayers + iLayer]);
        }
        continue;
      }

      // rotate track in new sector in case of sector crossing
      if (!AdjustSector(prop, trkWork)) {
        if (ENABLE_INFO) {
          GPUInfo("FollowProlongation: Adjusting sector failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        }
        continue;
      }

      // check if track is findable
      if (IsGeoFindable(trkWork, iLayer, prop->getAlpha())) {
        trkWork->SetIsFindable(iLayer);
      }

      // define search window
      roadY = 7.f * CAMath::Sqrt(trkWork->getSigmaY2() + 0.1f * 0.1f) + mExtraRoadY; // add constant to the road to account for uncertainty due to radial deviations (few mm)
      // roadZ = 7.f * CAMath::Sqrt(trkWork->getSigmaZ2() + 9.f * 9.f / 12.f); // take longest pad length
      roadZ = mRoadZ; // simply twice the longest pad length -> efficiency 99.996%
      //
      if (CAMath::Abs(trkWork->getZ()) - roadZ >= zMaxTRD) {
        if (ENABLE_INFO) {
          GPUInfo("FollowProlongation: Track out of TRD acceptance with z=%f in layer %i (eta=%f)", trkWork->getZ(), iLayer, trkWork->getEta());
        }
        continue;
      }

      // determine chamber(s) to be searched for tracklets
      FindChambersInRoad(trkWork, roadY, roadZ, iLayer, det, zMaxTRD, prop->getAlpha());

      // track debug information to be stored in case no matching tracklet can be found
      mDebug->SetTrackParameter(*trkWork, iLayer);

      // look for tracklets in chamber(s)
      for (int iDet = 0; iDet < nMaxChambersToSearch; iDet++) {
        int currDet = det[iDet];
        if (currDet == -1) {
          continue;
        }
        int currSec = mGeo->GetSector(currDet);
        if (currSec != GetSector(prop->getAlpha())) {
          if (!prop->rotate(GetAlphaOfSector(currSec))) {
            if (ENABLE_WARNING) {
              Warning("FollowProlongation", "Track could not be rotated in tracklet coordinate system");
            }
            break;
          }
        }
        if (currSec != GetSector(prop->getAlpha())) {
          Error("FollowProlongation", "Track is in sector %i and sector %i is searched for tracklets", GetSector(prop->getAlpha()), currSec);
          continue;
        }
        // propagate track to radius of chamber
        if (!prop->propagateToX(mR[currDet], .8f, .2f)) {
          if (ENABLE_WARNING) {
            Warning("FollowProlongation", "Track parameter for track %i, x=%f at chamber %i x=%f in layer %i cannot be retrieved", iTrack, trkWork->getX(), currDet, mR[currDet], iLayer);
          }
        }
        // first propagate track to x of tracklet
        for (int trkltIdx = mTrackletIndexArray[trkltIdxOffset + currDet]; trkltIdx < mTrackletIndexArray[trkltIdxOffset + currDet + 1]; ++trkltIdx) {
          if (CAMath::Abs(trkWork->getY() - mSpacePoints[trkltIdx].mX[0]) > roadY || CAMath::Abs(trkWork->getZ() - mSpacePoints[trkltIdx].mX[1]) > roadZ) {
            // skip tracklets which are too far away
            // although the radii of space points and tracks may differ by ~ few mm the roads are large enough to allow no efficiency loss by this cut
            continue;
          }
          float projY, projZ;
          prop->getPropagatedYZ(mSpacePoints[trkltIdx].mR, projY, projZ);
          // correction for tilted pads (only applied if deltaZ < l_pad && track z err << l_pad)
          float tiltCorr = tilt * (mSpacePoints[trkltIdx].mX[1] - projZ);
          float l_pad = pad->GetRowSize(mTracklets[trkltIdx].GetZbin());
          if (!((CAMath::Abs(mSpacePoints[trkltIdx].mX[1] - projZ) < l_pad) && (trkWork->getSigmaZ2() < (l_pad * l_pad / 12.f)))) {
            tiltCorr = 0.f;
          }
          // correction for mean z position of tracklet (is not the center of the pad if track eta != 0)
          float zPosCorr = mSpacePoints[trkltIdx].mX[1] + mZCorrCoefNRC * trkWork->getTgl();
          float yPosCorr = mSpacePoints[trkltIdx].mX[0] - tiltCorr;
          float deltaY = yPosCorr - projY;
          float deltaZ = zPosCorr - projZ;
          My_Float trkltPosTmpYZ[2] = {yPosCorr, zPosCorr};
          My_Float trkltCovTmp[3] = {0.f};
          if ((CAMath::Abs(deltaY) < roadY) && (CAMath::Abs(deltaZ) < roadZ)) { // TODO: check if this is still necessary after the cut before propagation of track
            // tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            RecalcTrkltCov(tilt, trkWork->getSnp(), pad->GetRowSize(mTracklets[trkltIdx].GetZbin()), trkltCovTmp);
            float chi2 = prop->getPredictedChi2(trkltPosTmpYZ, trkltCovTmp);
            // GPUInfo("layer %i: chi2 = %f", iLayer, chi2);
            if (chi2 < Param().rec.trdMaxChi2 && CAMath::Abs(GetAngularPull(mSpacePoints[trkltIdx].mDy, trkWork->getSnp())) < 4) {
              Hypothesis hypo(trkWork->GetNlayers(), iCandidate, trkltIdx, trkWork->GetChi2() + chi2);
              InsertHypothesis(hypo, nCurrHypothesis, hypothesisIdxOffset);
            } // end tracklet chi2 < Param().rec.trdMaxChi2
          }   // end tracklet in window
        }     // tracklet loop
      }       // chamber loop

      // add no update to hypothesis list
      Hypothesis hypoNoUpdate(trkWork->GetNlayers(), iCandidate, -1, trkWork->GetChi2() + Param().rec.trdPenaltyChi2);
      InsertHypothesis(hypoNoUpdate, nCurrHypothesis, hypothesisIdxOffset);
      isOK = true;
    } // end candidate loop

#ifdef ENABLE_GPUMC
    // in case matching tracklet exists in this layer -> store position information for debugging FIXME: does not yet work for time frames in o2, but here we anyway do not yet have MC labels...
    if (matchAvailableAll[iLayer].size() > 0 && mDebugOutput) {
      mDebug->SetNmatchAvail(matchAvailableAll[iLayer].size(), iLayer);
      int realTrkltId = matchAvailableAll[iLayer].at(0);
      int realTrkltDet = mTracklets[realTrkltId].GetDetector();
      prop->rotate(GetAlphaOfSector(mGeo->GetSector(realTrkltDet)));
      if (!prop->propagateToX(mSpacePoints[realTrkltId].mR, .8f, 2.f) || GetSector(prop->getAlpha()) != mGeo->GetSector(realTrkltDet)) {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Track parameter at x=%f for track %i at real tracklet x=%f in layer %i cannot be retrieved (pt=%f)", trkWork->getX(), iTrack, mSpacePoints[realTrkltId].mR, iLayer, trkWork->getPt());
        }
      } else {
        // track could be propagated, rotated and is in the same sector as the MC matching tracklet
        mDebug->SetTrackParameterReal(*trkWork, iLayer);
        float zPosCorrReal = mSpacePoints[realTrkltId].mX[1] + mZCorrCoefNRC * trkWork->getTgl();
        float deltaZReal = zPosCorrReal - trkWork->getZ();
        float tiltCorrReal = tilt * (mSpacePoints[realTrkltId].mX[1] - trkWork->getZ());
        float l_padReal = pad->GetRowSize(mTracklets[realTrkltId].GetZbin());
        if ((trkWork->getSigmaZ2() >= (l_padReal * l_padReal / 12.f)) || (CAMath::Abs(mSpacePoints[realTrkltId].mX[1] - trkWork->getZ()) >= l_padReal)) {
          tiltCorrReal = 0;
        }
        My_Float yzPosReal[2] = {mSpacePoints[realTrkltId].mX[0] - tiltCorrReal, zPosCorrReal};
        My_Float covReal[3] = {0.};
        RecalcTrkltCov(tilt, trkWork->getSnp(), pad->GetRowSize(mTracklets[realTrkltId].GetZbin()), covReal);
        mDebug->SetChi2Real(prop->getPredictedChi2(yzPosReal, covReal), iLayer);
        mDebug->SetRawTrackletPositionReal(mSpacePoints[realTrkltId].mR, mSpacePoints[realTrkltId].mX, iLayer);
        mDebug->SetCorrectedTrackletPositionReal(yzPosReal, iLayer);
        mDebug->SetTrackletPropertiesReal(mTracklets[realTrkltId].GetDetector(), iLayer);
      }
    }
#endif
    //
    mDebug->SetChi2Update(mHypothesis[0 + hypothesisIdxOffset].mChi2 - t->GetChi2(), iLayer); // only meaningful for ONE candidate!!!
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
          if (ENABLE_WARNING) {
            Warning("FollowProlongation", "No valid candidates for track %i in layer %i", iTrack, iLayer);
          }
          nCandidates = 0;
        }
        break;
      }
      nCandidates = iUpdate + 1;
      if (mNCandidates > 1) {
        mCandidates[2 * iUpdate + nextIdx] = mCandidates[2 * mHypothesis[iUpdate + hypothesisIdxOffset].mCandidateId + currIdx];
        trkWork = &mCandidates[2 * iUpdate + nextIdx];
      }
      if (mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId == -1) {
        // no matching tracklet found
        if (trkWork->GetIsFindable(iLayer)) {
          if (trkWork->GetNmissingConsecLayers(iLayer) > Param().rec.trdStopTrkAfterNMissLy) {
            trkWork->SetIsStopped();
          }
          trkWork->SetChi2(trkWork->GetChi2() + Param().rec.trdPenaltyChi2);
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
      int trkltSec = mGeo->GetSector(mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector());
      if (trkltSec != GetSector(prop->getAlpha())) {
        // if after a matching tracklet was found another sector was searched for tracklets the track needs to be rotated back
        prop->rotate(GetAlphaOfSector(trkltSec));
      }
      if (!prop->propagateToX(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mR, .8f, 2.f)) {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Final track propagation for track %i update %i in layer %i failed", iTrack, iUpdate, iLayer);
        }
        trkWork->SetChi2(trkWork->GetChi2() + Param().rec.trdPenaltyChi2);
        if (trkWork->GetIsFindable(iLayer)) {
          if (trkWork->GetNmissingConsecLayers(iLayer) >= Param().rec.trdStopTrkAfterNMissLy) {
            trkWork->SetIsStopped();
          }
        }
        if (iUpdate == 0 && mNCandidates > 1) {
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }

      float tiltCorrUp = tilt * (mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[1] - trkWork->getZ());
      float zPosCorrUp = mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[1] + mZCorrCoefNRC * trkWork->getTgl();
      float l_padTrklt = pad->GetRowSize(mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetZbin());
      if (!((trkWork->getSigmaZ2() < (l_padTrklt * l_padTrklt / 12.f)) && (CAMath::Abs(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[1] - trkWork->getZ()) < l_padTrklt))) {
        tiltCorrUp = 0.f;
      }
      My_Float trkltPosUp[2] = {mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[0] - tiltCorrUp, zPosCorrUp};
      My_Float trkltCovUp[3] = {0.f};
      RecalcTrkltCov(tilt, trkWork->getSnp(), pad->GetRowSize(mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetZbin()), trkltCovUp);

#ifdef ENABLE_GPUTRDDEBUG
      prop->setTrack(&trackNoUp);
      prop->rotate(GetAlphaOfSector(trkltSec));
      //prop->propagateToX(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mR, .8f, 2.f);
      prop->propagateToX(mR[mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector()], .8f, 2.f);
      prop->setTrack(trkWork);
#endif

      if (!wasTrackStored) {
#ifdef ENABLE_GPUTRDDEBUG
        mDebug->SetTrackParameterNoUp(trackNoUp, iLayer);
#endif
        mDebug->SetTrackParameter(*trkWork, iLayer);
        mDebug->SetRawTrackletPosition(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mR, mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX, iLayer);
        mDebug->SetCorrectedTrackletPosition(trkltPosUp, iLayer);
        mDebug->SetTrackletCovariance(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mCov, iLayer);
        mDebug->SetTrackletProperties(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mDy, mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector(), iLayer);
        wasTrackStored = true;
      }

      if (!prop->update(trkltPosUp, trkltCovUp)) {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);
        }
        trkWork->SetChi2(trkWork->GetChi2() + Param().rec.trdPenaltyChi2);
        if (trkWork->GetIsFindable(iLayer)) {
          if (trkWork->GetNmissingConsecLayers(iLayer) >= Param().rec.trdStopTrkAfterNMissLy) {
            trkWork->SetIsStopped();
          }
        }
        if (iUpdate == 0 && mNCandidates > 1) {
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }
      if (!trkWork->CheckNumericalQuality()) {
        if (ENABLE_INFO) {
          GPUInfo("FollowProlongation: Track %i has invalid covariance matrix. Aborting track following\n", iTrack);
        }
        return false;
      }
      trkWork->AddTracklet(iLayer, mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId);
      trkWork->SetChi2(mHypothesis[iUpdate + hypothesisIdxOffset].mChi2);
      trkWork->SetIsFindable(iLayer);
      if (iUpdate == 0 && mNCandidates > 1) {
        *t = mCandidates[2 * iUpdate + nextIdx];
      }
    } // end update loop

    if (!isOK) {
      if (ENABLE_INFO) {
        GPUInfo("FollowProlongation: Track %i cannot be followed. Stopped in layer %i", iTrack, iLayer);
      }
      return false;
    }
  } // end layer loop

  // --------------------------------------------------------------------------------
  // add some debug information (compare labels of attached tracklets to track label)
  // and store full track information
  // --------------------------------------------------------------------------------
  if (mDebugOutput) {
    int update[6] = {0};
    if (!mMCEvent) {
      for (int iLy = 0; iLy < kNLayers; iLy++) {
        if (t->GetTracklet(iLy) != -1) {
          update[iLy] = 1;
        }
      }
    } else {
      // for MC: check attached tracklets (match, related, fake)
      int nRelated = 0;
      int nMatching = 0;
      int nFake = 0;
      for (int iLy = 0; iLy < kNLayers; iLy++) {
        if (t->GetTracklet(iLy) != -1) {
          int lbTracklet;
          for (int il = 0; il < 3; il++) {
            if ((lbTracklet = mSpacePoints[t->GetTracklet(iLy)].mLabel[il]) < 0) {
              // no more valid labels
              continue;
            }
            if (lbTracklet == CAMath::Abs(trackID)) {
              update[iLy] = 1 + il;
              nMatching++;
              break;
            }
          }
#ifdef ENABLE_GPUMC
          if (update[iLy] < 1 && mMCEvent) {
            // no exact match, check in related labels
            bool isRelated = false;
            for (int il = 0; il < 3; il++) {
              if (isRelated) {
                break;
              }
              if ((lbTracklet = mSpacePoints[t->GetTracklet(iLy)].mLabel[il]) < 0) {
                // no more valid labels
                continue;
              }
              AliMCParticle* mcPart = (AliMCParticle*)mMCEvent->GetTrack(lbTracklet);
              while (mcPart) {
                int motherPart = mcPart->GetMother();
                if (motherPart == CAMath::Abs(trackID)) {
                  update[iLy] = 4 + il;
                  nRelated++;
                  isRelated = true;
                  break;
                }
                mcPart = motherPart >= 0 ? (AliMCParticle*)mMCEvent->GetTrack(motherPart) : 0;
              }
            }
          }
#endif
          if (update[iLy] < 1) {
            update[iLy] = 9;
            nFake++;
          }
        }
      }
      mDebug->SetTrackProperties(nMatching, nFake, nRelated);
#ifdef ENABLE_GPUMC
      AliMCParticle* mcPartDbg = (AliMCParticle*)mMCEvent->GetTrack(trackID);
      if (mcPartDbg) {
        mDebug->SetMCinfo(mcPartDbg->Xv(), mcPartDbg->Yv(), mcPartDbg->Zv(), mcPartDbg->PdgCode());
      }
#endif
    }
    mDebug->SetTrack(*t);
    mDebug->SetUpdates(update);
    mDebug->Output();
  }
  //GPUInfo("Ended track following for track %i at x=%f with pt=%f", t->GetTPCtrackId(), t->getX(), t->getPt());
  //GPUInfo("Attached %i tracklets", t->GetNtracklets());
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
      Info("AdjustSector", "Track %i with pT = %f crossing two sector boundaries at x = %f", t->GetTPCtrackId(), t->getPt(), t->getX());
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
    if (alphaNew > M_PI) {
      alphaNew -= 2 * M_PI;
    } else if (alphaNew < -M_PI) {
      alphaNew += 2 * M_PI;
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
    alpha += 2.f * M_PI;
  } else if (alpha >= 2.f * M_PI) {
    alpha -= 2.f * M_PI;
  }
  return (int)(alpha * kNSectors / (2.f * M_PI));
}

template <class TRDTRK, class PROP>
GPUd() float GPUTRDTracker_t<TRDTRK, PROP>::GetAlphaOfSector(const int sec) const
{
  //--------------------------------------------------------------------
  // rotation angle for TRD sector sec
  //--------------------------------------------------------------------
  float alpha = 2.0f * M_PI / (float)kNSectors * ((float)sec + 0.5f);
  if (alpha > M_PI) {
    alpha -= 2 * M_PI;
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
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::FindChambersInRoad(const TRDTRK* t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const
{
  //--------------------------------------------------------------------
  // determine initial chamber where the track ends up
  // add more chambers of the same sector or (and) neighbouring
  // stack if track is close the edge(s) of the chamber
  //--------------------------------------------------------------------

  const float yMax = CAMath::Abs(mGeo->GetCol0(iLayer));

  int currStack = mGeo->GetStack(t->getZ(), iLayer);
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
    if ((t->getZ() + roadZ) > pp->GetRow0() || (t->getZ() - roadZ) < pp->GetRowEnd()) {
      int addStack = t->getZ() > zCenter ? currStack - 1 : currStack + 1;
      if (addStack < kNStacks && addStack > -1) {
        det[nDets++] = mGeo->GetDetector(iLayer, addStack, currSec);
      }
    }
  } else {
    if (CAMath::Abs(t->getZ()) > zMax) {
      // shift track in z so it is in the TRD acceptance
      if (t->getZ() > 0) {
        currDet = mGeo->GetDetector(iLayer, 0, currSec);
      } else {
        currDet = mGeo->GetDetector(iLayer, kNStacks - 1, currSec);
      }
      det[nDets++] = currDet;
      currStack = mGeo->GetStack(currDet);
    } else {
      // track in between two stacks, add both surrounding chambers
      // gap between two stacks is 4 cm wide
      currDet = GetDetectorNumber(t->getZ() + 4.0f, alpha, iLayer);
      if (currDet != -1) {
        det[nDets++] = currDet;
      }
      currDet = GetDetectorNumber(t->getZ() - 4.0f, alpha, iLayer);
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
GPUd() bool GPUTRDTracker_t<TRDTRK, PROP>::IsGeoFindable(const TRDTRK* t, const int layer, const float alpha) const
{
  //--------------------------------------------------------------------
  // returns true if track position inside active area of the TRD
  // and not too close to the boundaries
  //--------------------------------------------------------------------

  int det = GetDetectorNumber(t->getZ(), alpha, layer);

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
  if (!((t->getZ() > zMin + epsZ) && (t->getZ() < zMax - epsZ))) {
    return false;
  }

  return true;
}

template <class TRDTRK, class PROP>
GPUd() void GPUTRDTracker_t<TRDTRK, PROP>::SetNCollisions(int nColl)
{
  // Set the number of collisions for a given time frame.
  // The number is taken from the TRD trigger records
  if (nColl < mNMaxCollisions) {
    mNCollisions = nColl;
  } else {
    GPUError("Cannot process more than %i collisions. The last %i collisions will be dropped", mNMaxCollisions, nColl - mNMaxCollisions);
    mNCollisions = mNMaxCollisions;
  }
}

#ifndef GPUCA_GPUCODE
namespace GPUCA_NAMESPACE
{
namespace gpu
{
#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
// instantiate version for non-GPU data types
template class GPUTRDTracker_t<GPUTRDTrack, GPUTRDPropagator>;
#endif
// always instantiate version for GPU data types
template class GPUTRDTracker_t<GPUTRDTrackGPU, GPUTRDPropagatorGPU>;
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#endif
