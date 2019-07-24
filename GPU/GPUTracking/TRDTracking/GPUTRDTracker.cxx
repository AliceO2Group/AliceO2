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
#include "GPUReconstruction.h"
#include "GPUMemoryResource.h"
#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"

using namespace GPUCA_NAMESPACE::gpu;

class GPUTPCGMMerger;

#ifndef GPUCA_GPUCODE

#ifndef __OPENCL__
#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <vector>
#endif
#ifdef GPUCA_ALIROOT_LIB
#include "TDatabasePDG.h"
#include "AliMCParticle.h"
#include "AliMCEvent.h"
static const float piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass();
#else
static const float piMass = 0.139f;
#endif

#include "GPUChainTracking.h"

void GPUTRDTracker::SetMaxData()
{
  mNMaxTracks = mChainTracking->mIOPtrs.nMergedTracks;
  mNMaxSpacePoints = mChainTracking->mIOPtrs.nTRDTracklets;
  if (mRec->GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_GLOBAL) {
    mNMaxTracks = 50000;
    mNMaxSpacePoints = 100000;
  }
}

void GPUTRDTracker::RegisterMemoryAllocation()
{
  mMemoryPermanent = mRec->RegisterMemoryAllocation(this, &GPUTRDTracker::SetPointersBase, GPUMemoryResource::MEMORY_PERMANENT, "TRDInitialize");
  mMemoryTracklets = mRec->RegisterMemoryAllocation(this, &GPUTRDTracker::SetPointersTracklets, GPUMemoryResource::MEMORY_INPUT, "TRDTracklets");
  auto type = GPUMemoryResource::MEMORY_INOUT;
  if (mRec->GetDeviceProcessingSettings().memoryAllocationStrategy == GPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    type = GPUMemoryResource::MEMORY_CUSTOM;
  }
  mMemoryTracks = mRec->RegisterMemoryAllocation(this, &GPUTRDTracker::SetPointersTracks, type, "TRDTracks");
}

void GPUTRDTracker::InitializeProcessor() { Init((TRD_GEOMETRY_CONST GPUTRDGeometry*)mChainTracking->GetTRDGeometry()); }

void* GPUTRDTracker::SetPointersBase(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for fixed size objects (needs to be done only once)
  //--------------------------------------------------------------------
  mMaxThreads = mRec->GetMaxThreads();
  computePointerWithAlignment(base, mR, kNLayers);
  computePointerWithAlignment(base, mNTrackletsInChamber, kNChambers);
  computePointerWithAlignment(base, mTrackletIndexArray, kNChambers);
  computePointerWithAlignment(base, mHypothesis, mNCandidates * mMaxThreads);
  computePointerWithAlignment(base, mCandidates, mNCandidates * 2 * mMaxThreads);
  return base;
}

void* GPUTRDTracker::SetPointersTracklets(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for tracklets and space points
  // (size might change for different events)
  //--------------------------------------------------------------------
  computePointerWithAlignment(base, mTracklets, mNMaxSpacePoints);
  computePointerWithAlignment(base, mSpacePoints, mNMaxSpacePoints);
  computePointerWithAlignment(base, mTrackletLabels, 3 * mNMaxSpacePoints);
  return base;
}

void* GPUTRDTracker::SetPointersTracks(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for tracks (this is done once per event)
  //--------------------------------------------------------------------
  computePointerWithAlignment(base, mTracks, mNMaxTracks);
  return base;
}

GPUTRDTracker::GPUTRDTracker()
  : mR(nullptr), mIsInitialized(false), mMemoryPermanent(-1), mMemoryTracklets(-1), mMemoryTracks(-1), mNMaxTracks(0), mNMaxSpacePoints(0), mTracks(nullptr), mNCandidates(1), mNTracks(0), mNEvents(0), mTracklets(nullptr), mMaxThreads(100), mNTracklets(0), mNTrackletsInChamber(nullptr), mTrackletIndexArray(nullptr), mHypothesis(nullptr), mCandidates(nullptr), mSpacePoints(nullptr), mTrackletLabels(nullptr), mGeo(nullptr), mDebugOutput(false), mMinPt(0.6f), mMaxEta(0.84f), mMaxChi2(15.0f), mMaxMissingLy(6), mChi2Penalty(12.0f), mZCorrCoefNRC(1.4f), mMCEvent(nullptr), mMerger(nullptr), mDebug(new GPUTRDTrackerDebug()), mChainTracking(nullptr)
{
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
}

GPUTRDTracker::~GPUTRDTracker()
{
  //--------------------------------------------------------------------
  // Destructor
  //--------------------------------------------------------------------
  delete mDebug;
}

bool GPUTRDTracker::Init(TRD_GEOMETRY_CONST GPUTRDGeometry* geo)
{
  //--------------------------------------------------------------------
  // Initialise tracker
  //--------------------------------------------------------------------
  if (!geo) {
    Error("Init", "TRD geometry must be provided externally");
    return false;
  }

  mGeo = geo;

#ifdef GPUCA_ALIROOT_LIB
  for (int iCandidate = 0; iCandidate < mNCandidates * 2 * mMaxThreads; ++iCandidate) {
    new (&mCandidates[iCandidate]) GPUTRDTrack;
  }
#endif

  for (int iDet = 0; iDet < kNChambers; ++iDet) {
    mNTrackletsInChamber[iDet] = 0;
    mTrackletIndexArray[iDet] = -1;
  }

  // obtain average radius of TRD layers (use default value w/o misalignment if no transformation matrix can be obtained)
  float x0[kNLayers] = { 300.2f, 312.8f, 325.4f, 338.0f, 350.6f, 363.2f };
  for (int iLy = 0; iLy < kNLayers; iLy++) {
    mR[iLy] = x0[iLy];
  }
  auto* matrix = mGeo->GetClusterMatrix(0);
  My_Float loc[3] = { mGeo->AnodePos(), 0.f, 0.f };
  My_Float glb[3] = { 0.f, 0.f, 0.f };
  for (int iLy = 0; iLy < kNLayers; iLy++) {
    for (int iSec = 0; iSec < kNSectors; iSec++) {
      matrix = mGeo->GetClusterMatrix(mGeo->GetDetector(iLy, 2, iSec));
      if (matrix) {
        break;
      }
    }
    if (!matrix) {
      Error("Init", "Could not get transformation matrix for layer %i. Using default x pos instead", iLy);
      continue;
    }
    matrix->LocalToMaster(loc, glb);
    mR[iLy] = glb[0];
  }

  mDebug->ExpandVectors();

  mIsInitialized = true;
  return true;
}

void GPUTRDTracker::Reset(bool fast)
{
  //--------------------------------------------------------------------
  // Reset tracker
  //--------------------------------------------------------------------
  mNTracklets = 0;
  mNTracks = 0;
  if (fast) {
    return;
  }
  for (int i = 0; i < mNMaxSpacePoints; ++i) {
    mTracklets[i] = 0x0;
    mSpacePoints[i].mR = 0.f;
    mSpacePoints[i].mX[0] = 0.f;
    mSpacePoints[i].mX[1] = 0.f;
    mSpacePoints[i].mCov[0] = 0.f;
    mSpacePoints[i].mCov[1] = 0.f;
    mSpacePoints[i].mCov[2] = 0.f;
    mSpacePoints[i].mDy = 0.f;
    mSpacePoints[i].mId = 0;
    mSpacePoints[i].mLabel[0] = -1;
    mSpacePoints[i].mLabel[1] = -1;
    mSpacePoints[i].mLabel[2] = -1;
    mSpacePoints[i].mVolumeId = 0;
  }
  for (int iDet = 0; iDet < kNChambers; ++iDet) {
    mNTrackletsInChamber[iDet] = 0;
    mTrackletIndexArray[iDet] = -1;
  }
}

void GPUTRDTracker::DoTracking()
{
  //--------------------------------------------------------------------
  // Steering function for the tracking
  //--------------------------------------------------------------------

  // sort tracklets and fill index array
  CAAlgo::sort(mTracklets, mTracklets + mNTracklets);
  int trkltCounter = 0;
  for (int iDet = 0; iDet < kNChambers; ++iDet) {
    if (mNTrackletsInChamber[iDet] != 0) {
      mTrackletIndexArray[iDet] = trkltCounter;
      trkltCounter += mNTrackletsInChamber[iDet];
    }
  }

  if (!CalculateSpacePoints()) {
    Error("DoTracking", "Space points for at least one chamber could not be calculated");
  }

  auto timeStart = std::chrono::high_resolution_clock::now();

  if (mRec->GetRecoStepsGPU() & GPUReconstruction::RecoStep::TRDTracking) {
    mChainTracking->DoTRDGPUTracking();
  } else {
#ifdef GPUCA_HAVE_OPENMP
#pragma omp parallel for
    for (int iTrk = 0; iTrk < mNTracks; ++iTrk) {
      if (omp_get_num_threads() > mMaxThreads) {
        Error("DoTracking", "number of parallel threads too high, aborting tracking");
        // break statement not possible in OpenMP for loop
        iTrk = mNTracks;
        continue;
      }
      DoTrackingThread(iTrk, &mChainTracking->GetTPCMerger(), omp_get_thread_num());
    }
#else
    for (int iTrk = 0; iTrk < mNTracks; ++iTrk) {
      DoTrackingThread(iTrk, &mChainTracking->GetTPCMerger());
    }
#endif
  }

  auto duration = std::chrono::high_resolution_clock::now() - timeStart;
  // std::cout << "--->  -----> -------> ---------> Time for event " << mNEvents << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;

  // DumpTracks();
  mNEvents++;
}

void GPUTRDTracker::SetNCandidates(int n)
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

void GPUTRDTracker::PrintSettings() const
{
  //--------------------------------------------------------------------
  // print current settings to screen
  //--------------------------------------------------------------------
  GPUInfo("##############################################################");
  GPUInfo("Current settings for GPU TRD tracker:");
  GPUInfo(" mMaxChi2(%.2f)\n mChi2Penalty(%.2f)\n nCandidates(%i)\n maxMissingLayers(%i)", mMaxChi2, mChi2Penalty, mNCandidates, mMaxMissingLy);
  GPUInfo(" ptCut = %.2f GeV\n abs(eta) < %.2f", mMinPt, mMaxEta);
  GPUInfo("##############################################################");
}

void GPUTRDTracker::CountMatches(const int trackID, std::vector<int>* matches) const
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
    for (int iTrklt = 0; iTrklt < mNTrackletsInChamber[k]; iTrklt++) {
      int trkltIdx = mTrackletIndexArray[k] + iTrklt;
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
        // continue; //FIXME uncomment to count only exact matches
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

GPUd() void GPUTRDTracker::CheckTrackRefs(const int trackID, bool* findableMC) const
{
#ifdef ENABLE_GPUMC
  //--------------------------------------------------------------------
  // loop over all track references for the input trackID and set
  // findableMC to true for each layer in which a track hit exiting
  // the TRD chamber exists
  // (in debug mode)
  //--------------------------------------------------------------------
  TParticle* particle;
  TClonesArray* trackRefs;

  int nHits = mMCEvent->GetParticleAndTR(trackID, particle, trackRefs);
  if (nHits < 1) {
    return;
  }
  int nHitsTrd = 0;
  for (int iHit = 0; iHit < nHits; ++iHit) {
    AliTrackReference* trackReference = static_cast<AliTrackReference*>(trackRefs->UncheckedAt(iHit));
    if (trackReference->DetectorId() != AliTrackReference::kTRD) {
      continue;
    }
    nHitsTrd++;
    float xLoc = trackReference->LocalX();
    // if (!((trackReference->TestBits(0x1 << 18)) || (trackReference->TestBits(0x1 << 17)))) {
    if (!trackReference->TestBits(0x1 << 18)) {
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
      Error("CheckTrackRefs", "No layer can be determined");
      GPUError("x=%f, y=%f, z=%f, layer=%i", xLoc, trackReference->LocalY(), trackReference->Z(), layer);
      continue;
    }
    findableMC[layer] = true;
  }
#endif
}

#endif //! GPUCA_GPUCODE

GPUd() int GPUTRDTracker::LoadTracklet(const GPUTRDTrackletWord& tracklet, const int* labels)
{
  //--------------------------------------------------------------------
  // Add single tracklet to tracker
  //--------------------------------------------------------------------
  if (mNTracklets >= mNMaxSpacePoints) {
    Error("LoadTracklet", "Running out of memory for tracklets, skipping tracklet(s). This should actually never happen.");
    return 1;
  }
  if (labels) {
    for (int i = 0; i < 3; ++i) {
      mTrackletLabels[3 * mNTracklets + i] = labels[i];
    }
  }
  mTracklets[mNTracklets++] = tracklet;
  mNTrackletsInChamber[tracklet.GetDetector()]++;
  return 0;
}

GPUd() void GPUTRDTracker::DumpTracks()
{
  //--------------------------------------------------------------------
  // helper function (only for debugging purposes)
  //--------------------------------------------------------------------
  for (int i = 0; i < mNTracks; ++i) {
    GPUTRDTrack* trk = &(mTracks[i]);
    GPUInfo("track %i: x=%f, alpha=%f, nTracklets=%i, pt=%f", i, trk->getX(), trk->getAlpha(), trk->GetNtracklets(), trk->getPt());
  }
}

GPUd() void GPUTRDTracker::DoTrackingThread(int iTrk, const GPUTPCGMMerger* merger, int threadId)
{
  //--------------------------------------------------------------------
  // perform the tracking for one track (must be threadsafe)
  //--------------------------------------------------------------------
  GPUTRDPropagator prop(merger);
  prop.setTrack(&(mTracks[iTrk]));
  FollowProlongation(&prop, &(mTracks[iTrk]), threadId);
}

GPUd() bool GPUTRDTracker::CalculateSpacePoints()
{
  //--------------------------------------------------------------------
  // Calculates TRD space points in sector tracking coordinates
  // from online tracklets
  //--------------------------------------------------------------------

  bool result = true;

  for (int iDet = 0; iDet < kNChambers; ++iDet) {

    int nTracklets = mNTrackletsInChamber[iDet];
    if (nTracklets == 0) {
      continue;
    }

    auto* matrix = mGeo->GetClusterMatrix(iDet);
    if (!matrix) {
      Error("CalculateSpacePoints", "Invalid TRD cluster matrix, skipping detector  %i", iDet);
      result = false;
      continue;
    }
    const GPUTRDpadPlane* pp = mGeo->GetPadPlane(iDet);
    float tilt = CAMath::Tan(M_PI / 180.f * pp->GetTiltingAngle());
    float t2 = tilt * tilt;      // tan^2 (tilt)
    float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
    float sy2 = 0.1f * 0.1f;     // sigma_rphi^2, currently assume sigma_rphi = 1 mm

    for (int iTrklt = 0; iTrklt < nTracklets; ++iTrklt) {
      int trkltIdx = mTrackletIndexArray[iDet] + iTrklt;
      int trkltZbin = mTracklets[trkltIdx].GetZbin();
      float sz2 = pp->GetRowSize(trkltZbin) * pp->GetRowSize(trkltZbin) / 12.f; // sigma_z = l_pad/sqrt(12) TODO try a larger z error
      My_Float xTrkltDet[3] = { 0.f };                                          // trklt position in chamber coordinates
      My_Float xTrkltSec[3] = { 0.f };                                          // trklt position in sector coordinates
      xTrkltDet[0] = mGeo->AnodePos();
      xTrkltDet[1] = mTracklets[trkltIdx].GetY();
      xTrkltDet[2] = pp->GetRowPos(trkltZbin) - pp->GetRowSize(trkltZbin) / 2.f - pp->GetRowPos(pp->GetNrows() / 2);
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
      // GPUInfo("Space point %i: x=%f, y=%f, z=%f", iTrklt, mSpacePoints[trkltIdx].mR, mSpacePoints[trkltIdx].mX[0], mSpacePoints[trkltIdx].mX[1]);
    }
  }
  return result;
}

GPUd() bool GPUTRDTracker::FollowProlongation(GPUTRDPropagator* prop, GPUTRDTrack* t, int threadId)
{
  //--------------------------------------------------------------------
  // Propagate TPC track layerwise through TRD and pick up closest
  // tracklet(s) on the way
  // -> returns false if prolongation could not be executed fully
  //    or track does not fullfill threshold conditions
  //--------------------------------------------------------------------

  if (!t->CheckNumericalQuality()) {
    return false;
  }

  // only propagate tracks within TRD acceptance
  if (CAMath::Abs(t->getEta()) > mMaxEta) {
    return false;
  }

  // introduce momentum cut on tracks
  if (t->getPt() < mMinPt) {
    return false;
  }

  mDebug->Reset();
  int iTrack = t->GetTPCtrackId();
  t->SetChi2(0.f);
  const GPUTRDpadPlane* pad = nullptr;

#ifdef ENABLE_GPUTRDDEBUG
  GPUTRDTrack trackNoUp(*t);
#endif

  // look for matching tracklets via MC label
  int trackID = t->GetLabel();

#ifdef ENABLE_GPUMC
  std::vector<int> matchAvailableAll[kNLayers]; // all available MC tracklet matches for this track
  if (mDebugOutput && trackID > 0 && mMCEvent) {
    CountMatches(trackID, matchAvailableAll);
    bool findableMC[kNLayers] = { false };
    CheckTrackRefs(trackID, findableMC);
    mDebug->SetFindableMC(findableMC);
  }
#endif

  int candidateIdxOffset = threadId * 2 * mNCandidates;
  int hypothesisIdxOffset = threadId * mNCandidates;

  // set input track to first candidate(s)
  mCandidates[candidateIdxOffset] = *t;
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

      int det[nMaxChambersToSearch] = { -1, -1, -1, -1 }; // TRD chambers to be searched for tracklets

      prop->setTrack(&mCandidates[2 * iCandidate + currIdx]);

      if (mCandidates[2 * iCandidate + currIdx].GetIsStopped()) {
        Hypothesis hypo(mCandidates[2 * iCandidate + currIdx].GetNlayers(), iCandidate, -1, mCandidates[2 * iCandidate + currIdx].GetChi2());
        InsertHypothesis(hypo, nCurrHypothesis, hypothesisIdxOffset);
        isOK = true;
        continue;
      }

      // propagate track to average radius of TRD layer iLayer
      if (!prop->PropagateToX(mR[iLayer], .8f, 2.f)) {
        if (ENABLE_INFO) {
          Info("FollowProlongation", "Track propagation failed for track %i candidate %i in layer %i (pt=%f, x=%f, mR[layer]=%f)", iTrack, iCandidate, iLayer, mCandidates[2 * iCandidate + currIdx].getPt(), mCandidates[2 * iCandidate + currIdx].getX(), mR[iLayer]);
        }
        continue;
      }

      // rotate track in new sector in case of sector crossing
      if (!AdjustSector(prop, &mCandidates[2 * iCandidate + currIdx], iLayer)) {
        if (ENABLE_INFO) {
          Info("FollowProlongation", "Adjusting sector failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        }
        continue;
      }

      // check if track is findable
      if (IsGeoFindable(&mCandidates[2 * iCandidate + currIdx], iLayer, prop->getAlpha())) {
        mCandidates[2 * iCandidate + currIdx].SetIsFindable(iLayer);
      }

      // define search window
      roadY = 7.f * sqrt(mCandidates[2 * iCandidate + currIdx].getSigmaY2() + 0.1f * 0.1f) + 2.f; // add constant to the road for better efficiency
      // roadZ = 7.f * sqrt(mCandidates[2*iCandidate+currIdx].getSigmaZ2() + 9.f * 9.f / 12.f); // take longest pad length
      roadZ = 18.f; // simply twice the longest pad length -> efficiency 99.996%
      //
      if (CAMath::Abs(mCandidates[2 * iCandidate + currIdx].getZ()) - roadZ >= zMaxTRD) {
        if (ENABLE_INFO) {
          Info("FollowProlongation", "Track out of TRD acceptance with z=%f in layer %i (eta=%f)", mCandidates[2 * iCandidate + currIdx].getZ(), iLayer, mCandidates[2 * iCandidate + currIdx].getEta());
        }
        continue;
      }

      // determine chamber(s) to be searched for tracklets
      FindChambersInRoad(&mCandidates[2 * iCandidate + currIdx], roadY, roadZ, iLayer, det, zMaxTRD, prop->getAlpha());

      // track debug information to be stored in case no matching tracklet can be found
      mDebug->SetTrackParameter(mCandidates[2 * iCandidate + currIdx], iLayer);

      // look for tracklets in chamber(s)
      for (int iDet = 0; iDet < nMaxChambersToSearch; iDet++) {
        int currDet = det[iDet];
        if (currDet == -1) {
          continue;
        }
        int currSec = mGeo->GetSector(currDet);
        if (currSec != GetSector(prop->getAlpha())) {
          float currAlpha = GetAlphaOfSector(currSec);
          if (!prop->rotate(currAlpha)) {
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
        // first propagate track to x of tracklet
        for (int iTrklt = 0; iTrklt < mNTrackletsInChamber[currDet]; ++iTrklt) {
          int trkltIdx = mTrackletIndexArray[currDet] + iTrklt;
          if (!prop->PropagateToX(mSpacePoints[trkltIdx].mR, .8f, 2.f)) {
            if (ENABLE_WARNING) {
              Warning("FollowProlongation", "Track parameter for track %i, x=%f at tracklet %i x=%f in layer %i cannot be retrieved", iTrack, mCandidates[2 * iCandidate + currIdx].getX(), iTrklt, mSpacePoints[trkltIdx].mR, iLayer);
            }
            continue;
          }
          // correction for tilted pads (only applied if deltaZ < l_pad && track z err << l_pad)
          float tiltCorr = tilt * (mSpacePoints[trkltIdx].mX[1] - mCandidates[2 * iCandidate + currIdx].getZ());
          float l_pad = pad->GetRowSize(mTracklets[trkltIdx].GetZbin());
          if (!((CAMath::Abs(mSpacePoints[trkltIdx].mX[1] - mCandidates[2 * iCandidate + currIdx].getZ()) < l_pad) && (mCandidates[2 * iCandidate + currIdx].getSigmaZ2() < (l_pad * l_pad / 12.f)))) {
            tiltCorr = 0.f;
          }
          // correction for mean z position of tracklet (is not the center of the pad if track eta != 0)
          float zPosCorr = mSpacePoints[trkltIdx].mX[1] + mZCorrCoefNRC * mCandidates[2 * iCandidate + currIdx].getTgl();
          float yPosCorr = mSpacePoints[trkltIdx].mX[0] - tiltCorr;
          float deltaY = yPosCorr - mCandidates[2 * iCandidate + currIdx].getY();
          float deltaZ = zPosCorr - mCandidates[2 * iCandidate + currIdx].getZ();
          My_Float trkltPosTmpYZ[2] = { yPosCorr, zPosCorr };
          My_Float trkltCovTmp[3] = { 0.f };
          if ((CAMath::Abs(deltaY) < roadY) && (CAMath::Abs(deltaZ) < roadZ)) {
            // tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            RecalcTrkltCov(tilt, mCandidates[2 * iCandidate + currIdx].getSnp(), pad->GetRowSize(mTracklets[trkltIdx].GetZbin()), trkltCovTmp);
            float chi2 = prop->getPredictedChi2(trkltPosTmpYZ, trkltCovTmp);
            // GPUInfo("layer %i: chi2 = %f", iLayer, chi2);
            if (chi2 < mMaxChi2) {
              Hypothesis hypo(mCandidates[2 * iCandidate + currIdx].GetNlayers(), iCandidate, trkltIdx, mCandidates[2 * iCandidate + currIdx].GetChi2() + chi2);
              InsertHypothesis(hypo, nCurrHypothesis, hypothesisIdxOffset);
            } // end tracklet chi2 < mMaxChi2
          }   // end tracklet in window
        }     // tracklet loop
      }       // chamber loop

      // add no update to hypothesis list
      Hypothesis hypoNoUpdate(mCandidates[2 * iCandidate + currIdx].GetNlayers(), iCandidate, -1, mCandidates[2 * iCandidate + currIdx].GetChi2() + mChi2Penalty);
      InsertHypothesis(hypoNoUpdate, nCurrHypothesis, hypothesisIdxOffset);
      isOK = true;
    } // end candidate loop

#ifdef ENABLE_GPUMC
    // in case matching tracklet exists in this layer -> store position information for debugging
    if (matchAvailableAll[iLayer].size() > 0 && mDebugOutput) {
      mDebug->SetNmatchAvail(matchAvailableAll[iLayer].size(), iLayer);
      int realTrkltId = matchAvailableAll[iLayer].at(0);
      prop->setTrack(&mCandidates[currIdx]);
      bool flag = prop->PropagateToX(mSpacePoints[realTrkltId].mR, .8f, 2.f);
      if (flag) {
        flag = AdjustSector(prop, &mCandidates[currIdx], iLayer);
      }
      if (!flag) {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Track parameter at x=%f for track %i at real tracklet x=%f in layer %i cannot be retrieved (pt=%f)", mCandidates[currIdx].getX(), iTrack, mSpacePoints[realTrkltId].mR, iLayer, mCandidates[currIdx].getPt());
        }
      } else {
        mDebug->SetTrackParameterReal(mCandidates[currIdx], iLayer);
        float zPosCorrReal = mSpacePoints[realTrkltId].mX[1] + mZCorrCoefNRC * mCandidates[currIdx].getTgl();
        float deltaZReal = zPosCorrReal - mCandidates[currIdx].getZ();
        float tiltCorrReal = tilt * (mSpacePoints[realTrkltId].mX[1] - mCandidates[currIdx].getZ());
        float l_padReal = pad->GetRowSize(mTracklets[realTrkltId].GetZbin());
        if ((mCandidates[currIdx].getSigmaZ2() >= (l_padReal * l_padReal / 12.f)) || (CAMath::Abs(mSpacePoints[realTrkltId].mX[1] - mCandidates[currIdx].getZ()) >= l_padReal)) {
          tiltCorrReal = 0;
        }
        My_Float yzPosReal[2] = { mSpacePoints[realTrkltId].mX[0] - tiltCorrReal, zPosCorrReal };
        My_Float covReal[3] = { 0. };
        RecalcTrkltCov(tilt, mCandidates[currIdx].getSnp(), pad->GetRowSize(mTracklets[realTrkltId].GetZbin()), covReal);
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
      mCandidates[2 * iUpdate + nextIdx] = mCandidates[2 * mHypothesis[iUpdate + hypothesisIdxOffset].mCandidateId + currIdx];
      if (mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId == -1) {
        // no matching tracklet found
        if (mCandidates[2 * iUpdate + nextIdx].GetIsFindable(iLayer)) {
          if (mCandidates[2 * iUpdate + nextIdx].GetNmissingConsecLayers(iLayer) > mMaxMissingLy) {
            mCandidates[2 * iUpdate + nextIdx].SetIsStopped();
          }
          mCandidates[2 * iUpdate + nextIdx].SetChi2(mCandidates[2 * iUpdate + nextIdx].GetChi2() + mChi2Penalty);
        }
        if (iUpdate == 0) {
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }
      // matching tracklet found
      prop->setTrack(&mCandidates[2 * iUpdate + nextIdx]);
      int trkltSec = mGeo->GetSector(mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector());
      if (trkltSec != GetSector(prop->getAlpha())) {
        // if after a matching tracklet was found another sector was searched for tracklets the track needs to be rotated back
        prop->rotate(GetAlphaOfSector(trkltSec));
      }
      if (!prop->PropagateToX(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mR, .8f, 2.f)) {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Final track propagation for track %i update %i in layer %i failed", iTrack, iUpdate, iLayer);
        }
        mCandidates[2 * iUpdate + nextIdx].SetChi2(mCandidates[2 * iUpdate + nextIdx].GetChi2() + mChi2Penalty);
        if (mCandidates[2 * iUpdate + nextIdx].GetIsFindable(iLayer)) {
          if (mCandidates[2 * iUpdate + nextIdx].GetNmissingConsecLayers(iLayer) >= mMaxMissingLy) {
            mCandidates[2 * iUpdate + nextIdx].SetIsStopped();
          }
        }
        if (iUpdate == 0) {
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }

      float tiltCorrUp = tilt * (mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[1] - mCandidates[2 * iUpdate + nextIdx].getZ());
      float zPosCorrUp = mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[1] + mZCorrCoefNRC * mCandidates[2 * iUpdate + nextIdx].getTgl();
      float l_padTrklt = pad->GetRowSize(mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetZbin());
      if (!((mCandidates[2 * iUpdate + nextIdx].getSigmaZ2() < (l_padTrklt * l_padTrklt / 12.f)) && (CAMath::Abs(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[1] - mCandidates[2 * iUpdate + nextIdx].getZ()) < l_padTrklt))) {
        tiltCorrUp = 0.f;
      }
      My_Float trkltPosUp[2] = { mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX[0] - tiltCorrUp, zPosCorrUp };
      My_Float trkltCovUp[3] = { 0.f };
      RecalcTrkltCov(tilt, mCandidates[2 * iUpdate + nextIdx].getSnp(), pad->GetRowSize(mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetZbin()), trkltCovUp);

#ifdef ENABLE_GPUTRDDEBUG
      prop->setTrack(&trackNoUp);
      prop->rotate(GetAlphaOfSector(trkltSec));
      prop->PropagateToX(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mR, .8f, 2.f);
      prop->PropagateToX(mR[iLayer], .8f, 2.f);
      prop->setTrack(&mCandidates[2 * iUpdate + nextIdx]);
#endif

      if (!wasTrackStored) {
#ifdef ENABLE_GPUTRDDEBUG
        mDebug->SetTrackParameterNoUp(trackNoUp, iLayer);
#endif
        mDebug->SetTrackParameter(mCandidates[2 * iUpdate + nextIdx], iLayer);
        mDebug->SetRawTrackletPosition(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mR, mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mX, iLayer);
        mDebug->SetCorrectedTrackletPosition(trkltPosUp, iLayer);
        mDebug->SetTrackletCovariance(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mCov, iLayer);
        mDebug->SetTrackletProperties(mSpacePoints[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].mDy, mTracklets[mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId].GetDetector(), iLayer);
        mDebug->SetRoad(roadY, roadZ, iLayer);
        wasTrackStored = true;
      }

      if (!prop->update(trkltPosUp, trkltCovUp)) {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);
        }
        mCandidates[2 * iUpdate + nextIdx].SetChi2(mCandidates[2 * iUpdate + nextIdx].GetChi2() + mChi2Penalty);
        if (mCandidates[2 * iUpdate + nextIdx].GetIsFindable(iLayer)) {
          if (mCandidates[2 * iUpdate + nextIdx].GetNmissingConsecLayers(iLayer) >= mMaxMissingLy) {
            mCandidates[2 * iUpdate + nextIdx].SetIsStopped();
          }
        }
        if (iUpdate == 0) {
          *t = mCandidates[2 * iUpdate + nextIdx];
        }
        continue;
      }
      if (!mCandidates[2 * iUpdate + nextIdx].CheckNumericalQuality()) {
        if (ENABLE_WARNING) {
          Info("FollowProlongation", "Track %i has invalid covariance matrix. Aborting track following\n", iTrack);
        }
        return false;
      }
      mCandidates[2 * iUpdate + nextIdx].AddTracklet(iLayer, mHypothesis[iUpdate + hypothesisIdxOffset].mTrackletId);
      mCandidates[2 * iUpdate + nextIdx].SetChi2(mHypothesis[iUpdate + hypothesisIdxOffset].mChi2);
      mCandidates[2 * iUpdate + nextIdx].SetIsFindable(iLayer);
      if (iUpdate == 0) {
        *t = mCandidates[2 * iUpdate + nextIdx];
      }
    } // end update loop

    if (!isOK) {
      if (ENABLE_INFO) {
        Info("FollowProlongation", "Track %i cannot be followed. Stopped in layer %i", iTrack, iLayer);
      }
      return false;
    }
  } // end layer loop

  // --------------------------------------------------------------------------------
  // add some debug information (compare labels of attached tracklets to track label)
  // and store full track information
  // --------------------------------------------------------------------------------
  if (mDebugOutput) {
    int update[6] = { 0 };
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

  return true;
}

GPUd() void GPUTRDTracker::InsertHypothesis(Hypothesis hypo, int& nCurrHypothesis, int idxOffset)
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

GPUd() int GPUTRDTracker::GetDetectorNumber(const float zPos, const float alpha, const int layer) const
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

GPUd() bool GPUTRDTracker::AdjustSector(GPUTRDPropagator* prop, GPUTRDTrack* t, const int layer) const
{
  //--------------------------------------------------------------------
  // rotate track in new sector if necessary and
  // propagate to correct x of layer
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
    if (!prop->rotate(alphaCurr + alpha * sign)) {
      return false;
    }
    if (!prop->PropagateToX(xTmp, .8f, 2.f)) {
      return false;
    }
    y = t->getY();
    ++nTries;
  }
  return true;
}

GPUd() int GPUTRDTracker::GetSector(float alpha) const
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

GPUd() float GPUTRDTracker::GetAlphaOfSector(const int sec) const
{
  //--------------------------------------------------------------------
  // rotation angle for TRD sector sec
  //--------------------------------------------------------------------
  return (2.0f * M_PI / (float)kNSectors * ((float)sec + 0.5f));
}

GPUd() void GPUTRDTracker::RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3])
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

GPUd() void GPUTRDTracker::FindChambersInRoad(const GPUTRDTrack* t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const
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
    if ((t->getZ() + roadZ) > pp->GetRowPos(0) || (t->getZ() - roadZ) < pp->GetRowPos(lastPadRow)) {
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

GPUd() bool GPUTRDTracker::IsGeoFindable(const GPUTRDTrack* t, const int layer, const float alpha) const
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
