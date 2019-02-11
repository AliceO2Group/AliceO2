//#define ENABLE_GPUTRDDEBUG
#define ENABLE_WARNING 0
#define ENABLE_INFO 0
#ifdef GPUCA_ALIROOT_LIB
#define ENABLE_GPUMC
#endif

#ifndef __OPENCL__
#ifdef GPUCA_HAVE_OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <vector>
#include <algorithm>
#endif

#include "AliGPUTRDTracker.h"
#include "AliGPUTRDTrackletWord.h"
#include "AliGPUTRDGeometry.h"
#include "AliGPUTRDTrackerDebug.h"
#include "AliGPUReconstruction.h"
#include "AliGPUMemoryResource.h"
#include "AliTPCCommonMath.h"

class AliGPUTPCGMMerger;

#ifdef GPUCA_ALIROOT_LIB
#include "TDatabasePDG.h"
#include "AliMCParticle.h"
#include "AliMCEvent.h"
static const float piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass();
#else
static const float piMass = 0.139f;
#endif

#ifndef GPUCA_GPUCODE
void AliGPUTRDTracker::SetMaxData()
{
  fNMaxTracks = mRec->mIOPtrs.nMergedTracks;
  fNMaxSpacePoints = mRec->mIOPtrs.nTRDTracklets;
  if (mRec->GetDeviceProcessingSettings().memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_GLOBAL)
  {
    fNMaxTracks = 50000;
    fNMaxSpacePoints = 100000;
  }
}

void AliGPUTRDTracker::RegisterMemoryAllocation()
{
  fMemoryPermanent = mRec->RegisterMemoryAllocation(this, &AliGPUTRDTracker::SetPointersBase, AliGPUMemoryResource::MEMORY_PERMANENT, "TRDInitialize");
  fMemoryTracklets = mRec->RegisterMemoryAllocation(this, &AliGPUTRDTracker::SetPointersTracklets, AliGPUMemoryResource::MEMORY_INPUT, "TRDTracklets");
  auto type = AliGPUMemoryResource::MEMORY_INOUT;
  if (mRec->GetDeviceProcessingSettings().memoryAllocationStrategy == AliGPUMemoryResource::ALLOCATION_INDIVIDUAL) {
    type = AliGPUMemoryResource::MEMORY_CUSTOM;
  }
  fMemoryTracks = mRec->RegisterMemoryAllocation(this, &AliGPUTRDTracker::SetPointersTracks, type, "TRDTracks");
}

void AliGPUTRDTracker::InitializeProcessor()
{
  Init((TRD_GEOMETRY_CONST AliGPUTRDGeometry*) mRec->GetTRDGeometry());
}

void* AliGPUTRDTracker::SetPointersBase(void* base)
{
  //--------------------------------------------------------------------
  // Allocate memory for fixed size objects (needs to be done only once)
  //--------------------------------------------------------------------
  fMaxThreads = mRec->GetMaxThreads();
  computePointerWithAlignment(base, fR, kNLayers);
  computePointerWithAlignment(base, fNtrackletsInChamber, kNChambers);
  computePointerWithAlignment(base, fTrackletIndexArray, kNChambers);
  computePointerWithAlignment(base, fHypothesis, fNhypothesis * fMaxThreads);
  computePointerWithAlignment(base, fCandidates, fNCandidates * 2 * fMaxThreads);
  return base;
}

void* AliGPUTRDTracker::SetPointersTracklets(void *base)
{
  //--------------------------------------------------------------------
  // Allocate memory for tracklets and space points
  // (size might change for different events)
  //--------------------------------------------------------------------
  computePointerWithAlignment(base, fTracklets, fNMaxSpacePoints);
  computePointerWithAlignment(base, fSpacePoints, fNMaxSpacePoints);
  computePointerWithAlignment(base, fTrackletLabels, 3*fNMaxSpacePoints);
  return base;
}

void* AliGPUTRDTracker::SetPointersTracks(void *base)
{
  //--------------------------------------------------------------------
  // Allocate memory for tracks (this is done once per event)
  //--------------------------------------------------------------------
  computePointerWithAlignment(base, fTracks, fNMaxTracks);
  return base;
}


AliGPUTRDTracker::AliGPUTRDTracker() :
  fR(nullptr),
  fIsInitialized(false),
  fMemoryPermanent(-1),
  fMemoryTracklets(-1),
  fMemoryTracks(-1),
  fNMaxTracks(0),
  fNMaxSpacePoints(0),
  fTracks(nullptr),
  fNCandidates(1),
  fNTracks(0),
  fNEvents(0),
  fTracklets(nullptr),
  fMaxThreads(100),
  fNTracklets(0),
  fNtrackletsInChamber(nullptr),
  fTrackletIndexArray(nullptr),
  fHypothesis(nullptr),
  fCandidates(nullptr),
  fSpacePoints(nullptr),
  fTrackletLabels(nullptr),
  fGeo(nullptr),
  fDebugOutput(false),
  fMinPt(0.6f),
  fMaxEta(0.84f),
  fMaxChi2(15.0f),
  fMaxMissingLy(6),
  fChi2Penalty(12.0f),
  fZCorrCoefNRC(1.4f),
  fNhypothesis(100),
  fMCEvent(nullptr),
  fDebug(new AliGPUTRDTrackerDebug())
{
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
}

AliGPUTRDTracker::~AliGPUTRDTracker()
{
  //--------------------------------------------------------------------
  // Destructor
  //--------------------------------------------------------------------
  delete fDebug;
}

bool AliGPUTRDTracker::Init(TRD_GEOMETRY_CONST AliGPUTRDGeometry *geo)
{
  //--------------------------------------------------------------------
  // Initialise tracker
  //--------------------------------------------------------------------
  if (!geo) {
    Error("Init", "TRD geometry must be provided externally");
    return false;
  }

  fGeo = geo;

#ifdef GPUCA_ALIROOT_LIB
  for (int iCandidate = 0; iCandidate < fNCandidates * 2 * fMaxThreads; ++iCandidate) {
    new(&fCandidates[iCandidate]) GPUTRDTrack;
  }
#endif

  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fNtrackletsInChamber[iDet] = 0;
    fTrackletIndexArray[iDet] = -1;
  }

  // obtain average radius of TRD layers (use default value w/o misalignment if no transformation matrix can be obtained)
  float x0[kNLayers]    = { 300.2f, 312.8f, 325.4f, 338.0f, 350.6f, 363.2f };
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = x0[iLy];
  }
  auto* matrix = fGeo->GetClusterMatrix(0);
  My_Float loc[3] = { fGeo->AnodePos(), 0.f, 0.f };
  My_Float glb[3] = { 0.f, 0.f, 0.f };
  for (int iLy=0; iLy<kNLayers; iLy++) {
    for (int iSec=0; iSec<kNSectors; iSec++) {
      matrix = fGeo->GetClusterMatrix(fGeo->GetDetector(iLy, 2, iSec));
      if (matrix) {
        break;
      }
    }
    if (!matrix) {
      Error("Init", "Could not get transformation matrix for layer %i. Using default x pos instead", iLy);
      continue;
    }
    matrix->LocalToMaster(loc, glb);
    fR[iLy] = glb[0];
  }

  fDebug->ExpandVectors();

  fIsInitialized = true;
  return true;
}

void AliGPUTRDTracker::Reset()
{
  //--------------------------------------------------------------------
  // Reset tracker
  //--------------------------------------------------------------------
  fNTracklets = 0;
  fNTracks = 0;
  for (int i=0; i<fNMaxSpacePoints; ++i) {
    fTracklets[i] = 0x0;
    fSpacePoints[i].fR        = 0.f;
    fSpacePoints[i].fX[0]     = 0.f;
    fSpacePoints[i].fX[1]     = 0.f;
    fSpacePoints[i].fCov[0]   = 0.f;
    fSpacePoints[i].fCov[1]   = 0.f;
    fSpacePoints[i].fCov[2]   = 0.f;
    fSpacePoints[i].fDy       = 0.f;
    fSpacePoints[i].fId       = 0;
    fSpacePoints[i].fLabel[0] = -1;
    fSpacePoints[i].fLabel[1] = -1;
    fSpacePoints[i].fLabel[2] = -1;
    fSpacePoints[i].fVolumeId = 0;
  }
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fNtrackletsInChamber[iDet] = 0;
    fTrackletIndexArray[iDet] = -1;
  }
}

void AliGPUTRDTracker::DoTracking()
{
  //--------------------------------------------------------------------
  // Steering function for the tracking
  //--------------------------------------------------------------------

  // sort tracklets and fill index array
  Quicksort(0, fNTracklets - 1, fNTracklets);
  int trkltCounter = 0;
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    if (fNtrackletsInChamber[iDet] != 0) {
      fTrackletIndexArray[iDet] = trkltCounter;
      trkltCounter += fNtrackletsInChamber[iDet];
    }
  }

  if (!CalculateSpacePoints()) {
    Error("DoTracking", "Space points for at least one chamber could not be calculated");
  }

  auto timeStart = std::chrono::high_resolution_clock::now();

  if (mRec->GetRecoStepsGPU() & AliGPUReconstruction::RecoStep::TRDTracking)
  {
    mRec->DoTRDGPUTracking();
  }
  else
  {
#ifdef GPUCA_HAVE_OPENMP
#pragma omp parallel for
  for (int iTrk=0; iTrk<fNTracks; ++iTrk) {
    if (omp_get_num_threads() > fMaxThreads) {
      Error("DoTracking", "number of parallel threads too high, aborting tracking");
      // break statement not possible in OpenMP for loop
      iTrk = fNTracks;
      continue;
    }
    DoTrackingThread(iTrk, &mRec->GetTPCMerger(), omp_get_thread_num());
  }
#else
  for (int iTrk=0; iTrk<fNTracks; ++iTrk) {
    DoTrackingThread(iTrk, &mRec->GetTPCMerger());
  }
#endif
  }

  auto duration = std::chrono::high_resolution_clock::now() - timeStart;
  //std::cout << "--->  -----> -------> ---------> Time for event " << fNEvents << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;

  //DumpTracks();
  fNEvents++;
}

void AliGPUTRDTracker::SetNCandidates(int n)
{
  //--------------------------------------------------------------------
  // set the number of candidates to be used
  //--------------------------------------------------------------------
  if (!fIsInitialized) {
    fNCandidates = n;
  } else {
    Error("SetNCandidates", "Cannot change fNCandidates after initialization");
  }
}

void AliGPUTRDTracker::PrintSettings() const
{
  //--------------------------------------------------------------------
  // print current settings to screen
  //--------------------------------------------------------------------
  printf("##############################################################\n");
  printf("Current settings for GPU TRD tracker:\n");
  printf(" fMaxChi2(%.2f)\n fChi2Penalty(%.2f)\n nCandidates(%i)\n nHypothesisMax(%i)\n maxMissingLayers(%i)\n",
          fMaxChi2, fChi2Penalty, fNCandidates, fNhypothesis, fMaxMissingLy);
  printf(" ptCut = %.2f GeV\n abs(eta) < %.2f\n", fMinPt, fMaxEta);
  printf("##############################################################\n");
}

void AliGPUTRDTracker::CountMatches(const int trackID, std::vector<int> *matches) const
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
    int layer = fGeo->GetLayer(k);
    for (int iTrklt = 0; iTrklt < fNtrackletsInChamber[k]; iTrklt++) {
      int trkltIdx = fTrackletIndexArray[k] + iTrklt;
      bool trkltStored = false;
      for (int il=0; il<3; il++) {
        int lb = fSpacePoints[trkltIdx].fLabel[il];
        if (lb<0) {
          // no more valid labels
          break;
        }
        if (lb == CAMath::Abs(trackID)) {
          matches[layer].push_back(trkltIdx);
          break;
        }
        if (!fMCEvent) {
          continue;
        }
        //continue; //FIXME uncomment to count only exact matches
        AliMCParticle *mcPart = (AliMCParticle*) fMCEvent->GetTrack(lb);
        while (mcPart) {
          lb = mcPart->GetMother();
          if (lb == CAMath::Abs(trackID)) {
            matches[layer].push_back(trkltIdx);
            trkltStored = true;
            break;
          }
          mcPart = lb >= 0 ? (AliMCParticle*) fMCEvent->GetTrack(lb) : 0;
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

GPUd() void AliGPUTRDTracker::CheckTrackRefs(const int trackID, bool *findableMC) const
{
#ifdef ENABLE_GPUMC
  //--------------------------------------------------------------------
  // loop over all track references for the input trackID and set
  // findableMC to true for each layer in which a track hit exiting
  // the TRD chamber exists
  // (in debug mode)
  //--------------------------------------------------------------------
  TParticle *particle;
  TClonesArray *trackRefs;

  int nHits = fMCEvent->GetParticleAndTR(trackID, particle, trackRefs);
  if (nHits < 1) {
    return;
  }
  int nHitsTrd = 0;
  for (int iHit = 0; iHit < nHits; ++iHit) {
    AliTrackReference *trackReference = static_cast<AliTrackReference*>(trackRefs->UncheckedAt(iHit));
    if (trackReference->DetectorId() != AliTrackReference::kTRD) {
      continue;
    }
    nHitsTrd++;
    float xLoc = trackReference->LocalX();
    //if (!((trackReference->TestBits(0x1 << 18)) || (trackReference->TestBits(0x1 << 17)))) {
    if (!trackReference->TestBits(0x1 << 18)) {
      // bit 17 - entering; bit 18 - exiting
      continue;
    }
    int layer = -1;
    if (xLoc < 304.f) {
      layer = 0;
    }
    else if (xLoc < 317.f) {
      layer = 1;
    }
    else if (xLoc < 330.f) {
      layer = 2;
    }
    else if (xLoc < 343.f) {
      layer = 3;
    }
    else if (xLoc < 356.f) {
      layer = 4;
    }
    else if (xLoc < 369.f) {
      layer = 5;
    }
    if (layer < 0) {
      Error("CheckTrackRefs", "No layer can be determined");
      printf("x=%f, y=%f, z=%f, layer=%i\n", xLoc, trackReference->LocalY(), trackReference->Z(), layer);
      continue;
    }
    findableMC[layer] = true;
  }
#endif
}

#endif //!GPUCA_GPUCODE

GPUd() int AliGPUTRDTracker::LoadTracklet(const AliGPUTRDTrackletWord &tracklet, const int *labels)
{
  //--------------------------------------------------------------------
  // Add single tracklet to tracker
  //--------------------------------------------------------------------
  if (fNTracklets >= fNMaxSpacePoints ) {
    Error("LoadTracklet", "Running out of memory for tracklets, skipping tracklet(s). This should actually never happen.");
    return 1;
  }
  if (labels) {
    for (int i=0; i<3; ++i) {
      fTrackletLabels[3*fNTracklets+i] = labels[i];
    }
  }
  fTracklets[fNTracklets++] = tracklet;
  fNtrackletsInChamber[tracklet.GetDetector()]++;
  return 0;
}

GPUd() void AliGPUTRDTracker::DumpTracks()
{
  //--------------------------------------------------------------------
  // helper function (only for debugging purposes)
  //--------------------------------------------------------------------
  for (int i=0; i<fNTracks; ++i) {
    GPUTRDTrack *trk = &(fTracks[i]);
    printf("track %i: x=%f, alpha=%f, nTracklets=%i, pt=%f\n", i, trk->getX(), trk->getAlpha(), trk->GetNtracklets(), trk->getPt());
  }
}

GPUd() void AliGPUTRDTracker::DoTrackingThread(int iTrk, const AliGPUTPCGMMerger* merger, int threadId)
{
  //--------------------------------------------------------------------
  // perform the tracking for one track (must be threadsafe)
  //--------------------------------------------------------------------
  GPUTRDPropagator prop(merger);
  prop.setTrack(&(fTracks[iTrk]));
  FollowProlongation(&prop, &(fTracks[iTrk]), threadId);
}


GPUd() bool AliGPUTRDTracker::CalculateSpacePoints()
{
  //--------------------------------------------------------------------
  // Calculates TRD space points in sector tracking coordinates
  // from online tracklets
  //--------------------------------------------------------------------

  bool result = true;

  for (int iDet=0; iDet<kNChambers; ++iDet) {

    int nTracklets = fNtrackletsInChamber[iDet];
    if (nTracklets == 0) {
      continue;
    }

    auto* matrix = fGeo->GetClusterMatrix(iDet);
    if (!matrix){
      Error("CalculateSpacePoints", "Invalid TRD cluster matrix, skipping detector  %i", iDet);
      result = false;
      continue;
    }
    const AliGPUTRDpadPlane *pp = fGeo->GetPadPlane(iDet);
    float tilt = CAMath::Tan( M_PI / 180.f * pp->GetTiltingAngle());
    float t2 = tilt * tilt; // tan^2 (tilt)
    float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
    float sy2 = 0.1f * 0.1f; // sigma_rphi^2, currently assume sigma_rphi = 1 mm

    for (int iTrklt=0; iTrklt<nTracklets; ++iTrklt) {
      int trkltIdx = fTrackletIndexArray[iDet] + iTrklt;
      int trkltZbin = fTracklets[trkltIdx].GetZbin();
      float sz2 = pp->GetRowSize(trkltZbin) * pp->GetRowSize(trkltZbin) / 12.f; // sigma_z = l_pad/sqrt(12) TODO try a larger z error
      My_Float xTrkltDet[3] = { 0.f }; // trklt position in chamber coordinates
      My_Float xTrkltSec[3] = { 0.f }; // trklt position in sector coordinates
      xTrkltDet[0] = fGeo->AnodePos();
      xTrkltDet[1] = fTracklets[trkltIdx].GetY();
      xTrkltDet[2] = pp->GetRowPos(trkltZbin) - pp->GetRowSize(trkltZbin)/2.f - pp->GetRowPos(pp->GetNrows()/2);
      matrix->LocalToMaster(xTrkltDet, xTrkltSec);
      fSpacePoints[trkltIdx].fR = xTrkltSec[0];
      fSpacePoints[trkltIdx].fX[0] = xTrkltSec[1];
      fSpacePoints[trkltIdx].fX[1] = xTrkltSec[2];
      fSpacePoints[trkltIdx].fId = fTracklets[trkltIdx].GetId();
      for (int i=0; i<3; i++) {
        fSpacePoints[trkltIdx].fLabel[i] = fTrackletLabels[3*fTracklets[trkltIdx].GetId() + i];
      }
      fSpacePoints[trkltIdx].fCov[0] = c2 * (sy2 + t2 * sz2);
      fSpacePoints[trkltIdx].fCov[1] = c2 * tilt * (sz2 - sy2);
      fSpacePoints[trkltIdx].fCov[2] = c2 * (t2 * sy2 + sz2);
      fSpacePoints[trkltIdx].fDy = 0.014f * fTracklets[trkltIdx].GetdY();

      int modId   = fGeo->GetSector(iDet) * AliGPUTRDGeometry::kNstack + fGeo->GetStack(iDet); // global TRD stack number
      unsigned short volId = fGeo->GetGeomManagerVolUID(iDet, modId);
      fSpacePoints[trkltIdx].fVolumeId = volId;
      //printf("Space point %i: x=%f, y=%f, z=%f\n", iTrklt, fSpacePoints[trkltIdx].fR, fSpacePoints[trkltIdx].fX[0], fSpacePoints[trkltIdx].fX[1]);
    }
  }
  return result;
}


GPUd() bool AliGPUTRDTracker::FollowProlongation(GPUTRDPropagator *prop, GPUTRDTrack *t, int threadId)
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
  if (CAMath::Abs(t->getEta()) > fMaxEta) {
    return false;
  }

  // introduce momentum cut on tracks
  if (t->getPt() < fMinPt) {
    return false;
  }

  fDebug->Reset();
  int iTrack = t->GetTPCtrackId();
  t->SetChi2(0.f);
  const AliGPUTRDpadPlane *pad = nullptr;

#ifdef ENABLE_GPUTRDDEBUG
  GPUTRDTrack trackNoUp(*t);
#endif

  // look for matching tracklets via MC label
  int trackID = t->GetLabel();

#ifdef ENABLE_GPUMC
  std::vector<int> matchAvailableAll[kNLayers]; // all available MC tracklet matches for this track
  if (fDebugOutput && trackID > 0 && fMCEvent) {
    CountMatches(trackID, matchAvailableAll);
    bool findableMC[kNLayers] = { false };
    CheckTrackRefs(trackID, findableMC);
    fDebug->SetFindableMC(findableMC);
  }
#endif

  int candidateIdxOffset = threadId * 2 * fNCandidates;
  int hypothesisIdxOffset = threadId * fNhypothesis;

  // set input track to first candidate(s)
  fCandidates[candidateIdxOffset] = *t;
  int nCandidates = 1;

  // search window
  float roadY = 0.f;
  float roadZ = 0.f;
  const int nMaxChambersToSearch = 4;

  fDebug->SetGeneralInfo(fNEvents, fNTracks, iTrack, trackID, t->getPt());

  for (int iLayer=0; iLayer<kNLayers; ++iLayer) {
    int nCurrHypothesis = 0;
    bool isOK = false; // if at least one candidate could be propagated or the track was stopped this becomes true
    int currIdx = candidateIdxOffset + iLayer % 2;
    int nextIdx = candidateIdxOffset + (iLayer + 1) % 2;
    pad = fGeo->GetPadPlane(iLayer, 0);
    float tilt = CAMath::Tan( M_PI / 180.f * pad->GetTiltingAngle()); // tilt is signed!
    const float zMaxTRD = pad->GetRow0();

    // --------------------------------------------------------------------------------
    //
    // for each candidate, propagate to layer radius and look for close tracklets
    //
    // --------------------------------------------------------------------------------
    for (int iCandidate=0; iCandidate<nCandidates; iCandidate++) {

      int det[nMaxChambersToSearch] = { -1, -1, -1, -1 }; // TRD chambers to be searched for tracklets

      prop->setTrack(&fCandidates[2*iCandidate+currIdx]);

      if (fCandidates[2*iCandidate+currIdx].GetIsStopped()) {
        if (nCurrHypothesis < fNhypothesis) {
          fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2();
          fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
          fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fTrackletId = -1;
          nCurrHypothesis++;
        }
        else {
          Quicksort(hypothesisIdxOffset, nCurrHypothesis - 1 + hypothesisIdxOffset, nCurrHypothesis, 1);
          if ( fCandidates[2*iCandidate+currIdx].GetReducedChi2() <
               (fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fChi2 / CAMath::Max(fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fLayers, 1)) ) {
            fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2();
            fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
            fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fCandidateId = iCandidate;
            fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fTrackletId = -1;
          }
        }
        isOK = true;
        continue;
      }

      // propagate track to average radius of TRD layer iLayer
      if (!prop->PropagateToX(fR[iLayer], .8f, 2.f)) {
        if (ENABLE_INFO) {
          Info("FollowProlongation", "Track propagation failed for track %i candidate %i in layer %i (pt=%f, x=%f, fR[layer]=%f)",
            iTrack, iCandidate, iLayer, fCandidates[2*iCandidate+currIdx].getPt(), fCandidates[2*iCandidate+currIdx].getX(), fR[iLayer]);
        }
        continue;
      }

      // rotate track in new sector in case of sector crossing
      if (!AdjustSector(prop, &fCandidates[2*iCandidate+currIdx], iLayer)) {
        if (ENABLE_INFO) {
          Info("FollowProlongation", "Adjusting sector failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        }
        continue;
      }

      // check if track is findable
      if (IsGeoFindable(&fCandidates[2*iCandidate+currIdx], iLayer, prop->getAlpha() )) {
        fCandidates[2*iCandidate+currIdx].SetIsFindable(iLayer);
      }

      // define search window
      roadY = 7.f * sqrt(fCandidates[2*iCandidate+currIdx].getSigmaY2() + 0.1f * 0.1f) + 2.f; // add constant to the road for better efficiency
      //roadZ = 7.f * sqrt(fCandidates[2*iCandidate+currIdx].getSigmaZ2() + 9.f * 9.f / 12.f); // take longest pad length
      roadZ = 18.f; // simply twice the longest pad length -> efficiency 99.996%
      //
      if (CAMath::Abs(fCandidates[2*iCandidate+currIdx].getZ()) - roadZ >= zMaxTRD ) {
        if (ENABLE_INFO) {
          Info("FollowProlongation", "Track out of TRD acceptance with z=%f in layer %i (eta=%f)",
            fCandidates[2*iCandidate+currIdx].getZ(), iLayer, fCandidates[2*iCandidate+currIdx].getEta());
        }
        continue;
      }

      // determine chamber(s) to be searched for tracklets
      FindChambersInRoad(&fCandidates[2*iCandidate+currIdx], roadY, roadZ, iLayer, det, zMaxTRD, prop->getAlpha());

      // track debug information to be stored in case no matching tracklet can be found
      fDebug->SetTrackParameter(fCandidates[2*iCandidate+currIdx], iLayer);

      // look for tracklets in chamber(s)
      for (int iDet = 0; iDet < nMaxChambersToSearch; iDet++) {
        int currDet = det[iDet];
        if (currDet == -1) {
          continue;
        }
        int currSec = fGeo->GetSector(currDet);
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
          Error("FollowProlongation", "Track is in sector %i and sector %i is searched for tracklets",
                    GetSector(prop->getAlpha()), currSec);
          continue;
        }
        // first propagate track to x of tracklet
        for (int iTrklt=0; iTrklt<fNtrackletsInChamber[currDet]; ++iTrklt) {
          int trkltIdx = fTrackletIndexArray[currDet] + iTrklt;
          if (!prop->PropagateToX(fSpacePoints[trkltIdx].fR, .8f, 2.f)) {
            if (ENABLE_WARNING) {
              Warning("FollowProlongation", "Track parameter for track %i, x=%f at tracklet %i x=%f in layer %i cannot be retrieved",
                iTrack, fCandidates[2*iCandidate+currIdx].getX(), iTrklt, fSpacePoints[trkltIdx].fR, iLayer);
            }
            continue;
          }
          // correction for tilted pads (only applied if deltaZ < l_pad && track z err << l_pad)
          float tiltCorr = tilt * (fSpacePoints[trkltIdx].fX[1] - fCandidates[2*iCandidate+currIdx].getZ());
          float l_pad = pad->GetRowSize(fTracklets[trkltIdx].GetZbin());
          if (!( (CAMath::Abs(fSpacePoints[trkltIdx].fX[1] - fCandidates[2*iCandidate+currIdx].getZ()) <  l_pad) &&
               (fCandidates[2*iCandidate+currIdx].getSigmaZ2() < (l_pad*l_pad/12.f)) ))
          {
            tiltCorr = 0.f;
          }
          // correction for mean z position of tracklet (is not the center of the pad if track eta != 0)
          float zPosCorr = fSpacePoints[trkltIdx].fX[1] + fZCorrCoefNRC * fCandidates[2*iCandidate+currIdx].getTgl();
          float yPosCorr = fSpacePoints[trkltIdx].fX[0] - tiltCorr;
          float deltaY = yPosCorr - fCandidates[2*iCandidate+currIdx].getY();
          float deltaZ = zPosCorr - fCandidates[2*iCandidate+currIdx].getZ();
          My_Float trkltPosTmpYZ[2] = { yPosCorr, zPosCorr };
          My_Float trkltCovTmp[3] = { 0.f };
          if ( (CAMath::Abs(deltaY) < roadY) && (CAMath::Abs(deltaZ) < roadZ) )
          {
            //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            RecalcTrkltCov(tilt, fCandidates[2*iCandidate+currIdx].getSnp(), pad->GetRowSize(fTracklets[trkltIdx].GetZbin()), trkltCovTmp);
            float chi2 = prop->getPredictedChi2(trkltPosTmpYZ, trkltCovTmp);
            //printf("layer %i: chi2 = %f\n", iLayer, chi2);
            if (chi2 < fMaxChi2) {
              if (nCurrHypothesis < fNhypothesis) {
                fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + chi2;
                fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
                fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fCandidateId = iCandidate;
                fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fTrackletId = trkltIdx;
                nCurrHypothesis++;
              }
              else {
                // maximum number of hypothesis reached -> sort them and replace worst hypo with new one if it is better
                Quicksort(hypothesisIdxOffset, nCurrHypothesis - 1 + hypothesisIdxOffset, nCurrHypothesis, 1);
                if ( ((chi2 + fCandidates[2*iCandidate+currIdx].GetChi2()) / CAMath::Max(fCandidates[2*iCandidate+currIdx].GetNlayers(), 1)) <
                      (fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fChi2 / CAMath::Max(fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fLayers, 1)) ) {
                  fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + chi2;
                  fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
                  fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fCandidateId = iCandidate;
                  fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fTrackletId = trkltIdx;
                }
              }
            } // end tracklet chi2 < fMaxChi2
          } // end tracklet in window
        } // tracklet loop
      } // chamber loop

      // add no update to hypothesis list
      if (nCurrHypothesis < fNhypothesis) {
        fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty;
        fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
        fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fCandidateId = iCandidate;
        fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fTrackletId = -1;
        nCurrHypothesis++;
      }
      else {
        Quicksort(hypothesisIdxOffset, nCurrHypothesis - 1 + hypothesisIdxOffset, nCurrHypothesis, 1);
        if ( ((fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty) / CAMath::Max(fCandidates[2*iCandidate+currIdx].GetNlayers(), 1)) <
             (fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fChi2 / CAMath::Max(fHypothesis[nCurrHypothesis + hypothesisIdxOffset].fLayers, 1)) ) {
          fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty;
          fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
          fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis-1 + hypothesisIdxOffset].fTrackletId = -1;
        }
      }
      isOK = true;
    } // end candidate loop

#ifdef ENABLE_GPUMC
    // in case matching tracklet exists in this layer -> store position information for debugging
    if (matchAvailableAll[iLayer].size() > 0 && fDebugOutput) {
      fDebug->SetNmatchAvail(matchAvailableAll[iLayer].size(), iLayer);
      int realTrkltId = matchAvailableAll[iLayer].at(0);
      prop->setTrack(&fCandidates[currIdx]);
      bool flag = prop->PropagateToX(fSpacePoints[realTrkltId].fR, .8f, 2.f);
      if (flag) {
        flag = AdjustSector(prop, &fCandidates[currIdx], iLayer);
      }
      if (!flag) {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Track parameter at x=%f for track %i at real tracklet x=%f in layer %i cannot be retrieved (pt=%f)",
            fCandidates[currIdx].getX(), iTrack, fSpacePoints[realTrkltId].fR, iLayer, fCandidates[currIdx].getPt());
        }
      }
      else {
        fDebug->SetTrackParameterReal(fCandidates[currIdx], iLayer);
        float zPosCorrReal = fSpacePoints[realTrkltId].fX[1] + fZCorrCoefNRC * fCandidates[currIdx].getTgl();
        float deltaZReal = zPosCorrReal - fCandidates[currIdx].getZ();
        float tiltCorrReal = tilt * (fSpacePoints[realTrkltId].fX[1] - fCandidates[currIdx].getZ());
        float l_padReal = pad->GetRowSize(fTracklets[realTrkltId].GetZbin());
        if ( (fCandidates[currIdx].getSigmaZ2() >= (l_padReal*l_padReal/12.f)) ||
             (CAMath::Abs(fSpacePoints[realTrkltId].fX[1] - fCandidates[currIdx].getZ()) >= l_padReal) )
        {
          tiltCorrReal = 0;
        }
        My_Float yzPosReal[2] = { fSpacePoints[realTrkltId].fX[0] - tiltCorrReal, zPosCorrReal };
        My_Float covReal[3] = { 0. };
        RecalcTrkltCov(tilt, fCandidates[currIdx].getSnp(), pad->GetRowSize(fTracklets[realTrkltId].GetZbin()), covReal);
        fDebug->SetChi2Real(prop->getPredictedChi2(yzPosReal, covReal), iLayer);
        fDebug->SetRawTrackletPositionReal(fSpacePoints[realTrkltId].fR, fSpacePoints[realTrkltId].fX, iLayer);
        fDebug->SetCorrectedTrackletPositionReal(yzPosReal, iLayer);
        fDebug->SetTrackletPropertiesReal(fTracklets[realTrkltId].GetDetector(), iLayer);
      }
    }
#endif
    //
    Quicksort(hypothesisIdxOffset, nCurrHypothesis - 1 + hypothesisIdxOffset, nCurrHypothesis, 1);
    fDebug->SetChi2Update(fHypothesis[0 + hypothesisIdxOffset].fChi2 - t->GetChi2(), iLayer); // only meaningful for ONE candidate!!!
    fDebug->SetRoad(roadY, roadZ, iLayer); // only meaningful for ONE candidate
    bool wasTrackStored = false;
    // --------------------------------------------------------------------------------
    //
    // loop over the best N_candidates hypothesis
    //
    // --------------------------------------------------------------------------------
    //printf("nCurrHypothesis=%i, nCandidates=%i\n", nCurrHypothesis, nCandidates);
    //for (int idx=0; idx<10; ++idx) { printf("fHypothesis[%i]: candidateId=%i, nLayers=%i, trackletId=%i, chi2=%f\n", idx, fHypothesis[idx].fCandidateId,  fHypothesis[idx].fLayers, fHypothesis[idx].fTrackletId, fHypothesis[idx].fChi2); }
    for (int iUpdate = 0; iUpdate < nCurrHypothesis && iUpdate < fNCandidates; iUpdate++) {
      if (fHypothesis[iUpdate + hypothesisIdxOffset].fCandidateId == -1) {
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
      fCandidates[2*iUpdate+nextIdx] = fCandidates[2*fHypothesis[iUpdate + hypothesisIdxOffset].fCandidateId+currIdx];
      if (fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId == -1) {
        // no matching tracklet found
        if (fCandidates[2*iUpdate+nextIdx].GetIsFindable(iLayer)) {
          if (fCandidates[2*iUpdate+nextIdx].GetNmissingConsecLayers(iLayer) > fMaxMissingLy) {
            fCandidates[2*iUpdate+nextIdx].SetIsStopped();
          }
          fCandidates[2*iUpdate+nextIdx].SetChi2(fCandidates[2*iUpdate+nextIdx].GetChi2() + fChi2Penalty);
        }
        if (iUpdate == 0) {
          *t = fCandidates[2*iUpdate+nextIdx];
        }
        continue;
      }
      // matching tracklet found
      prop->setTrack(&fCandidates[2*iUpdate+nextIdx]);
      int trkltSec = fGeo->GetSector(fTracklets[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].GetDetector());
      if ( trkltSec != GetSector(prop->getAlpha())) {
        // if after a matching tracklet was found another sector was searched for tracklets the track needs to be rotated back
        prop->rotate( GetAlphaOfSector(trkltSec) );
      }
      if (!prop->PropagateToX(fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fR, .8f, 2.f)){
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Final track propagation for track %i update %i in layer %i failed", iTrack, iUpdate, iLayer);
        }
        fCandidates[2*iUpdate+nextIdx].SetChi2(fCandidates[2*iUpdate+nextIdx].GetChi2() + fChi2Penalty);
        if (fCandidates[2*iUpdate+nextIdx].GetIsFindable(iLayer)) {
          if (fCandidates[2*iUpdate+nextIdx].GetNmissingConsecLayers(iLayer) >= fMaxMissingLy) {
            fCandidates[2*iUpdate+nextIdx].SetIsStopped();
          }
        }
        if (iUpdate == 0) {
          *t = fCandidates[2*iUpdate+nextIdx];
        }
        continue;
      }

      float tiltCorrUp = tilt * (fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fX[1] - fCandidates[2*iUpdate+nextIdx].getZ());
      float zPosCorrUp = fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fX[1] + fZCorrCoefNRC * fCandidates[2*iUpdate+nextIdx].getTgl();
      float l_padTrklt = pad->GetRowSize(fTracklets[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].GetZbin());
      if (!((fCandidates[2*iUpdate+nextIdx].getSigmaZ2() < (l_padTrklt*l_padTrklt/12.f)) &&
           (CAMath::Abs(fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fX[1] - fCandidates[2*iUpdate+nextIdx].getZ()) < l_padTrklt)))
      {
        tiltCorrUp = 0.f;
      }
      My_Float trkltPosUp[2] = { fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fX[0] - tiltCorrUp, zPosCorrUp };
      My_Float trkltCovUp[3] = { 0.f };
      RecalcTrkltCov(tilt, fCandidates[2*iUpdate+nextIdx].getSnp(), pad->GetRowSize(fTracklets[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].GetZbin()), trkltCovUp);

#ifdef ENABLE_GPUTRDDEBUG
      prop->setTrack(&trackNoUp);
      prop->rotate(GetAlphaOfSector(trkltSec));
      prop->PropagateToX(fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fR, .8f, 2.f);
      prop->PropagateToX(fR[iLayer], .8f, 2.f);
      prop->setTrack(&fCandidates[2*iUpdate+nextIdx]);
#endif

      if (!wasTrackStored) {
#ifdef ENABLE_GPUTRDDEBUG
        fDebug->SetTrackParameterNoUp(trackNoUp, iLayer);
#endif
        fDebug->SetTrackParameter(fCandidates[2*iUpdate+nextIdx], iLayer);
        fDebug->SetRawTrackletPosition(fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fR, fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fX, iLayer);
        fDebug->SetCorrectedTrackletPosition(trkltPosUp, iLayer);
        fDebug->SetTrackletCovariance(fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fCov, iLayer);
        fDebug->SetTrackletProperties(fSpacePoints[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].fDy, fTracklets[fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId].GetDetector(), iLayer);
        fDebug->SetRoad(roadY, roadZ, iLayer);
        wasTrackStored = true;
      }

      if (!prop->update(trkltPosUp, trkltCovUp))
      {
        if (ENABLE_WARNING) {
          Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);
        }
        fCandidates[2*iUpdate+nextIdx].SetChi2(fCandidates[2*iUpdate+nextIdx].GetChi2() + fChi2Penalty);
        if (fCandidates[2*iUpdate+nextIdx].GetIsFindable(iLayer)) {
          if (fCandidates[2*iUpdate+nextIdx].GetNmissingConsecLayers(iLayer) >= fMaxMissingLy) {
            fCandidates[2*iUpdate+nextIdx].SetIsStopped();
          }
        }
        if (iUpdate == 0) {
          *t = fCandidates[2*iUpdate+nextIdx];
        }
        continue;
      }
      if (!fCandidates[2*iUpdate+nextIdx].CheckNumericalQuality()) {
        if (ENABLE_WARNING) {
          Info("FollowProlongation", "Track %i has invalid covariance matrix. Aborting track following\n", iTrack);
        }
        return false;
      }
      fCandidates[2*iUpdate+nextIdx].AddTracklet(iLayer, fHypothesis[iUpdate + hypothesisIdxOffset].fTrackletId);
      fCandidates[2*iUpdate+nextIdx].SetChi2(fHypothesis[iUpdate + hypothesisIdxOffset].fChi2);
      fCandidates[2*iUpdate+nextIdx].SetIsFindable(iLayer);
      if (iUpdate == 0) {
        *t = fCandidates[2*iUpdate+nextIdx];
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
  if (fDebugOutput) {
    int update[6] = { 0 };
    if (!fMCEvent) {
      for (int iLy = 0; iLy < kNLayers; iLy++) {
        if (t->GetTracklet(iLy) != -1) {
          update[iLy] = 1;
        }
      }
    }
    else {
      // for MC: check attached tracklets (match, related, fake)
      int nRelated = 0;
      int nMatching = 0;
      int nFake = 0;
      for (int iLy = 0; iLy < kNLayers; iLy++) {
        if (t->GetTracklet(iLy) != -1) {
          int lbTracklet;
          for (int il=0; il<3; il++) {
            if ( (lbTracklet = fSpacePoints[t->GetTracklet(iLy)].fLabel[il]) < 0 ) {
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
          if (update[iLy] < 1 && fMCEvent) {
            // no exact match, check in related labels
            bool isRelated = false;
            for (int il=0; il<3; il++) {
              if (isRelated) {
                break;
              }
              if ( (lbTracklet = fSpacePoints[t->GetTracklet(iLy)].fLabel[il]) < 0 ) {
                // no more valid labels
                continue;
              }
              AliMCParticle *mcPart = (AliMCParticle*) fMCEvent->GetTrack(lbTracklet);
              while (mcPart) {
                int motherPart = mcPart->GetMother();
                if (motherPart == CAMath::Abs(trackID)) {
                  update[iLy] = 4 + il;
                  nRelated++;
                  isRelated = true;
                  break;
                }
                mcPart = motherPart >= 0 ? (AliMCParticle*) fMCEvent->GetTrack(motherPart) : 0;
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
      fDebug->SetTrackProperties(nMatching, nFake, nRelated);
#ifdef ENABLE_GPUMC
      AliMCParticle *mcPartDbg = (AliMCParticle*) fMCEvent->GetTrack(trackID);
      if (mcPartDbg) {
        fDebug->SetMCinfo(mcPartDbg->Xv(), mcPartDbg->Yv(), mcPartDbg->Zv(), mcPartDbg->PdgCode());
      }
#endif
    }
    fDebug->SetTrack(*t);
    fDebug->SetUpdates(update);
    fDebug->Output();
  }

  return true;
}

GPUd() int AliGPUTRDTracker::GetDetectorNumber(const float zPos, const float alpha, const int layer) const
{
  //--------------------------------------------------------------------
  // if track position is within chamber return the chamber number
  // otherwise return -1
  //--------------------------------------------------------------------
  int stack = fGeo->GetStack(zPos, layer);
  if (stack < 0) {
    return -1;
  }
  int sector = GetSector(alpha);

  return fGeo->GetDetector(layer, stack, sector);
}

GPUd() bool AliGPUTRDTracker::AdjustSector(GPUTRDPropagator *prop, GPUTRDTrack *t, const int layer) const
{
  //--------------------------------------------------------------------
  // rotate track in new sector if necessary and
  // propagate to correct x of layer
  // cancel if track crosses two sector boundaries
  //--------------------------------------------------------------------
  float alpha     = fGeo->GetAlpha();
  float xTmp      = t->getX();
  float y         = t->getY();
  float yMax      = t->getX() * CAMath::Tan(0.5f * alpha);
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

GPUd() int AliGPUTRDTracker::GetSector(float alpha) const
{
  //--------------------------------------------------------------------
  // TRD sector number for reference system alpha
  //--------------------------------------------------------------------
  if (alpha < 0) {
    alpha += 2.f * M_PI;
  }
  else if (alpha >= 2.f * M_PI) {
    alpha -= 2.f * M_PI;
  }
  return (int) (alpha * kNSectors / (2.f * M_PI));
}

GPUd() float AliGPUTRDTracker::GetAlphaOfSector(const int sec) const
{
  //--------------------------------------------------------------------
  // rotation angle for TRD sector sec
  //--------------------------------------------------------------------
  return (2.0f * M_PI / (float) kNSectors * ((float) sec + 0.5f));
}

GPUd() void AliGPUTRDTracker::RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3])
{
  //--------------------------------------------------------------------
  // recalculate tracklet covariance taking track phi angle into account
  // store the new values in cov
  //--------------------------------------------------------------------
  float t2 = tilt * tilt; // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = GetRPhiRes(snp);
  float sz2 = rowSize * rowSize / 12.f;
  cov[0] = c2 * (sy2 + t2 * sz2);
  cov[1] = c2 * tilt * (sz2 - sy2);
  cov[2] = c2 * (t2 * sy2 + sz2);
}

GPUd() void AliGPUTRDTracker::FindChambersInRoad(const GPUTRDTrack *t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const
{
  //--------------------------------------------------------------------
  // determine initial chamber where the track ends up
  // add more chambers of the same sector or (and) neighbouring
  // stack if track is close the edge(s) of the chamber
  //--------------------------------------------------------------------

  const float yMax    = CAMath::Abs(fGeo->GetCol0(iLayer));

  int currStack = fGeo->GetStack(t->getZ(), iLayer);
  int currSec = GetSector(alpha);
  int currDet;

  int nDets = 0;

  if (currStack > -1) {
    // chamber unambiguous
    currDet = fGeo->GetDetector(iLayer, currStack, currSec);
    det[nDets++] = currDet;
    const AliGPUTRDpadPlane *pp = fGeo->GetPadPlane(iLayer, currStack);
    int lastPadRow = fGeo->GetRowMax(iLayer, currStack, 0);
    float zCenter = pp->GetRowPos(lastPadRow / 2);
    if ( ( t->getZ() + roadZ ) > pp->GetRowPos(0) || ( t->getZ() - roadZ ) < pp->GetRowPos(lastPadRow) ) {
      int addStack = t->getZ() > zCenter ? currStack - 1 : currStack + 1;
      if (addStack < kNStacks && addStack > -1) {
        det[nDets++] = fGeo->GetDetector(iLayer, addStack, currSec);
      }
    }
  }
  else {
    if (CAMath::Abs(t->getZ()) > zMax) {
      // shift track in z so it is in the TRD acceptance
      if (t->getZ() > 0) {
          currDet = fGeo->GetDetector(iLayer, 0, currSec);
      }
      else {
        currDet = fGeo->GetDetector(iLayer, kNStacks-1, currSec);
      }
      det[nDets++] = currDet;
      currStack = fGeo->GetStack(currDet);
    }
    else {
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
  if ( ( CAMath::Abs(t->getY()) + roadY ) > yMax ) {
    const int nStacksToSearch = nDets;
    int newSec;
    if (t->getY() > 0) {
      newSec = (currSec + 1) % kNSectors;
    }
    else {
      newSec = (currSec > 0) ? currSec - 1 : kNSectors - 1;
    }
    for (int idx = 0; idx < nStacksToSearch; ++idx) {
      currStack = fGeo->GetStack(det[idx]);
      det[nDets++] = fGeo->GetDetector(iLayer, currStack, newSec);
    }
  }
  // skip PHOS hole and non-existing chamber 17_4_4
  for (int iDet = 0; iDet < nDets; iDet++) {
    if (!fGeo->ChamberInGeometry(det[iDet])) {
      det[iDet] = -1;
    }
  }
}

GPUd() bool AliGPUTRDTracker::IsGeoFindable(const GPUTRDTrack *t, const int layer, const float alpha) const
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
  if (!fGeo->ChamberInGeometry(det)) {
    return false;
  }

  const AliGPUTRDpadPlane *pp = fGeo->GetPadPlane(det);
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

GPUd() void AliGPUTRDTracker::SwapTracklets(const int left, const int right)
{
  //--------------------------------------------------------------------
  // swapping function for tracklets
  //--------------------------------------------------------------------
  AliGPUTRDTrackletWord tmp = fTracklets[left];
  fTracklets[left] = fTracklets[right];
  fTracklets[right] = tmp;
}

GPUd() int AliGPUTRDTracker::PartitionTracklets(const int left, const int right)
{
  //--------------------------------------------------------------------
  // partitioning for tracklets
  //--------------------------------------------------------------------
  const int mid = left + (right - left) / 2;
  AliGPUTRDTrackletWord pivot = fTracklets[mid];
  SwapTracklets(mid, left);
  int i = left + 1;
  int j = right;
  while (i <= j) {
    while (i <= j && fTracklets[i] <= pivot) {
      i++;
    }
    while (i <= j && fTracklets[j] > pivot) {
      j--;
    }
    if (i < j) {
      SwapTracklets(i, j);
    }
  }
  SwapTracklets(i-1, left);
  return i - 1;
}

GPUd() void AliGPUTRDTracker::SwapHypothesis(const int left, const int right)
{
  //--------------------------------------------------------------------
  // swapping function for hypothesis
  //--------------------------------------------------------------------
  Hypothesis tmp = fHypothesis[left];
  fHypothesis[left] = fHypothesis[right];
  fHypothesis[right] = tmp;
}

GPUd() int AliGPUTRDTracker::PartitionHypothesis(const int left, const int right)
{
  //--------------------------------------------------------------------
  // partitioning for hypothesis
  //--------------------------------------------------------------------
  const int mid = left + (right - left) / 2;
  Hypothesis pivot = fHypothesis[mid];
  SwapHypothesis(mid, left);
  int i = left + 1;
  int j = right;
  while (i <= j) {
    int nLayersPivot = (pivot.fLayers > 0) ? pivot.fLayers : 1;
    int nLayersElem = (fHypothesis[i].fLayers > 0) ? fHypothesis[i].fLayers : 1;
    while (i <= j && (fHypothesis[i].fChi2 / nLayersElem) <= (pivot.fChi2 / nLayersPivot)) {
      i++;
      nLayersElem =  (fHypothesis[i].fLayers > 0) ? fHypothesis[i].fLayers : 1;
    }
    nLayersElem =  (fHypothesis[j].fLayers > 0) ? fHypothesis[j].fLayers : 1;
    while (i <= j && (fHypothesis[j].fChi2 / nLayersElem) > (pivot.fChi2 / nLayersPivot)) {
      j--;
      nLayersElem =  (fHypothesis[j].fLayers > 0) ? fHypothesis[j].fLayers : 1;
    }
    if (i < j) {
      SwapHypothesis(i, j);
    }
  }
  SwapHypothesis(i-1, left);
  return i - 1;
}

GPUd() void AliGPUTRDTracker::Quicksort(const int left, const int right, const int size, const int type)
{
  //--------------------------------------------------------------------
  // use own quicksort implementation since std::sort not available
  // sorting either tracklet array (type == 0)
  // or hypothesis array (type == 1)
  //--------------------------------------------------------------------
  if (left >= right) {
    return;
  }
  int part;
  if (type == 0) {
    part = PartitionTracklets(left, right);
  }
  else {
    part = PartitionHypothesis(left, right);
  }
  Quicksort(left, part - 1, size, type);
  Quicksort(part + 1, right, size, type);
}
