//#define ENABLE_HLTTRDDEBUG
#define ENABLE_WARNING 0
#define ENABLE_INFO 0
#ifdef HLTCA_BUILD_ALIROOT_LIB
#define ENABLE_HLTMC
#endif

#include <vector>
#include <algorithm>
#include "AliHLTTRDTracker.h"
#include "AliHLTTRDTrackletWord.h"
#include "AliHLTTRDGeometry.h"
#include "AliHLTTRDTrack.h"
#include "AliHLTTRDTrackerDebug.h"
#include "AliHLTTPCGMMerger.h"

#ifdef HLTCA_BUILD_ALIROOT_LIB
#include "TDatabasePDG.h"
#include "AliMCParticle.h"
#include "AliMCEvent.h"
static const float piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass();
#else
static const float piMass = 0.139f;
#endif

// parameters for track propagation
static const float fgkMaxSnp = 0.8;
static const float fgkMaxStep = 2.0;

// parameters from TRD calibration
static const int fgkNmaskedChambers = 68;

static const AliHLTTPCGMMerger fgkMerger;

#ifndef HLTCA_GPUCODE
AliHLTTRDTracker::AliHLTTRDTracker() :
  fR(nullptr),
  fIsInitialized(false),
  fTracks(nullptr),
  fNCandidates(1),
  fNTracks(0),
  fNEvents(0),
  fTracklets(nullptr),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fNtrackletsInChamber(nullptr),
  fTrackletIndexArray(nullptr),
  fHypothesis(nullptr),
  fCandidates(nullptr),
  fSpacePoints(nullptr),
  fGeo(nullptr),
  fDebugOutput(false),
  fMinPt(0.6),
  fMaxEta(0.84),
  fMaxChi2(15.0),
  fMaxMissingLy(6),
  fChi2Penalty(12.0),
  fZCorrCoefNRC(1.4),
  fNhypothesis(100),
  fMaskedChambers(nullptr),
  fMCEvent(nullptr),
  fMerger(&fgkMerger),
  fDebug(nullptr)
{
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
}

AliHLTTRDTracker::~AliHLTTRDTracker()
{
  //--------------------------------------------------------------------
  // Destructor
  //--------------------------------------------------------------------
  if (fIsInitialized) {
    delete[] fTracklets;
    delete[] fTracks;
    delete[] fSpacePoints;
    delete[] fHypothesis;
    delete[] fCandidates;
    delete[] fNtrackletsInChamber;
    delete[] fTrackletIndexArray;
    delete[] fR;
    delete fGeo;
    delete fDebug;
  }
}

#endif

GPUd() void AliHLTTRDTracker::Init()
{
  //--------------------------------------------------------------------
  // Initialise tracker
  //--------------------------------------------------------------------
  if(!AliHLTTRDGeometry::CheckGeometryAvailable()){
    Error("Init", "Could not get geometry.");
  }

  fNtrackletsInChamber = new int[kNChambers];
  fTrackletIndexArray = new int[kNChambers];
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fNtrackletsInChamber[iDet] = 0;
    fTrackletIndexArray[iDet] = -1;
  }

  //FIXME hard-coded masked chambers -> this should eventually be taken from the OCDB (but this is not available before calibration...)
  // at least permanently dead chambers can be hard-coded for the future, but not for older runs
  // to be re-evaluated after the repair in LS2 is done

  // masked chambers are excluded from tracklet search
  // run 244340 (pp)
  //int maskedChambers[] = { 5, 17, 26, 27, 40, 43, 50, 55, 113, 132, 181, 219, 221, 226, 227, 228, 230, 231, 233, 236, 238,
  //                          241, 249, 265, 277, 287, 302, 308, 311, 318, 319, 320, 335, 368, 377, 389, 402, 403, 404, 405,
  //                          406, 407, 432, 433, 434, 435, 436, 437, 452, 461, 462, 463, 464, 465, 466, 467, 570, 490, 491,
  //                          493, 494, 500, 504, 538 };
  // run 245353 (PbPb)
  //int maskedChambers[] = {    5, 17, 26, 27, 32, 40, 41, 43, 50, 55, 113, 132, 181, 219, 221, 226, 227, 228, 230, 231, 232, 233,
  //                          236, 238, 241, 249, 265, 277, 287, 302, 308, 311, 318, 317, 320, 335, 368, 377, 389, 402, 403, 404,
  //                          405, 406, 407, 432, 433, 434, 435, 436, 437, 452, 461, 462, 463, 464, 465, 466, 467, 470, 490, 491,
  //                          493, 494, 500, 502, 504, 538 };

  //int nChambersMasked = sizeof(maskedChambers) / sizeof(int);
  //fMaskedChambers.insert(fMaskedChambers.begin(), maskedChambers, maskedChambers+nChambersMasked);

  // run 245353 (PbPb)
  fMaskedChambers = new unsigned short [fgkNmaskedChambers] {
    5, 17, 26, 27, 32, 40, 41, 43, 50, 55, 113, 132, 181, 219, 221, 226, 227, 228, 230, 231, 232, 233,
    236, 238, 241, 249, 265, 277, 287, 302, 308, 311, 318, 317, 320, 335, 368, 377, 389, 402, 403, 404,
    405, 406, 407, 432, 433, 434, 435, 436, 437, 452, 461, 462, 463, 464, 465, 466, 467, 470, 490, 491,
    493, 494, 500, 502, 504, 538 };

  fTracklets = new AliHLTTRDTrackletWord[fNtrackletsMax];
  fHypothesis = new Hypothesis[fNhypothesis];
  for (int iHypo=0; iHypo < fNhypothesis; iHypo++) {
    fHypothesis[iHypo].fChi2 = 1e4;
    fHypothesis[iHypo].fLayers = 0;
    fHypothesis[iHypo].fCandidateId = -1;
    fHypothesis[iHypo].fTrackletId = -1;
  }
  fCandidates = new HLTTRDTrack[2*fNCandidates];
  fSpacePoints = new AliHLTTRDSpacePointInternal[fNtrackletsMax];

  // obtain average radius of TRD layers (use default value w/o misalignment if no transformation matrix can be obtained)
  float x0[kNLayers]    = { 300.2, 312.8, 325.4, 338.0, 350.6, 363.2 };
  fR = new float[kNLayers];
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = x0[iLy];
  }
  fGeo = new AliHLTTRDGeometry();
  if (!fGeo) {
    Error("Init", "TRD geometry could not be loaded");
  }
  fGeo->CreateClusterMatrixArray();
  TGeoHMatrix *matrix = nullptr;
  double loc[3] = { fGeo->AnodePos(), 0., 0. };
  double glb[3] = { 0., 0., 0. };
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

  fDebug = new AliHLTTRDTrackerDebug();
  fDebug->ExpandVectors();

  fIsInitialized = true;
}

GPUd() void AliHLTTRDTracker::Reset()
{
  //--------------------------------------------------------------------
  // Reset tracker
  //--------------------------------------------------------------------
  fNTracklets = 0;
  for (int i=0; i<fNtrackletsMax; ++i) {
    fTracklets[i] = 0x0;
    fSpacePoints[i].fR        = 0.;
    fSpacePoints[i].fX[0]     = 0.;
    fSpacePoints[i].fX[1]     = 0.;
    fSpacePoints[i].fCov[0]   = 0.;
    fSpacePoints[i].fCov[1]   = 0.;
    fSpacePoints[i].fCov[2]   = 0.;
    fSpacePoints[i].fDy       = 0.;
    fSpacePoints[i].fId       = 0;
    fSpacePoints[i].fVolumeId = 0;
  }
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fNtrackletsInChamber[iDet] = 0;
    fTrackletIndexArray[iDet] = -1;
  }
}

GPUd() void AliHLTTRDTracker::StartLoadTracklets(const int nTrklts)
{
  //--------------------------------------------------------------------
  // Prepare tracker for the tracklets
  // - adjust array size if nTrklts > nTrkltsMax
  //--------------------------------------------------------------------
  if (nTrklts > fNtrackletsMax) {
    delete[] fTracklets;
    delete[] fSpacePoints;
    fNtrackletsMax += nTrklts - fNtrackletsMax;
    fTracklets = new AliHLTTRDTrackletWord[fNtrackletsMax];
    fSpacePoints = new AliHLTTRDSpacePointInternal[fNtrackletsMax];
  }
}

GPUd() void AliHLTTRDTracker::LoadTracklet(const AliHLTTRDTrackletWord &tracklet)
{
  //--------------------------------------------------------------------
  // Add single tracklet to tracker
  //--------------------------------------------------------------------
  if (fNTracklets >= fNtrackletsMax ) {
    Error("LoadTracklet", "Running out of memory for tracklets, skipping tracklet(s). This should actually never happen.");
    return;
  }
  fTracklets[fNTracklets++] = tracklet;
  fNtrackletsInChamber[tracklet.GetDetector()]++;
}

GPUd() void AliHLTTRDTracker::DoTracking( HLTTRDTrack *tracksTPC, int *tracksTPClab, int nTPCtracks, int *tracksTPCnTrklts, int *tracksTRDlabel )
{
  //--------------------------------------------------------------------
  // Steering function for the tracking
  //--------------------------------------------------------------------

  // sort tracklets and fill index array
  //std::sort(fTracklets, fTracklets + fNTracklets);
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

  delete[] fTracks;
  fTracks = new HLTTRDTrack[nTPCtracks];
  fNTracks = 0;

  for (int i=0; i<nTPCtracks; ++i) {
    // TODO is this copying necessary or can it be omitted for optimization?
    HLTTRDTrack tMI(tracksTPC[i]);
    HLTTRDTrack *t = &tMI;
    t->SetTPCtrackId(i);
    t->SetLabel(tracksTPClab[i]);
    if (tracksTPCnTrklts) {
      t->SetNtrackletsOffline(tracksTPCnTrklts[i]);
    }
    if (tracksTRDlabel) {
      t->SetLabelOffline(tracksTRDlabel[i]);
    }
    HLTTRDPropagator prop(fMerger);
    prop.setTrack(t);
    FollowProlongation(&prop, t, nTPCtracks);
    fTracks[fNTracks++] = *t;
  }

  fNEvents++;
}


GPUd() bool AliHLTTRDTracker::CalculateSpacePoints()
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

    TGeoHMatrix *matrix = fGeo->GetClusterMatrix(iDet);
    if (!matrix){
	    Error("CalculateSpacePoints", "Invalid TRD cluster matrix, skipping detector  %i", iDet);
      result = false;
	    continue;
    }
    AliHLTTRDpadPlane *pp = fGeo->GetPadPlane(iDet);
    float tilt = tanf( M_PI / 180. * pp->GetTiltingAngle());
    float t2 = tilt * tilt; // tan^2 (tilt)
    float c2 = 1. / (1. + t2); // cos^2 (tilt)
    float sy2 = pow(0.10, 2); // sigma_rphi^2, currently assume sigma_rphi = 1 mm

    for (int iTrklt=0; iTrklt<nTracklets; ++iTrklt) {
      int trkltIdx = fTrackletIndexArray[iDet] + iTrklt;
      int trkltZbin = fTracklets[trkltIdx].GetZbin();
      float sz2 = pow(pp->GetRowSize(trkltZbin), 2) / 12.; // sigma_z = l_pad/sqrt(12) TODO try a larger z error
      double xTrkltDet[3] = { 0. }; // trklt position in chamber coordinates
      double xTrkltSec[3] = { 0. }; // trklt position in sector coordinates
      xTrkltDet[0] = fGeo->AnodePos();
      xTrkltDet[1] = fTracklets[trkltIdx].GetY();
      xTrkltDet[2] = pp->GetRowPos(trkltZbin) - pp->GetRowSize(trkltZbin)/2. - pp->GetRowPos(pp->GetNrows()/2);
      matrix->LocalToMaster(xTrkltDet, xTrkltSec);
      fSpacePoints[trkltIdx].fR = xTrkltSec[0];
      fSpacePoints[trkltIdx].fX[0] = xTrkltSec[1];
      fSpacePoints[trkltIdx].fX[1] = xTrkltSec[2];
      fSpacePoints[trkltIdx].fId = fTracklets[trkltIdx].GetId();
      for (int i=0; i<3; i++) {
        fSpacePoints[trkltIdx].fLabel[i] = fTracklets[trkltIdx].GetLabel(i);
      }
      fSpacePoints[trkltIdx].fCov[0] = c2 * (sy2 + t2 * sz2);
      fSpacePoints[trkltIdx].fCov[1] = c2 * tilt * (sz2 - sy2);
      fSpacePoints[trkltIdx].fCov[2] = c2 * (t2 * sy2 + sz2);
      fSpacePoints[trkltIdx].fDy = 0.014 * fTracklets[trkltIdx].GetdY();

      int modId   = fGeo->GetSector(iDet) * AliHLTTRDGeometry::kNstack + fGeo->GetStack(iDet); // global TRD stack number
      unsigned short volId = fGeo->GetGeomManagerVolUID(iDet, modId);
      fSpacePoints[trkltIdx].fVolumeId = volId;
    }
  }
  return result;
}


GPUd() bool AliHLTTRDTracker::FollowProlongation(HLTTRDPropagator *prop, HLTTRDTrack *t, int nTPCtracks)
{
  //--------------------------------------------------------------------
  // Propagate TPC track layerwise through TRD and pick up closest
  // tracklet(s) on the way
  // -> returns false if prolongation could not be executed fully
  //    or track does not fullfill threshold conditions
  //--------------------------------------------------------------------

  // only propagate tracks within TRD acceptance
  if (fabsf(t->getEta()) > fMaxEta) {
    return false;
  }

  // introduce momentum cut on tracks
  if (t->getPt() < fMinPt) {
    return false;
  }

  fDebug->Reset();
  int iTrack = t->GetTPCtrackId();
  t->SetChi2(0.f);
  AliHLTTRDpadPlane *pad = nullptr;

#ifdef ENABLE_HLTTRDDEBUG
  HLTTRDTrack *trackNoUpdates = new HLTTRDTrack(*t);
#endif

  // look for matching tracklets via MC label
  int trackID = t->GetLabel();

#ifdef ENABLE_HLTMC
  std::vector<int> matchAvailableAll[kNLayers]; // all available MC tracklet matches for this track
  if (fDebugOutput && trackID > 0 && fMCEvent) {
    CountMatches(trackID, matchAvailableAll);
    bool findableMC[kNLayers] = { false };
    CheckTrackRefs(trackID, findableMC);
    fDebug->SetFindableMC(findableMC);
  }
#endif

  // set input track to first candidate(s)
  fCandidates[0] = *t;
  int nCurrHypothesis = 0;
  int nCandidates = 1;

  // search window
  float roadY = 0.f;
  float roadZ = 0.f;

  fDebug->SetGeneralInfo(fNEvents, nTPCtracks, iTrack, trackID, t->getPt());

  for (int iLayer=0; iLayer<kNLayers; ++iLayer) {

    bool isOK = false; // if at least one candidate could be propagated or the track was stopped this becomes true
    int currIdx = iLayer % 2;
    int nextIdx = (iLayer + 1) % 2;
    pad = fGeo->GetPadPlane(iLayer, 0);
    float tilt = tanf( M_PI / 180. * pad->GetTiltingAngle()); // tilt is signed!
    const float zMaxTRD = pad->GetRowPos(0);

    for (int iCandidate=0; iCandidate<nCandidates; iCandidate++) {

      int det[4] = { -1, -1, -1, -1 }; // TRD chambers to be searched for tracklets

      prop->setTrack(&fCandidates[2*iCandidate+currIdx]);

      if (fCandidates[2*iCandidate+currIdx].GetIsStopped()) {
        if (nCurrHypothesis < fNhypothesis) {
          fHypothesis[nCurrHypothesis].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2();
          fHypothesis[nCurrHypothesis].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
          fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis].fTrackletId = -1;
          nCurrHypothesis++;
        }
        else {
          //std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
          Quicksort(0, nCurrHypothesis - 1, nCurrHypothesis, 1);
          if ( fCandidates[2*iCandidate+currIdx].GetReducedChi2() <
               (fHypothesis[nCurrHypothesis].fChi2 / CAMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
            fHypothesis[nCurrHypothesis-1].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2();
            fHypothesis[nCurrHypothesis-1].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
            fHypothesis[nCurrHypothesis-1].fCandidateId = iCandidate;
            fHypothesis[nCurrHypothesis-1].fTrackletId = -1;
          }
        }
        isOK = true;
        continue;
      }

      // propagate track to average radius of TRD layer iLayer
      if (!prop->PropagateToX(fR[iLayer], fgkMaxSnp, fgkMaxStep)) {
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
      roadY = 7.f * sqrt(fCandidates[2*iCandidate+currIdx].getSigmaY2() + pow(0.10f, 2.f)) + 2.f; // add constant to the road for better efficiency
      //roadZ = 7.f * sqrt(fCandidates[2*iCandidate+currIdx].getSigmaZ2() + power(9.f, 2.f) / 12.f); // take longest pad length
      roadZ = 18.f; // simply twice the longest pad length -> efficiency 99.996%
      //
      if (fabsf(fCandidates[2*iCandidate+currIdx].getZ()) - roadZ >= zMaxTRD ) {
        if (ENABLE_INFO) {
          Info("FollowProlongation", "Track out of TRD acceptance with z=%f in layer %i (eta=%f)",
            fCandidates[2*iCandidate+currIdx].getZ(), iLayer, fCandidates[2*iCandidate+currIdx].getEta());
        }
        continue;
      }

      // determine chamber(s) to be searched for tracklets
      //det.clear();
      FindChambersInRoad(&fCandidates[2*iCandidate+currIdx], roadY, roadZ, iLayer, det, zMaxTRD, prop->getAlpha());

      // track debug information to be stored in case no matching tracklet can be found
      fDebug->SetTrackParameter(fCandidates[2*iCandidate+currIdx], iLayer);

      // look for tracklets in chamber(s)
      bool wasTrackRotated = false;
      for (int iDet = 0; iDet < 4; iDet++) {
        int currDet = det[iDet];
        if (currDet == -1) {
          continue;
        }
        int currSec = fGeo->GetSector(currDet);
        if (currSec != GetSector(prop->getAlpha()) && !wasTrackRotated) {
          float currAlpha = GetAlphaOfSector(currSec);
          if (!prop->rotate(currAlpha)) {
            if (ENABLE_WARNING) {
              Warning("FollowProlongation", "Track could not be rotated in tracklet coordinate system");
            }
            break;
          }
          wasTrackRotated = true; // tracks need to be rotated max. once per layer
        }
        if (currSec != GetSector(prop->getAlpha())) {
          Error("FollowProlongation", "Track is in sector %i and sector %i is searched for tracklets",
                    GetSector(prop->getAlpha()), currSec);
          continue;
        }
        // first propagate track to x of tracklet
        for (int iTrklt=0; iTrklt<fNtrackletsInChamber[currDet]; ++iTrklt) {
          int trkltIdx = fTrackletIndexArray[currDet] + iTrklt;
          if (!prop->PropagateToX(fSpacePoints[trkltIdx].fR, fgkMaxSnp, fgkMaxStep)) {
            if (ENABLE_WARNING) {
              Warning("FollowProlongation", "Track parameter for track %i, x=%f at tracklet %i x=%f in layer %i cannot be retrieved",
                iTrack, fCandidates[2*iCandidate+currIdx].getX(), iTrklt, fSpacePoints[trkltIdx].fR, iLayer);
            }
            continue;
          }
          float zPosCorr = fSpacePoints[trkltIdx].fX[1] + fZCorrCoefNRC * fCandidates[2*iCandidate+currIdx].getTgl();
          float deltaY = fSpacePoints[trkltIdx].fX[0] - fCandidates[2*iCandidate+currIdx].getY();
          float deltaZ = zPosCorr - fCandidates[2*iCandidate+currIdx].getZ();
          float tiltCorr = tilt * (fSpacePoints[trkltIdx].fX[1] - fCandidates[2*iCandidate+currIdx].getZ());
          // tilt correction only makes sense if deltaZ < l_pad && track z err << l_pad
          float l_pad = pad->GetRowSize(fTracklets[trkltIdx].GetZbin());
          if ( (fabsf(fSpacePoints[trkltIdx].fX[1] - fCandidates[2*iCandidate+currIdx].getZ()) <  l_pad) &&
               (fCandidates[2*iCandidate+currIdx].getSigmaZ2() < (l_pad*l_pad/12.)) )
          {
            deltaY -= tiltCorr;
          }
          My_Float trkltPosTmpYZ[2] = { fSpacePoints[trkltIdx].fX[0] - tiltCorr, zPosCorr };
          if ( (fabsf(deltaY) < roadY) && (fabsf(deltaZ) < roadZ) )
          {
            //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            RecalcTrkltCov(trkltIdx, tilt, fCandidates[2*iCandidate+currIdx].getSnp(), pad->GetRowSize(fTracklets[trkltIdx].GetZbin()));
            float chi2 = prop->getPredictedChi2(trkltPosTmpYZ, fSpacePoints[trkltIdx].fCov);
            if (chi2 < fMaxChi2) {
              if (nCurrHypothesis < fNhypothesis) {
                fHypothesis[nCurrHypothesis].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + chi2;
                fHypothesis[nCurrHypothesis].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
                fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
                fHypothesis[nCurrHypothesis].fTrackletId = trkltIdx;
                nCurrHypothesis++;
              }
              else {
                //std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
                Quicksort(0, nCurrHypothesis - 1, nCurrHypothesis, 1);
                if ( ((chi2 + fCandidates[2*iCandidate+currIdx].GetChi2()) / CAMath::Max(fCandidates[2*iCandidate+currIdx].GetNlayers(), 1)) <
                      (fHypothesis[nCurrHypothesis].fChi2 / CAMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
                  fHypothesis[nCurrHypothesis-1].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + chi2;
                  fHypothesis[nCurrHypothesis-1].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
                  fHypothesis[nCurrHypothesis-1].fCandidateId = iCandidate;
                  fHypothesis[nCurrHypothesis-1].fTrackletId = trkltIdx;
                }
              }
            } // end tracklet chi2 < fMaxChi2
          } // end tracklet in window
        } // tracklet loop
      } // chamber loop

      // add no update to hypothesis list
      if (nCurrHypothesis < fNhypothesis) {
        fHypothesis[nCurrHypothesis].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty;
        fHypothesis[nCurrHypothesis].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
        fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
        fHypothesis[nCurrHypothesis].fTrackletId = -1;
        nCurrHypothesis++;
      }
      else {
        //std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
        Quicksort(0, nCurrHypothesis - 1, nCurrHypothesis, 1);
        if ( ((fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty) / CAMath::Max(fCandidates[2*iCandidate+currIdx].GetNlayers(), 1)) <
             (fHypothesis[nCurrHypothesis].fChi2 / CAMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
          fHypothesis[nCurrHypothesis-1].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty;
          fHypothesis[nCurrHypothesis-1].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
          fHypothesis[nCurrHypothesis-1].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis-1].fTrackletId = -1;
        }
      }
      isOK = true;
    } // end candidate loop

#ifdef ENABLE_HLTMC
    // in case matching tracklet exists in this layer -> store position information for debugging
    if (matchAvailableAll[iLayer].size() > 0 && fDebugOutput) {
      fDebug->SetNmatchAvail(matchAvailableAll[iLayer].size(), iLayer);
      int realTrkltId = matchAvailableAll[iLayer].at(0);
      prop->setTrack(&fCandidates[currIdx]);
      bool flag = prop->PropagateToX(fSpacePoints[realTrkltId].fR, fgkMaxSnp, fgkMaxStep);
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
             (fabsf(fSpacePoints[realTrkltId].fX[1] - fCandidates[currIdx].getZ()) >= l_padReal) )
        {
          tiltCorrReal = 0;
        }
        My_Float yzPosReal[2] = { fSpacePoints[realTrkltId].fX[0] - tiltCorrReal, zPosCorrReal };
        RecalcTrkltCov(realTrkltId, tilt, fCandidates[currIdx].getSnp(), pad->GetRowSize(fTracklets[realTrkltId].GetZbin()));
        fDebug->SetChi2Real(prop->getPredictedChi2(yzPosReal, fSpacePoints[realTrkltId].fCov), iLayer);
        fDebug->SetRawTrackletPositionReal(fSpacePoints[realTrkltId].fR, fSpacePoints[realTrkltId].fX, iLayer);
        fDebug->SetCorrectedTrackletPositionReal(yzPosReal, iLayer);
        fDebug->SetTrackletPropertiesReal(fGeo->GetSector(fTracklets[realTrkltId].GetDetector()), fTracklets[realTrkltId].GetDetector(), iLayer);
      }
    }
#endif
    //
    //std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
    Quicksort(0, nCurrHypothesis - 1, nCurrHypothesis, 1);
    fDebug->SetChi2Update(fHypothesis[0].fChi2 - t->GetChi2(), iLayer); // only meaningful for ONE candidate!!!
    fDebug->SetRoad(roadY, roadZ, iLayer);
    bool wasTrackStored = false;
    //
    // loop over the best N_candidates hypothesis
    //
    for (int iUpdate = 0; iUpdate < nCurrHypothesis && iUpdate < fNCandidates; iUpdate++) {
      if (fHypothesis[iUpdate].fCandidateId == -1) {
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
      fCandidates[2*iUpdate+nextIdx] = fCandidates[2*fHypothesis[iUpdate].fCandidateId+currIdx];
      if (fHypothesis[iUpdate].fTrackletId == -1) {
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
      int trkltSec = fGeo->GetSector(fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector());
      if ( trkltSec != GetSector(prop->getAlpha())) {
        // if after a matching tracklet was found another sector was searched for tracklets the track needs to be rotated back
        prop->rotate( GetAlphaOfSector(trkltSec) );
      }
      if (!prop->PropagateToX(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR, fgkMaxSnp, fgkMaxStep)){
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
      RecalcTrkltCov(fHypothesis[iUpdate].fTrackletId, tilt, fCandidates[2*iUpdate+nextIdx].getSnp(), pad->GetRowSize(fTracklets[fHypothesis[iUpdate].fTrackletId].GetZbin()));

      float zPosCorrUpdate = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1] + fZCorrCoefNRC * fCandidates[2*iUpdate+nextIdx].getTgl();
      float yCorr = 0;
      float l_padTrklt = pad->GetRowSize(fTracklets[fHypothesis[iUpdate].fTrackletId].GetZbin());
      if ( (fCandidates[2*iUpdate+nextIdx].getSigmaZ2() < (l_padTrklt*l_padTrklt/12.f)) &&
           (fabsf(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1] - fCandidates[2*iUpdate+nextIdx].getZ()) < l_padTrklt) )
      {
        yCorr = tilt * (fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1] - fCandidates[2*iUpdate+nextIdx].getZ());
      }
      My_Float trkltPosYZ[2] = { fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0] - yCorr, zPosCorrUpdate };

#ifdef ENABLE_HLTTRDDEBUG
      prop->setTrack(trackNoUpdates);
      prop->rotate(GetAlphaOfSector(trkltSec));
      prop->PropagateToX(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR, fgkMaxSnp, fgkMaxStep);
      prop->setTrack(&fCandidates[2*iUpdate+nextIdx]);
#endif

      if (!wasTrackStored) {
#ifdef ENABLE_HLTTRDDEBUG
        fDebug->SetTrackParameterNoUp(*trackNoUpdates, iLayer);
#endif
        fDebug->SetTrackParameter(fCandidates[2*iUpdate+nextIdx], iLayer);
        fDebug->SetRawTrackletPosition(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX, iLayer);
        fDebug->SetCorrectedTrackletPosition(trkltPosYZ, iLayer);
        fDebug->SetTrackletCovariance(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov, iLayer);
        fDebug->SetTrackletProperties(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fDy, fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector(), iLayer);
        fDebug->SetRoad(roadY, roadZ, iLayer);
        wasTrackStored = true;
      }

      if (!prop->update(trkltPosYZ, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov))
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
          Warning("FollowProlongation", "Track %i has invalid covariance matrix. Aborting track following\n", iTrack);
        }
        return false;
      }
      fCandidates[2*iUpdate+nextIdx].AddTracklet(iLayer, fHypothesis[iUpdate].fTrackletId);
      fCandidates[2*iUpdate+nextIdx].SetChi2(fHypothesis[iUpdate].fChi2);
      fCandidates[2*iUpdate+nextIdx].SetIsFindable(iLayer);
      if (iUpdate == 0) {
        *t = fCandidates[2*iUpdate+nextIdx];
      }
    } // end update loop

    // reset struct with hypothesis
    for (int iHypothesis=0; iHypothesis<=nCurrHypothesis; iHypothesis++) {
      if (iHypothesis == fNhypothesis) {
        break;
      }
      fHypothesis[iHypothesis].fChi2 = 1e4;
      fHypothesis[iHypothesis].fLayers = 0;
      fHypothesis[iHypothesis].fCandidateId = -1;
      fHypothesis[iHypothesis].fTrackletId = -1;
    }
    nCurrHypothesis = 0;
    if (!isOK) {
      if (ENABLE_INFO) {
        Info("FollowProlongation", "Track %i cannot be followed. Stopped in layer %i", iTrack, iLayer);
      }
      return false;
    }
  } // end layer loop

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
            if (lbTracklet == fabsf(trackID)) {
              update[iLy] = 1 + il;
              nMatching++;
              break;
            }
          }
#ifdef ENABLE_HLTMC
          if (update[iLy] < 1 && fMCEvent) {
            // no exact match, check in related labels
            for (int il=0; il<3; il++) {
              if ( (lbTracklet = fSpacePoints[t->GetTracklet(iLy)].fLabel[il]) < 0 ) {
                // no more valid labels
                continue;
              }
              AliMCParticle *mcPart = (AliMCParticle*) fMCEvent->GetTrack(lbTracklet);
              while (mcPart) {
                int motherPart = mcPart->GetMother();
                if (motherPart == fabsf(trackID)) {
                  update[iLy] = 4 + il;
                  nRelated++;
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
#ifdef ENABLE_HLTMC
      AliMCParticle *mcPartDbg = (AliMCParticle*) fMCEvent->GetTrack(trackID);
      if (mcPartDbg) {
        fDebug->SetMCinfo(mcPartDbg->Xv(), mcPartDbg->Yv(), mcPartDbg->Zv(), mcPartDbg->PdgCode());
      }
#endif
    }

    fDebug->SetTrack(*t);
#ifdef ENABLE_HLTTRDDEBUG
    delete trackNoUpdates;
#endif
    fDebug->SetUpdates(update);
    fDebug->Output();
  }

  return true;
}

GPUd() int AliHLTTRDTracker::GetDetectorNumber(const float zPos, const float alpha, const int layer) const
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

GPUd() bool AliHLTTRDTracker::AdjustSector(HLTTRDPropagator *prop, HLTTRDTrack *t, const int layer) const
{
  //--------------------------------------------------------------------
  // rotate track in new sector if necessary and
  // propagate to correct x of layer
  // cancel if track crosses two sector boundaries
  //--------------------------------------------------------------------
  float alpha     = fGeo->GetAlpha();
  float xTmp      = t->getX();
  float y         = t->getY();
  float yMax      = t->getX() * tanf(0.5 * alpha);
  float alphaCurr = t->getAlpha();

  if (fabsf(y) > 2.f * yMax) {
    if (ENABLE_INFO) {
      Info("AdjustSector", "Track %i with pT = %f crossing two sector boundaries at x = %f", t->GetTPCtrackId(), t->getPt(), t->getX());
    }
    return false;
  }

  int nTries = 0;
  while (fabsf(y) > yMax) {
    if (nTries >= 2) {
      return false;
    }
    int sign = (y > 0) ? 1 : -1;
    if (!prop->rotate(alphaCurr + alpha * sign)) {
      return false;
    }
    if (!prop->PropagateToX(xTmp, fgkMaxSnp, fgkMaxStep)) {
      return false;
    }
    y = t->getY();
    ++nTries;
  }
  return true;
}

GPUd() int AliHLTTRDTracker::GetSector(float alpha) const
{
  //--------------------------------------------------------------------
  // TRD sector number for reference system alpha
  //--------------------------------------------------------------------
  if (alpha < 0) {
    alpha += 2. * M_PI;
  }
  return (int) (alpha * kNSectors / (2. * M_PI));
}

GPUd() float AliHLTTRDTracker::GetAlphaOfSector(const int sec) const
{
  //--------------------------------------------------------------------
  // rotation angle for TRD sector sec
  //--------------------------------------------------------------------
  return (2.0f * M_PI / (float) kNSectors * ((float) sec + 0.5f));
}

GPUd() void AliHLTTRDTracker::RecalcTrkltCov(const int trkltIdx, const float tilt, const float snp, const float rowSize)
{
  //--------------------------------------------------------------------
  // recalculate tracklet covariance taking track phi angle into account
  //--------------------------------------------------------------------
  float t2 = tilt * tilt; // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = GetRPhiRes(snp);
  float sz2 = rowSize * rowSize / 12.f;
  fSpacePoints[trkltIdx].fCov[0] = c2 * (sy2 + t2 * sz2);
  fSpacePoints[trkltIdx].fCov[1] = c2 * tilt * (sz2 - sy2);
  fSpacePoints[trkltIdx].fCov[2] = c2 * (t2 * sy2 + sz2);
}

void AliHLTTRDTracker::CountMatches(const int trackID, std::vector<int> *matches) const
{
  //--------------------------------------------------------------------
  // search in all TRD chambers for matching tracklets
  // including all tracklets created by the track and its daughters
  // important: tracklets far away / pointing in different direction of
  // the track should be rejected (or this has to be done afterwards in analysis)
  //--------------------------------------------------------------------
#ifndef HLTCA_GPUCODE
#ifdef ENABLE_HLTMC
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
	      if (lb == fabsf(trackID)) {
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
          if (lb == fabsf(trackID)) {
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

GPUd() void AliHLTTRDTracker::CheckTrackRefs(const int trackID, bool *findableMC) const
{
#ifdef ENABLE_HLTMC
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

GPUd() void AliHLTTRDTracker::FindChambersInRoad(const HLTTRDTrack *t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const
{
  //--------------------------------------------------------------------
  // determine initial chamber where the track ends up
  // add more chambers of the same sector or (and) neighbouring
  // stack if track is close the edge(s) of the chamber
  //--------------------------------------------------------------------

  const float yMax    = fabsf(fGeo->GetCol0(iLayer));

  int currStack = fGeo->GetStack(t->getZ(), iLayer);
  int currSec = GetSector(alpha);
  int currDet;

  int nDets = 0;

  if (currStack > -1) {
    // chamber unambiguous
    currDet = fGeo->GetDetector(iLayer, currStack, currSec);
    det[nDets++] = currDet;
    AliHLTTRDpadPlane *pp = fGeo->GetPadPlane(iLayer, currStack);
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
    if (fabsf(t->getZ()) > zMax) {
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
      currDet = GetDetectorNumber(t->getZ()+ 4.0f, alpha, iLayer);
      if (currDet != -1) {
        det[nDets++] = currDet;
      }
      currDet = GetDetectorNumber(t->getZ()-4.0f, alpha, iLayer);
      if (currDet != -1) {
        det[nDets++] = currDet;
      }
    }
  }
  // add chamber(s) from neighbouring sector in case the track is close to the boundary
  if ( ( fabsf(t->getY()) + roadY ) > yMax ) {
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
    if (fGeo->IsHole(iLayer, fGeo->GetStack(det[iDet]), fGeo->GetSector(det[iDet])) || det[iDet] == 538) {
      det[iDet] = -1;
    }
  }
}

GPUd() bool AliHLTTRDTracker::IsGeoFindable(const HLTTRDTrack *t, const int layer, const float alpha) const
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

  // reject tracks inside masked chambers
  for (int i=0; i<fgkNmaskedChambers; i++) {
    if (det == fMaskedChambers[i]) {
      return false;
    }
  }
  //if (std::find(fMaskedChambers.begin(), fMaskedChambers.end(), det) != fMaskedChambers.end()) {
  //  return false;
  //}

  AliHLTTRDpadPlane *pp = fGeo->GetPadPlane(layer, fGeo->GetStack(det));
  int rowIdx = pp->GetNrows() - 1;
  float yMax = fabsf(pp->GetColPos(0));
  float zMax = pp->GetRowPos(0);
  float zMin = pp->GetRowPos(rowIdx) - pp->GetRowSize(rowIdx);

  // reject tracks closer than 5 cm to pad plane boundary
  if (yMax - fabsf(t->getY()) < 5.f) {
    return false;
  }
  // reject tracks closer than 5 cm to stack boundary
  if ( (zMax - t->getZ() < 5.f) || (t->getZ() - zMin < 5.f) ) {
    return false;
  }

  return true;
}

GPUd() void AliHLTTRDTracker::SwapTracklets(const int left, const int right)
{
  AliHLTTRDTrackletWord tmp = fTracklets[left];
  fTracklets[left] = fTracklets[right];
  fTracklets[right] = tmp;
}

GPUd() int AliHLTTRDTracker::PartitionTracklets(const int left, const int right)
{
  const int mid = left + (right - left) / 2;
  AliHLTTRDTrackletWord pivot = fTracklets[mid];
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

GPUd() void AliHLTTRDTracker::SwapHypothesis(const int left, const int right)
{
  Hypothesis tmp = fHypothesis[left];
  fHypothesis[left] = fHypothesis[right];
  fHypothesis[right] = tmp;
}

GPUd() int AliHLTTRDTracker::PartitionHypothesis(const int left, const int right)
{
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

GPUd() void AliHLTTRDTracker::Quicksort(const int left, const int right, const int size, const int type)
// use quicksort to order the tracklet array (type 0) or the hypothesis array (type 1)
{
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
  Quicksort(left, part - 1, size);
  Quicksort(part + 1, right, size);
}

GPUd() void AliHLTTRDTracker::SetNCandidates(int n)
{
  if (!fIsInitialized) {
    fNCandidates = n;
  } else {
    Error("SetNCandidates", "Cannot change fNCandidates after initialization");
  }
}

GPUd() void AliHLTTRDTracker::PrintSettings() const
{
  printf("Current settings for HLT TRD tracker:\n");
  printf("fMaxChi2(%f), fChi2Penalty(%f), nCandidates(%i), nHypothesisMax(%i), maxMissingLayers(%i)\n",
          fMaxChi2, fChi2Penalty, fNCandidates, fNhypothesis, fMaxMissingLy);
  printf("ptCut = %f GeV, abs(eta) < %f\n", fMinPt, fMaxEta);
}
