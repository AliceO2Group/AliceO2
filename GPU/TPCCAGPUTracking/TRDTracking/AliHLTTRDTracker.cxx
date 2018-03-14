#include <vector>
#include "AliHLTTRDTracker.h"
#include "AliGeomManager.h"
#include "AliTRDgeometry.h"
#include "TDatabasePDG.h"
#include "TGeoMatrix.h"

#include "AliMCParticle.h"

//#define ENABLE_HLTTRDDEBUG
#define ENABLE_WARNING 0
#include "AliHLTTRDTrackerDebug.h"

ClassImp(AliHLTTRDTracker)

// default values taken from AliTRDtrackerV1.cxx
const float AliHLTTRDTracker::fgkX0[kNLayers]    = { 300.2, 312.8, 325.4, 338.0, 350.6, 363.2 };
const float AliHLTTRDTracker::fgkXshift = 0.0;

AliHLTTRDTracker::AliHLTTRDTracker() :
  fR(0x0),
  fIsInitialized(false),
  fTracks(0x0),
  fNCandidates(1),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fNtrackletsInChamber(0x0),
  fTrackletIndexArray(0x0),
  fHypothesis(0x0),
  fCandidates(0x0),
  fSpacePoints(0x0),
  fGeo(0x0),
  fDebugOutput(false),
  fMinPt(0.6),
  fMaxEta(0.84),
  fMaxChi2(15.0),
  fMaxMissingLy(6),
  fChi2Penalty(12.0),
  fZCorrCoefNRC(1.4),
  fNhypothesis(100),
  fMaskedChambers(0x0),
  fMCEvent(0x0),
  fDebug(0x0)
{
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
}

AliHLTTRDTracker::AliHLTTRDTracker(const AliHLTTRDTracker &tracker) :
  fR(0x0),
  fIsInitialized(tracker.fIsInitialized),
  fTracks(0x0),
  fNCandidates(tracker.fNCandidates),
  fNTracks(tracker.fNTracks),
  fNEvents(tracker.fNEvents),
  fTracklets(0x0),
  fNtrackletsMax(tracker.fNtrackletsMax),
  fNTracklets(tracker.fNTracklets),
  fNtrackletsInChamber(0x0),
  fTrackletIndexArray(0x0),
  fHypothesis(0x0),
  fCandidates(0x0),
  fSpacePoints(0x0),
  fGeo(0x0),
  fDebugOutput(tracker.fDebugOutput),
  fMinPt(tracker.fMinPt),
  fMaxEta(tracker.fMaxEta),
  fMaxChi2(tracker.fMaxChi2),
  fMaxMissingLy(tracker.fMaxMissingLy),
  fChi2Penalty(tracker.fChi2Penalty),
  fZCorrCoefNRC(tracker.fZCorrCoefNRC),
  fNhypothesis(tracker.fNhypothesis),
  fMaskedChambers(0x0),
  fMCEvent(0x0),
  fDebug(0x0)
{
  //--------------------------------------------------------------------
  // Copy constructor (dummy!)
  //--------------------------------------------------------------------
  Error("AliHLTTRDTracker", "Copy constructor is dummy!");
}

AliHLTTRDTracker & AliHLTTRDTracker::operator=(const AliHLTTRDTracker &tracker){
  //--------------------------------------------------------------------
  // Assignment operator
  //--------------------------------------------------------------------
  Error("AliHLTTRDTracker", "Assignment operator is dummy!");
  this->~AliHLTTRDTracker();
  new(this) AliHLTTRDTracker(tracker);
  return *this;
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
    delete fGeo;
    delete fDebug;
  }
}


void AliHLTTRDTracker::Init()
{
  //--------------------------------------------------------------------
  // Initialise tracker
  //--------------------------------------------------------------------
  if(!AliGeomManager::GetGeometry()){
    Error("Init", "Could not get geometry.");
  }

  fNtrackletsInChamber = new int[kNChambers];
  fTrackletIndexArray = new int[kNChambers];
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fNtrackletsInChamber[iDet] = 0;
    fTrackletIndexArray[iDet] = -1;
  }

  fR = new float[kNLayers];
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = fgkX0[iLy];
  }

  //FIXME hard-coded masked chambers -> this should eventually be taken from the OCDB (but this is not available before calibration...)
  // at least permanently dead chambers can be hard-coded for the future, but not for older runs
  // to be re-evaluated after the repair in LS2 is done

  // run 244340 (pp)
  //int maskedChambers[] = { 5, 17, 26, 27, 40, 43, 50, 55, 113, 132, 181, 219, 221, 226, 227, 228, 230, 231, 233, 236, 238,
  //                          241, 249, 265, 277, 287, 302, 308, 311, 318, 319, 320, 335, 368, 377, 389, 402, 403, 404, 405,
  //                          406, 407, 432, 433, 434, 435, 436, 437, 452, 461, 462, 463, 464, 465, 466, 467, 570, 490, 491,
  //                          493, 494, 500, 504, 538 };
  // run 245353 (PbPb)
  int maskedChambers[] = { 5, 17, 26, 27, 32, 40, 41, 43, 50, 55, 113, 132, 181, 219, 221, 226, 227, 228, 230, 231, 232, 233,
                            236, 238, 241, 249, 265, 277, 287, 302, 308, 311, 318, 317, 320, 335, 368, 377, 389, 402, 403, 404,
                            405, 406, 407, 432, 433, 434, 435, 436, 437, 452, 461, 462, 463, 464, 465, 466, 467, 470, 490, 491,
                            493, 494, 500, 502, 504, 538 };

  int nChambersMasked = sizeof(maskedChambers) / sizeof(int);
  fMaskedChambers.insert(fMaskedChambers.begin(), maskedChambers, maskedChambers+nChambersMasked);

  fTracklets = new AliHLTTRDTrackletWord[fNtrackletsMax];
  fHypothesis = new Hypothesis[fNhypothesis];
  for (int iHypo=0; iHypo < fNhypothesis; iHypo++) {
    fHypothesis[iHypo].fChi2 = 1e4;
    fHypothesis[iHypo].fLayers = 0;
    fHypothesis[iHypo].fCandidateId = -1;
    fHypothesis[iHypo].fTrackletId = -1;
  }
  fCandidates = new AliHLTTRDTrack[2*fNCandidates];
  fSpacePoints = new AliHLTTRDSpacePointInternal[fNtrackletsMax];

  fGeo = new AliTRDgeometry();
  if (!fGeo) {
    Error("Init", "TRD geometry could not be loaded");
  }

  fGeo->CreateClusterMatrixArray();
  TGeoHMatrix *matrix = 0x0;
  double loc[3] = { fGeo->AnodePos() + fgkXshift, 0., 0. };
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

void AliHLTTRDTracker::Reset()
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

void AliHLTTRDTracker::StartLoadTracklets(const int nTrklts)
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

void AliHLTTRDTracker::LoadTracklet(const AliHLTTRDTrackletWord &tracklet)
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

void AliHLTTRDTracker::DoTracking( AliExternalTrackParam *tracksTPC, int *tracksTPClab, int nTPCtracks, int *tracksTPCnTrklts )
{
  //--------------------------------------------------------------------
  // Steering function for the tracking
  //--------------------------------------------------------------------

  // sort tracklets and fill index array
  std::sort(fTracklets, fTracklets + fNTracklets);
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
  fNTracks = 0;
  fTracks = new AliHLTTRDTrack[nTPCtracks];

  float piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass();

  for (int i=0; i<nTPCtracks; ++i) {
    AliHLTTRDTrack tMI(tracksTPC[i]);
    AliHLTTRDTrack *t = &tMI;
    t->SetTPCtrackId(i);
    t->SetLabel(tracksTPClab[i]);
    if (tracksTPCnTrklts != 0x0) {
      t->SetNtrackletsOffline(tracksTPCnTrklts[i]);
    }

    //FIXME can this be deleted? Or can it happen that a track has no mass assigned?
    //if (TMath::Abs(t->GetMass() - piMass) > 1e-4) {
    //  Warning("DoTracking", "Particle mass (%f) deviates from pion mass (%f)", t->GetMass(), piMass);
    //  t->SetMass(piMass);
    //}

    FollowProlongation(t, nTPCtracks);
    fTracks[fNTracks++] = *t;
  }

  fNEvents++;
}


bool AliHLTTRDTracker::CalculateSpacePoints()
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
    AliTRDpadPlane *pp = fGeo->GetPadPlane(iDet);
    float tilt = TMath::Tan(TMath::DegToRad() * pp->GetTiltingAngle());
    float t2 = tilt * tilt; // tan^2 (tilt)
    float c2 = 1. / (1. + t2); // cos^2 (tilt)
    float sy2 = TMath::Power(0.10, 2); // sigma_rphi^2, currently assume sigma_rphi = 1 mm

    for (int iTrklt=0; iTrklt<nTracklets; ++iTrklt) {
      int trkltIdx = fTrackletIndexArray[iDet] + iTrklt;
      int trkltZbin = fTracklets[trkltIdx].GetZbin();
      float sz2 = TMath::Power(pp->GetRowSize(trkltZbin), 2) / 12.; // sigma_z = l_pad/sqrt(12) TODO try a larger z error
      double xTrkltDet[3] = { 0. }; // trklt position in chamber coordinates
      double xTrkltSec[3] = { 0. }; // trklt position in sector coordinates
      xTrkltDet[0] = fGeo->AnodePos() + fgkXshift;
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

      AliGeomManager::ELayerID iLayer = AliGeomManager::ELayerID(AliGeomManager::kTRD1+fGeo->GetLayer(iDet));
      int modId   = fGeo->GetSector(iDet) * AliTRDgeometry::kNstack + fGeo->GetStack(iDet); // global TRD stack number
      unsigned short volId = AliGeomManager::LayerToVolUID(iLayer, modId);
      fSpacePoints[trkltIdx].fVolumeId = volId;
    }
  }
  return result;
}


bool AliHLTTRDTracker::FollowProlongation(AliHLTTRDTrack *t, int nTPCtracks)
{
  //--------------------------------------------------------------------
  // Propagate TPC track layerwise through TRD and pick up closest
  // tracklet(s) on the way
  // -> returns false if prolongation could not be executed fully
  //    or track does not fullfill threshold conditions
  //--------------------------------------------------------------------

  // only propagate tracks within TRD acceptance
  if (TMath::Abs(t->Eta()) > fMaxEta) {
    return false;
  }

  // introduce momentum cut on tracks
  if (t->Pt() < fMinPt) {
    return false;
  }

  fDebug->Reset();

  int iTrack = t->GetTPCtrackId();
  float mass = t->GetMass();

  t->SetChi2(0.);

  AliTRDpadPlane *pad = 0x0;

#ifdef ENABLE_HLTTRDDEBUG
  AliHLTTRDTrack *trackNoUpdates = new AliHLTTRDTrack(*t);
#endif

  // look for matching tracklets via MC label
  int trackID = t->GetLabel();

  std::vector<int> matchAvailableAll[kNLayers]; // all available MC tracklet matches for this track
  if (fDebugOutput && trackID > 0 && fMCEvent) {
    CountMatches(trackID, matchAvailableAll);
    bool findableMC[6] = { false };
    CheckTrackRefs(trackID, findableMC);
    fDebug->SetFindableMC(findableMC);
  }

  // the vector det holds the numbers of the detectors which are searched for tracklets
  std::vector<int> det;
  std::vector<int>::iterator iDet;

  // set input track to first candidate(s)
  fCandidates[0] = *t;
  int nCurrHypothesis = 0;
  int nCandidates = 1;

  // search window
  float roadY = 0;
  float roadZ = 0;

  fDebug->SetGeneralInfo(fNEvents, nTPCtracks, iTrack, trackID);

  for (int iLayer=0; iLayer<kNLayers; ++iLayer) {

    bool isOK = false; // if at least one candidate could be propagated or the track was stopped this becomes true
    int currIdx = iLayer % 2;
    int nextIdx = (iLayer + 1) % 2;
    pad = fGeo->GetPadPlane(iLayer, 0);
    float tilt = TMath::Tan(TMath::DegToRad() * pad->GetTiltingAngle()); // tilt is signed!
    const float zMaxTRD = pad->GetRowPos(0);

    for (int iCandidate=0; iCandidate<nCandidates; iCandidate++) {

      if (fCandidates[2*iCandidate+currIdx].GetIsStopped()) {
        if (nCurrHypothesis < fNhypothesis) {
          fHypothesis[nCurrHypothesis].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2();
          fHypothesis[nCurrHypothesis].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
          fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis].fTrackletId = -1;
          nCurrHypothesis++;
        }
        else {
          std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
          if ( (fCandidates[2*iCandidate+currIdx].GetChi2() / TMath::Max(fCandidates[2*iCandidate+currIdx].GetNlayers(), 1)) <
               (fHypothesis[nCurrHypothesis].fChi2 / TMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
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
      if (!PropagateTrackToBxByBz(&fCandidates[2*iCandidate+currIdx], fR[iLayer], mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
        Info("FollowProlongation", "Track propagation failed for track %i candidate %i in layer %i (pt=%f)",
          iTrack, iCandidate, iLayer, fCandidates[2*iCandidate+currIdx].Pt());
        continue;
      }

      // rotate track in new sector in case of sector crossing
      if (!AdjustSector(&fCandidates[2*iCandidate+currIdx], iLayer)) {
        Info("FollowProlongation", "Adjusting sector failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        continue;
      }

      // check if track is findable
      if (IsGeoFindable(&fCandidates[2*iCandidate+currIdx], iLayer )) {
        fCandidates[2*iCandidate+currIdx].SetIsFindable(iLayer);
      }

      // define search window
      roadY = 7. * TMath::Sqrt(fCandidates[2*iCandidate+currIdx].GetSigmaY2() + TMath::Power(0.10, 2)) + 2; // add constant to the road for better efficiency
      //roadZ = 7. * TMath::Sqrt(fCandidates[2*iCandidate+currIdx].GetSigmaZ2() + TMath::Power(9., 2) / 12.); // take longest pad length
      roadZ = 18.; // simply twice the longest pad length -> efficiency 99.996%
      //
      if (TMath::Abs(fCandidates[2*iCandidate+currIdx].GetZ()) - roadZ >= zMaxTRD ) {
        Info("FollowProlongation", "Track out of TRD acceptance with z=%f in layer %i (eta=%f)",
          fCandidates[2*iCandidate+currIdx].GetZ(), iLayer, fCandidates[2*iCandidate+currIdx].Eta());
        continue;
      }

      // determine chamber(s) to be searched for tracklets
      det.clear();
      FindChambersInRoad(&fCandidates[2*iCandidate+currIdx], roadY, roadZ, iLayer, det, zMaxTRD);

      // track debug information to be stored in case no matching tracklet can be found
      fDebug->SetTrackParameter(fCandidates[2*iCandidate+currIdx], iLayer);

      // look for tracklets in chamber(s)
      bool wasTrackRotated = false;
      for (iDet = det.begin(); iDet != det.end(); ++iDet) {
        int detToSearch = *iDet;
        int sectorToSearch = fGeo->GetSector(detToSearch);
        if (sectorToSearch != GetSector(fCandidates[2*iCandidate+currIdx].GetAlpha()) && !wasTrackRotated) {
          float alphaToSearch = GetAlphaOfSector(sectorToSearch);
          if (!fCandidates[2*iCandidate+currIdx].Rotate(alphaToSearch)) {
            Error("FollowProlongation", "Track could not be rotated in tracklet coordinate system");
            break;
          }
          wasTrackRotated = true; // tracks need to be rotated max. once per layer
        }
        if (sectorToSearch != GetSector(fCandidates[2*iCandidate+currIdx].GetAlpha())) {
          Error("FollowProlongation", "Track is in sector %i and sector %i is searched for tracklets",
                    GetSector(fCandidates[2*iCandidate+currIdx].GetAlpha()), sectorToSearch);
          continue;
        }
        // first propagate track to x of tracklet
        for (int iTrklt=0; iTrklt<fNtrackletsInChamber[detToSearch]; ++iTrklt) {
          int trkltIdx = fTrackletIndexArray[detToSearch] + iTrklt;
          if (!PropagateTrackToBxByBz(&fCandidates[2*iCandidate+currIdx], fSpacePoints[trkltIdx].fR, mass, 2.0, kFALSE, 0.8)) {
            if (ENABLE_WARNING) {Warning("FollowProlongation", "Track parameter for track %i, x=%f at tracklet %i x=%f in layer %i cannot be retrieved",
              iTrack, fCandidates[2*iCandidate+currIdx].GetX(), iTrklt, fSpacePoints[trkltIdx].fR, iLayer);}
            continue;
          }
          float zPosCorr = fSpacePoints[trkltIdx].fX[1] + fZCorrCoefNRC * fCandidates[2*iCandidate+currIdx].GetTgl();
          float deltaY = fSpacePoints[trkltIdx].fX[0] - fCandidates[2*iCandidate+currIdx].GetY();
          float deltaZ = zPosCorr - fCandidates[2*iCandidate+currIdx].GetZ();
          float tiltCorr = tilt * (fSpacePoints[trkltIdx].fX[1] - fCandidates[2*iCandidate+currIdx].GetZ());
          // tilt correction only makes sense if deltaZ < l_pad && track z err << l_pad
          float l_pad = pad->GetRowSize(fTracklets[trkltIdx].GetZbin());
          if ( (TMath::Abs(fSpacePoints[trkltIdx].fX[1] - fCandidates[2*iCandidate+currIdx].GetZ()) <  l_pad) &&
               (fCandidates[2*iCandidate+currIdx].GetSigmaZ2() < (l_pad*l_pad/12.)) )
          {
            deltaY -= tiltCorr;
          }
          double trkltPosTmpYZ[2] = { fSpacePoints[trkltIdx].fX[0] - tiltCorr, zPosCorr };
          if ( (TMath::Abs(deltaY) < roadY) && (TMath::Abs(deltaZ) < roadZ) )
          {
            //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            RecalcTrkltCov(trkltIdx, tilt, fCandidates[2*iCandidate+currIdx].GetSnp(), pad->GetRowSize(fTracklets[trkltIdx].GetZbin()));
            float chi2 = fCandidates[2*iCandidate+currIdx].GetPredictedChi2(trkltPosTmpYZ, fSpacePoints[trkltIdx].fCov);
            if (chi2 < fMaxChi2) {
              if (nCurrHypothesis < fNhypothesis) {
                fHypothesis[nCurrHypothesis].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + chi2;
                fHypothesis[nCurrHypothesis].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
                fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
                fHypothesis[nCurrHypothesis].fTrackletId = trkltIdx;
                nCurrHypothesis++;
              }
              else {
                std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
                if ( ((chi2 + fCandidates[2*iCandidate+currIdx].GetChi2()) / TMath::Max(fCandidates[2*iCandidate+currIdx].GetNlayers(), 1)) <
                      (fHypothesis[nCurrHypothesis].fChi2 / TMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
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
        std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
        if ( ((fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty) / TMath::Max(fCandidates[2*iCandidate+currIdx].GetNlayers(), 1)) <
             (fHypothesis[nCurrHypothesis].fChi2 / TMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
          fHypothesis[nCurrHypothesis-1].fChi2 = fCandidates[2*iCandidate+currIdx].GetChi2() + fChi2Penalty;
          fHypothesis[nCurrHypothesis-1].fLayers = fCandidates[2*iCandidate+currIdx].GetNlayers();
          fHypothesis[nCurrHypothesis-1].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis-1].fTrackletId = -1;
        }
      }
      isOK = true;
    } // end candidate loop

    // in case matching tracklet exists in this layer -> store position information for debugging
    if (matchAvailableAll[iLayer].size() > 0 && fDebugOutput) {
      fDebug->SetNmatchAvail(matchAvailableAll[iLayer].size(), iLayer);
      int realTrkltId = matchAvailableAll[iLayer].at(0);
      bool flag = PropagateTrackToBxByBz(&fCandidates[currIdx], fSpacePoints[realTrkltId].fR, mass, 2.0, kFALSE, 0.8);
      if (flag) {
        flag = AdjustSector(&fCandidates[currIdx], iLayer);
      }
      if (!flag) {
        if (ENABLE_WARNING) {Warning("FollowProlongation", "Track parameter at x=%f for track %i at real tracklet x=%f in layer %i cannot be retrieved (pt=%f)",
          fCandidates[currIdx].GetX(), iTrack, fSpacePoints[realTrkltId].fR, iLayer, fCandidates[currIdx].Pt());}
      }
      else {
        fDebug->SetTrackParameterReal(fCandidates[currIdx], iLayer);
        float zPosCorrReal = fSpacePoints[realTrkltId].fX[1] + fZCorrCoefNRC * fCandidates[currIdx].GetTgl();
        float deltaZReal = zPosCorrReal - fCandidates[currIdx].GetZ();
        float tiltCorrReal = tilt * (fSpacePoints[realTrkltId].fX[1] - fCandidates[currIdx].GetZ());
        float l_padReal = pad->GetRowSize(fTracklets[realTrkltId].GetZbin());
        if ( (fCandidates[currIdx].GetSigmaZ2() >= (l_padReal*l_padReal/12.)) ||
             (TMath::Abs(fSpacePoints[realTrkltId].fX[1] - fCandidates[currIdx].GetZ()) >= l_padReal) )
        {
          tiltCorrReal = 0;
        }
        double yzPosReal[2] = { fSpacePoints[realTrkltId].fX[0] - tiltCorrReal, zPosCorrReal };
        RecalcTrkltCov(realTrkltId, tilt, fCandidates[currIdx].GetSnp(), pad->GetRowSize(fTracklets[realTrkltId].GetZbin()));
        fDebug->SetChi2Real(fCandidates[currIdx].GetPredictedChi2(yzPosReal, fSpacePoints[realTrkltId].fCov), iLayer);
        fDebug->SetRawTrackletPositionReal(fSpacePoints[realTrkltId].fR, fSpacePoints[realTrkltId].fX, iLayer);
        fDebug->SetCorrectedTrackletPositionReal(yzPosReal, iLayer);
        fDebug->SetTrackletPropertiesReal(fTracklets[realTrkltId].GetDetector(), fGeo->GetSector(fTracklets[realTrkltId].GetDetector()), iLayer);
      }
    }
    //
    std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
    fDebug->SetChi2Update(fHypothesis[0].fChi2 - t->GetChi2(), iLayer); // only meaningful for ONE candidate!!!
    fDebug->SetRoad(roadY, roadZ, iLayer);
    bool wasTrackStored = false;
    //
    // loop over the best N_candidates hypothesis
    //
    for (int iUpdate = 0; iUpdate < TMath::Min(nCurrHypothesis, fNCandidates); iUpdate++) {
      if (fHypothesis[iUpdate].fCandidateId == -1) {
        // no more candidates
        if (iUpdate == 0) {
          if (ENABLE_WARNING) {Warning("FollowProlongation", "No valid candidates for track %i in layer %i", iTrack, iLayer);}
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
      int trkltSec = fGeo->GetSector(fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector());
      if ( trkltSec != GetSector(fCandidates[2*iUpdate+nextIdx].GetAlpha())) {
        // if after a matching tracklet was found another sector was searched for tracklets the track needs to be rotated back
        fCandidates[2*iUpdate+nextIdx].Rotate( GetAlphaOfSector(trkltSec) );
      }
      if (!PropagateTrackToBxByBz(&fCandidates[2*iUpdate+nextIdx], fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR, mass, 2.0, kFALSE, 0.8)){
        if (ENABLE_WARNING) {Warning("FollowProlongation", "Final track propagation for track %i update %i in layer %i failed", iTrack, iUpdate, iLayer);}
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
      RecalcTrkltCov(fHypothesis[iUpdate].fTrackletId, tilt, fCandidates[2*iUpdate+nextIdx].GetSnp(), pad->GetRowSize(fTracklets[fHypothesis[iUpdate].fTrackletId].GetZbin()));

      float zPosCorrUpdate = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1] + fZCorrCoefNRC * fCandidates[2*iUpdate+nextIdx].GetTgl();
      float deltaZup = zPosCorrUpdate - fCandidates[2*iUpdate+nextIdx].GetZ();
      float yCorr = 0;
      float l_padTrklt = pad->GetRowSize(fTracklets[fHypothesis[iUpdate].fTrackletId].GetZbin());
      if ( (fCandidates[2*iUpdate+nextIdx].GetSigmaZ2() < (l_padTrklt*l_padTrklt/12.)) &&
           (TMath::Abs(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1] - fCandidates[2*iUpdate+nextIdx].GetZ()) < l_padTrklt) )
      {
        yCorr = tilt * (fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1] - fCandidates[2*iUpdate+nextIdx].GetZ());
      }
      double trkltPosYZ[2] = { fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0] - yCorr, zPosCorrUpdate };

#ifdef ENABLE_HLTTRDDEBUG
      trackNoUpdates->Rotate(GetAlphaOfSector(trkltSec));
      PropagateTrackToBxByBz(trackNoUpdates, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR, mass, 2.0, kFALSE, 0.8);
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

      if (!fCandidates[2*iUpdate+nextIdx].Update(trkltPosYZ, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov))
      {
        if (ENABLE_WARNING) {Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);}
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
      Info("FollowProlongation", "Track %i cannot be followed. Stopped in layer %i", iTrack, iLayer);
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
            if (lbTracklet == TMath::Abs(trackID)) {
              update[iLy] = 1 + il;
              nMatching++;
              break;
            }
          }
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
                if (motherPart == TMath::Abs(trackID)) {
                  update[iLy] = 4 + il;
                  nRelated++;
                  break;
                }
                mcPart = motherPart >= 0 ? (AliMCParticle*) fMCEvent->GetTrack(motherPart) : 0;
              }
            }
          }
          if (update[iLy] < 1) {
            update[iLy] = 9;
            nFake++;
          }
        }
      }
      fDebug->SetTrackProperties(nMatching, nFake, nRelated);
      AliMCParticle *mcPartDbg = (AliMCParticle*) fMCEvent->GetTrack(trackID);
      if (mcPartDbg) {
        fDebug->SetMCinfo(mcPartDbg->Xv(), mcPartDbg->Yv(), mcPartDbg->Zv(), mcPartDbg->PdgCode());
      }
    }

    fDebug->SetTrack(*t);
#ifdef ENABLE_HLTTRDDEBUG
    fDebug->SetTrackNoUp(*trackNoUpdates);
#endif
    fDebug->SetUpdates(update);
    fDebug->Output();
  }

  return true;
}

int AliHLTTRDTracker::GetDetectorNumber(const float zPos, const float alpha, const int layer) const
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

bool AliHLTTRDTracker::AdjustSector(AliHLTTRDTrack *t, const int layer) const
{
  //--------------------------------------------------------------------
  // rotate track in new sector if necessary and
  // propagate to correct x of layer
  // cancel if track crosses two sector boundaries
  //--------------------------------------------------------------------
  float alpha     = fGeo->GetAlpha();
  float xTmp      = t->GetX();
  float y         = t->GetY();
  float yMax      = t->GetX() * TMath::Tan(0.5 * alpha);
  float alphaCurr = t->GetAlpha();

  if (TMath::Abs(y) > 2. * yMax) {
    Info("AdjustSector", "Track %i with pT = %f crossing two sector boundaries at x = %f", t->GetTPCtrackId(), t->Pt(), t->GetX());
    return false;
  }

  while (TMath::Abs(y) > yMax) {
    int sign = (y > 0) ? 1 : -1;
    if (!t->Rotate(alphaCurr + alpha * sign)) {
      return false;
    }
    if (!PropagateTrackToBxByBz(t, xTmp, TDatabasePDG::Instance()->GetParticle(211)->Mass(), 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
      return false;
    }
    y = t->GetY();
  }
  return true;
}

int AliHLTTRDTracker::GetSector(float alpha) const
{
  //--------------------------------------------------------------------
  // TRD sector number for reference system alpha
  //--------------------------------------------------------------------
  if (alpha < 0) {
    alpha += 2. * TMath::Pi();
  }
  return (int) (alpha * kNSectors / (2. * TMath::Pi()));
}

float AliHLTTRDTracker::GetAlphaOfSector(const int sec) const
{
  //--------------------------------------------------------------------
  // rotation angle for TRD sector sec
  //--------------------------------------------------------------------
  return (2.0 * TMath::Pi() / (float) kNSectors * ((float) sec + 0.5));
}

void AliHLTTRDTracker::RecalcTrkltCov(const int trkltIdx, const float tilt, const float snp, const float rowSize)
{
  //--------------------------------------------------------------------
  // recalculate tracklet covariance taking track phi angle into account
  //--------------------------------------------------------------------
  float t2 = tilt * tilt; // tan^2 (tilt)
  float c2 = 1. / (1. + t2); // cos^2 (tilt)
  float sy2 = GetRPhiRes(snp);
  float sz2 = rowSize * rowSize / 12.;
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
	      if (lb == TMath::Abs(trackID)) {
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
          if (lb == TMath::Abs(trackID)) {
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
}

void AliHLTTRDTracker::CheckTrackRefs(const int trackID, bool *findableMC) const
{
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
    if (xLoc < 304.) {
      layer = 0;
    }
    else if (xLoc < 317.) {
      layer = 1;
    }
    else if (xLoc < 330.) {
      layer = 2;
    }
    else if (xLoc < 343.) {
      layer = 3;
    }
    else if (xLoc < 356.) {
      layer = 4;
    }
    else if (xLoc < 369.) {
      layer = 5;
    }
    if (layer < 0) {
      Error("CheckTrackRefs", "No layer can be determined");
      printf("x=%f, y=%f, z=%f, layer=%i\n", xLoc, trackReference->LocalY(), trackReference->Z(), layer);
      continue;
    }
    findableMC[layer] = true;
  }
}

void AliHLTTRDTracker::FindChambersInRoad(const AliHLTTRDTrack *t, const float roadY, const float roadZ, const int iLayer, std::vector<int> &det, const float zMax) const
{
  //--------------------------------------------------------------------
  // determine initial chamber where the track ends up
  // add more chambers of the same sector or (and) neighbouring
  // stack if track is close the edge(s) of the chamber
  //--------------------------------------------------------------------

  const float yMax    = TMath::Abs(fGeo->GetCol0(iLayer));

  int currStack = fGeo->GetStack(t->GetZ(), iLayer);
  int currSec = GetSector(t->GetAlpha());
  int currDet;

  if (currStack > -1) {
    // chamber unambiguous
    currDet = fGeo->GetDetector(iLayer, currStack, currSec);
    det.push_back(currDet);
    AliTRDpadPlane *pp = fGeo->GetPadPlane(iLayer, currStack);
    int lastPadRow = fGeo->GetRowMax(iLayer, currStack, 0);
    float zCenter = pp->GetRowPos(lastPadRow / 2);
    if ( ( t->GetZ() + roadZ ) > pp->GetRowPos(0) || ( t->GetZ() - roadZ ) < pp->GetRowPos(lastPadRow) ) {
      int addStack = t->GetZ() > zCenter ? currStack - 1 : currStack + 1;
      if (addStack < kNStacks && addStack > -1) {
        det.push_back(fGeo->GetDetector(iLayer, addStack, currSec));
      }
    }
  }
  else {
    if (TMath::Abs(t->GetZ()) > zMax) {
      // shift track in z so it is in the TRD acceptance
      if (t->GetZ() > 0) {
          currDet = fGeo->GetDetector(iLayer, 0, currSec);
      }
      else {
        currDet = fGeo->GetDetector(iLayer, kNStacks-1, currSec);
      }
      det.push_back(currDet);
      currStack = fGeo->GetStack(currDet);
    }
    else {
      // track in between two stacks, add both surrounding chambers
      // gap between two stacks is 4 cm wide
      currDet = GetDetectorNumber(t->GetZ()+ 4.0, t->GetAlpha(), iLayer);
      if (currDet != -1) {
        det.push_back(currDet);
      }
      currDet = GetDetectorNumber(t->GetZ()-4.0, t->GetAlpha(), iLayer);
      if (currDet != -1) {
        det.push_back(currDet);
      }
    }
  }
  // add chamber(s) from neighbouring sector in case the track is close to the boundary
  if ( ( TMath::Abs(t->GetY()) + roadY ) > yMax ) {
    const int nStacksToSearch = det.size();
    int newSec;
    if (t->GetY() > 0) {
      newSec = (currSec + 1) % kNSectors;
    }
    else {
      newSec = (currSec > 0) ? currSec - 1 : kNSectors - 1;
    }
    for (int idx = 0; idx < nStacksToSearch; ++idx) {
      int currStack = fGeo->GetStack(det.at(idx));
      det.push_back(fGeo->GetDetector(iLayer, currStack, newSec));
    }
  }
  // skip PHOS hole and non-existing chamber 17_4_4
  int last = 0;
  int nDet = det.size();
  for (int iDet = 0; iDet < nDet; iDet++, last++) {
    while (fGeo->IsHole(iLayer, fGeo->GetStack(det.at(iDet)), fGeo->GetSector(det.at(iDet))) || det.at(iDet) == 538) {
      ++iDet;
      if (iDet >= nDet) {
        break;
      }
    }
    if (iDet >= nDet) {
      break;
    }
    det.at(last) = det.at(iDet);
  }
  det.resize(last);
}

bool AliHLTTRDTracker::IsGeoFindable(const AliHLTTRDTrack *t, const int layer) const
{
  //--------------------------------------------------------------------
  // returns true if track position inside active area of the TRD
  // and not too close to the boundaries
  //--------------------------------------------------------------------

  int det = GetDetectorNumber(t->GetZ(), t->GetAlpha(), layer);

  // reject tracks between stacks
  if (det < 0) {
    return false;
  }

  // reject tracks inside masked chambers
  if (std::find(fMaskedChambers.begin(), fMaskedChambers.end(), det) != fMaskedChambers.end()) {
    return false;
  }

  AliTRDpadPlane *pp = fGeo->GetPadPlane(layer, fGeo->GetStack(det));
  int rowIdx = pp->GetNrows() - 1;
  float yMax = TMath::Abs(pp->GetColPos(0));
  float zMax = pp->GetRowPos(0);
  float zMin = pp->GetRowPos(rowIdx) - pp->GetRowSize(rowIdx);

  // reject tracks closer than 5 cm to pad plane boundary
  if (yMax - TMath::Abs(t->GetY()) < 5.) {
    return false;
  }
  // reject tracks closer than 5 cm to stack boundary
  if ( (zMax - t->GetZ() < 5.) || (t->GetZ() - zMin < 5.) ) {
    return false;
  }

  return true;
}

void AliHLTTRDTracker::PrintSettings() const
{
  printf("Current settings for HLT TRD tracker:\n");
  printf("fMaxChi2(%f), fChi2Penalty(%f), nCandidates(%i), nHypothesisMax(%i), maxMissingLayers(%i)\n",
          fMaxChi2, fChi2Penalty, fNCandidates, fNhypothesis, fMaxMissingLy);
  printf("ptCut = %f GeV, abs(eta) < %f\n", fMinPt, fMaxEta);
}
