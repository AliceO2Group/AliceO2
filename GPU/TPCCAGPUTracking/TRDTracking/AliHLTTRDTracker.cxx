#include <vector>
#include "AliHLTTRDTracker.h"
#include "AliGeomManager.h"
#include "AliTRDgeometry.h"
#include "TDatabasePDG.h"
#include "TGeoMatrix.h"

#include "AliMCParticle.h"

#include "TTreeStream.h"
#include "TVectorF.h"

ClassImp(AliHLTTRDTracker)

// default values taken from AliTRDtrackerV1.cxx
const double AliHLTTRDTracker::fgkX0[kNLayers]    = { 300.2, 312.8, 325.4, 338.0, 350.6, 363.2 };
const double AliHLTTRDTracker::fgkXshift = 0.0;

AliHLTTRDTracker::AliHLTTRDTracker() :
  fR(0x0),
  fIsInitialized(false),
  fTracks(0x0),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fNtrackletsInChamber(0x0),
  fTrackletIndexArray(0x0),
  fHypothesis(0x0),
  fSpacePoints(0x0),
  fGeo(0x0),
  fDebugOutput(false),
  fMinPt(1.0),
  fMaxEta(0.84),
  fMaxChi2(15.0),
  fMaxMissingLy(6),
  fChi2Penalty(10.0),
  fZCorrCoefNRC(1.4),
  fNhypothesis(100),
  fMaskedChambers(0),
  fMCEvent(0),
  fStreamer(0x0)
{
  for (int idx=0; idx<(2*kNcandidates); idx++) {
    fCandidates[idx/kNcandidates][idx%kNcandidates] = 0x0;
  }
  //--------------------------------------------------------------------
  // Default constructor
  //--------------------------------------------------------------------
}

AliHLTTRDTracker::AliHLTTRDTracker(const AliHLTTRDTracker &tracker) :
  fIsInitialized(false),
  fTracks(0x0),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fHypothesis(0x0),
  fSpacePoints(0x0),
  fGeo(0x0),
  fDebugOutput(false),
  fMinPt(1.0),
  fMaxEta(0.84),
  fMaxChi2(15.0),
  fMaxMissingLy(3),
  fChi2Penalty(10.0),
  fZCorrCoefNRC(1.4),
  fNhypothesis(100),
  fMaskedChambers(0),
  fMCEvent(0),
  fStreamer(0x0)
{
  //--------------------------------------------------------------------
  // Copy constructor (dummy!)
  //--------------------------------------------------------------------
  Error("AliHLTTRDTracker", "copy constructor is dummy!");
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = tracker.fR[iLy];
  }
  for (int idx=0; idx<(2*kNcandidates); idx++) {
    fCandidates[idx/kNcandidates][idx%kNcandidates] = tracker.fCandidates[idx/kNcandidates][idx%kNcandidates];
  }
  for (int iDet=0; iDet<kNChambers; iDet++) {
    fNtrackletsInChamber[iDet] = tracker.fNtrackletsInChamber[iDet];
    fTrackletIndexArray[iDet] = tracker.fTrackletIndexArray[iDet];
  }
}

AliHLTTRDTracker & AliHLTTRDTracker::operator=(const AliHLTTRDTracker &tracker){
  //--------------------------------------------------------------------
  // Assignment operator
  //--------------------------------------------------------------------
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
    for (int idx=0; idx<(2*kNcandidates); idx++) {
      delete fCandidates[idx/kNcandidates][idx%kNcandidates];
    }
    delete fGeo;
    if (fDebugOutput) {
      delete fStreamer;
    }
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

  fR = new double[kNLayers];
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
  for (int idx=0; idx<(2*kNcandidates); idx++) {
    fCandidates[idx/kNcandidates][idx%kNcandidates] = new AliHLTTRDTrack();
  }
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

  if (fDebugOutput) {
    Info("Init", "Created streamer for debug information");
    fStreamer = new TTreeSRedirector("TRDhlt.root", "recreate");
  }

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

  // test the correctness of the tracklet index array
  // this can be deleted later...
  //if (!IsTrackletSortingOk()) {
  //  Error("DoTracking", "Bug in tracklet index array");
  //}

  if (!CalculateSpacePoints()) {
    Error("DoTracking", "Space points for at least one chamber could not be calculated");
  }

  delete[] fTracks;
  fNTracks = 0;
  fTracks = new AliHLTTRDTrack[nTPCtracks];

  double piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass();

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

  // currently not needed
  //if (fDebugOutput) {
  //  (*fStreamer) << "eventStats" <<
  //    "nEvents=" << fNEvents <<
  //    "nTrackletsTotal=" << fNTracklets <<
  //    "nTPCtracksTotal=" << nTPCtracks <<
  //    "\n";
  //}

  fNEvents++;
}

bool AliHLTTRDTracker::IsTrackletSortingOk() const
{
  //--------------------------------------------------------------------
  // Check the sorting of the tracklet array (paranoia check)
  //--------------------------------------------------------------------
  int nTrklts = 0;
  for (int iDet=0; iDet<kNChambers; iDet++) {
    for (int iTrklt=0; iTrklt<fNtrackletsInChamber[iDet]; iTrklt++) {
      ++nTrklts;
      int detTracklet = fTracklets[fTrackletIndexArray[iDet]+iTrklt].GetDetector();
      if (iDet != detTracklet) {
        return false;
      }
    }
  }
  if (nTrklts != fNTracklets) {
    return false;
  }
  return true;
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
    //double tilt = TMath::Tan(TMath::DegToRad() * pp->GetTiltingAngle());
    //double t2 = tilt * tilt; // tan^2 (tilt)
    //double c2 = 1. / (1. + t2); // cos^2 (tilt)
    double sy2 = TMath::Power(0.10, 2); // sigma_rphi^2, currently assume sigma_rphi = 1 mm

    for (int iTrklt=0; iTrklt<nTracklets; ++iTrklt) {
      int trkltIdx = fTrackletIndexArray[iDet] + iTrklt;
      int trkltZbin = fTracklets[trkltIdx].GetZbin();
      double sz2 = TMath::Power(pp->GetRowSize(trkltZbin), 2);// / 12.; // sigma_z = l_pad/sqrt(12) TODO trying a larger z error
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
      fSpacePoints[trkltIdx].fCov[0] = sy2;
      fSpacePoints[trkltIdx].fCov[1] = 0;
      fSpacePoints[trkltIdx].fCov[2] = sz2;
      //fSpacePoints[trkltIdx].fCov[0] = c2 * (sy2 + t2 * sz2);
      //fSpacePoints[trkltIdx].fCov[1] = c2 * tilt * (sz2 - sy2);
      //fSpacePoints[trkltIdx].fCov[2] = c2 * (t2 * sy2 + sz2);
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

  int iTrack = t->GetTPCtrackId(); // for debugging individual tracks
  t->SetChi2(0.);

  // only propagate tracks within TRD acceptance
  if (TMath::Abs(t->Eta()) > fMaxEta) {
    return false;
  }

  // introduce momentum cut on tracks
  if (t->Pt() < fMinPt) {
    return false;
  }

  double mass = t->GetMass();
  AliTRDpadPlane *pad = 0x0;

  TVectorF findable(kNLayers);
  TVectorF findableMC(kNLayers);
  TVectorF update(kNLayers);

  TVectorF xPosMC(kNLayers);
  TVectorF yPosMC(kNLayers);
  TVectorF zPosMC(kNLayers);
  TVectorF ptMC(kNLayers);

  AliExternalTrackParam param[kNLayers];
  AliExternalTrackParam paramNoUp[kNLayers];
  AliExternalTrackParam paramOneUp;
  // track properties (for debugging)
  AliHLTTRDTrack *trackNoUpdates = new AliHLTTRDTrack();
  *trackNoUpdates = *t;
  TVectorF trackNoUpX(kNLayers);
  TVectorF trackNoUpY(kNLayers);
  TVectorF trackNoUpZ(kNLayers);
  TVectorF trackNoUpYerr(kNLayers);
  TVectorF trackNoUpZerr(kNLayers);
  TVectorF trackNoUpPhi(kNLayers);
  TVectorF trackNoUpSec(kNLayers);
  TVectorF trackX(kNLayers);
  TVectorF trackY(kNLayers);
  TVectorF trackZ(kNLayers);
  TVectorF trackPt(kNLayers);
  TVectorF trackYerr(kNLayers);
  TVectorF trackZerr(kNLayers);
  TVectorF trackPhi(kNLayers);
  TVectorF trackSec(kNLayers);
  // tracklet properties (used for update)
  TVectorF trackletX(kNLayers);
  TVectorF trackletY(kNLayers);
  TVectorF trackletZ(kNLayers);
  TVectorF trackletDy(kNLayers);
  TVectorF trackletDet(kNLayers);
  TVectorF trackletYerr(kNLayers);
  TVectorF trackletYZerr(kNLayers);
  TVectorF trackletZerr(kNLayers);
  // other debugging params
  TVectorF chi2Update(kNLayers);
  TVectorF trackAngle(kNLayers);
  TVectorF roadTrkY(kNLayers);
  TVectorF roadTrkZ(kNLayers);
  // tracklet properties (for matching tracklets)
  TVectorF nMatchingTracklets(kNLayers);
  TVectorF trackletXReal(kNLayers);
  TVectorF trackletYReal(kNLayers);
  TVectorF trackletZReal(kNLayers);
  TVectorF trackletDetReal(kNLayers);
  TVectorF trackletSecReal(kNLayers);
  TVectorF trackXReal(kNLayers);
  TVectorF trackYReal(kNLayers);
  TVectorF trackZReal(kNLayers);
  TVectorF trackSecReal(kNLayers);
  TVectorF chi2Real(kNLayers);

  // look for matching tracklets via MC label
  int trackID = t->GetLabel();

  std::vector<int> matchAvailableAll[kNLayers]; // all available MC tracklet matches for this track
  if (fDebugOutput && trackID > 0) {
    CountMatches(trackID, matchAvailableAll);
    CheckTrackRefs(trackID, findableMC, xPosMC, yPosMC, zPosMC, ptMC);
  }

  // the vector det holds the numbers of the detectors which are searched for tracklets
  std::vector<int> det;
  std::vector<int>::iterator iDet;

  // set input track to first candidate(s)
  *fCandidates[0][0] = *t;
  int nCurrHypothesis = 0;
  int nCandidates = 1;

  // to help browsing 2D array of candidates
  int currIdx;
  int nextIdx;
  // search window
  double roadY = 0;
  double roadZ = 0;

  for (int iLayer=0; iLayer<kNLayers; ++iLayer) {

    bool isOK = false; // if at least one candidate could be propagated or the track was stopped this becomes true
    currIdx = iLayer % 2;
    nextIdx = (iLayer + 1) % 2;
    pad = fGeo->GetPadPlane(iLayer, 0);
    double tilt = TMath::Tan(TMath::DegToRad() * pad->GetTiltingAngle()); // tilt is signed!
    const float zMaxTRD = pad->GetRowPos(0);

    for (int iCandidate=0; iCandidate<nCandidates; iCandidate++) {

      if (fCandidates[currIdx][iCandidate]->GetIsStopped()) {
        if (nCurrHypothesis < fNhypothesis) {
          fHypothesis[nCurrHypothesis].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2();
          fHypothesis[nCurrHypothesis].fLayers = fCandidates[currIdx][iCandidate]->GetNlayers();
          fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis].fTrackletId = -1;
          nCurrHypothesis++;
        }
        else {
          std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
          if ( (fCandidates[currIdx][iCandidate]->GetChi2() / TMath::Max(fCandidates[currIdx][iCandidate]->GetNlayers(), 1)) <
               (fHypothesis[nCurrHypothesis].fChi2 / TMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
            fHypothesis[nCurrHypothesis-1].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2();
            fHypothesis[nCurrHypothesis-1].fLayers = fCandidates[currIdx][iCandidate]->GetNlayers();
            fHypothesis[nCurrHypothesis-1].fCandidateId = iCandidate;
            fHypothesis[nCurrHypothesis-1].fTrackletId = -1;
          }
        }
        isOK = true;
        continue;
      }

      // propagate track to average radius of TRD layer iLayer
      if (!PropagateTrackToBxByBz(fCandidates[currIdx][iCandidate], fR[iLayer], mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
        Info("FollowProlongation", "Track propagation failed for track %i candidate %i in layer %i (pt=%f)", iTrack, iCandidate, iLayer, fCandidates[currIdx][iCandidate]->Pt());
        continue;
      }

      // rotate track in new sector in case of sector crossing
      if (!AdjustSector(fCandidates[currIdx][iCandidate], iLayer)) {
        Info("FollowProlongation", "Adjusting sector failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        continue;
      }

      // check if track is findable
      if (IsFindable(fCandidates[currIdx][iCandidate], iLayer )) {
        findable(iLayer) = 1;
        fCandidates[currIdx][iCandidate]->SetNlayers(fCandidates[currIdx][iCandidate]->GetNlayers() + 1);
      }

      // define search window
      roadY = 7. * TMath::Sqrt(fCandidates[currIdx][iCandidate]->GetSigmaY2() + TMath::Power(0.10, 2)) + 2; // add constant to the road for better efficiency
      //roadZ = 7. * TMath::Sqrt(fCandidates[currIdx][iCandidate]->GetSigmaZ2() + TMath::Power(9., 2) / 12.); // take longest pad length
      roadZ = 18.; // simply twice the longest pad length -> efficiency 99.996%
      //
      if (TMath::Abs(fCandidates[currIdx][iCandidate]->GetZ()) - roadZ >= zMaxTRD ) {
        Info("FollowProlongation", "Track out of TRD acceptance with z=%f in layer %i (eta=%f)", fCandidates[currIdx][iCandidate]->GetZ(), iLayer, fCandidates[currIdx][iCandidate]->Eta());
        continue;
      }

      // determine chamber(s) to be searched for tracklets
      det.clear();
      FindChambersInRoad(fCandidates[currIdx][iCandidate], roadY, roadZ, iLayer, det, zMaxTRD);

      // track debug information to be stored in case no matching tracklet can be found
        param[iLayer] = *fCandidates[currIdx][iCandidate];
        trackX(iLayer) = fCandidates[currIdx][iCandidate]->GetX();
        trackY(iLayer) = fCandidates[currIdx][iCandidate]->GetY();
        trackZ(iLayer) = fCandidates[currIdx][iCandidate]->GetZ();
        trackPt(iLayer) = fCandidates[currIdx][iCandidate]->Pt();
        trackYerr(iLayer) = fCandidates[currIdx][iCandidate]->GetSigmaY2();
        trackZerr(iLayer) = fCandidates[currIdx][iCandidate]->GetSigmaZ2();
        trackPhi(iLayer) = fCandidates[currIdx][iCandidate]->GetSnp();
        trackSec(iLayer) = GetSector(fCandidates[currIdx][iCandidate]->GetAlpha());
      //

      // look for tracklets in chamber(s)
      bool wasTrackRotated = false;
      for (iDet = det.begin(); iDet != det.end(); ++iDet) {
        int detToSearch = *iDet;
        int sectorToSearch = fGeo->GetSector(detToSearch);
        if (sectorToSearch != GetSector(fCandidates[currIdx][iCandidate]->GetAlpha()) && !wasTrackRotated) {
          float alphaToSearch = GetAlphaOfSector(sectorToSearch);
          if (!fCandidates[currIdx][iCandidate]->Rotate(alphaToSearch)) {
            Error("FollowProlongation", "Track could not be rotated in tracklet coordinate system");
            break;
          }
          wasTrackRotated = true; // tracks need to be rotated max. once per layer
        }
        if (sectorToSearch != GetSector(fCandidates[currIdx][iCandidate]->GetAlpha())) {
          Error("FollowProlongation", "Track is in sector %i and sector %i is searched for tracklets",
                    GetSector(fCandidates[currIdx][iCandidate]->GetAlpha()), sectorToSearch);
          continue;
        }
        // first propagate track to x of tracklet
        for (int iTrklt=0; iTrklt<fNtrackletsInChamber[detToSearch]; ++iTrklt) {
          int trkltIdx = fTrackletIndexArray[detToSearch] + iTrklt;
          if (!PropagateTrackToBxByBz(fCandidates[currIdx][iCandidate], fSpacePoints[trkltIdx].fR, mass, 2.0, kFALSE, 0.8)) {
            Warning("FollowProlongation", "Track parameter for track %i, x=%f at tracklet %i x=%f in layer %i cannot be retrieved", iTrack, fCandidates[currIdx][iCandidate]->GetX(), iTrklt, fSpacePoints[trkltIdx].fR, iLayer);
            continue;
          }
          double zPosCorr = fSpacePoints[trkltIdx].fX[1] + fZCorrCoefNRC * fCandidates[currIdx][iCandidate]->GetTgl();
          double deltaY = fSpacePoints[trkltIdx].fX[0] - fCandidates[currIdx][iCandidate]->GetY();
          double deltaZ = zPosCorr - fCandidates[currIdx][iCandidate]->GetZ();
          double tiltCorr = tilt * deltaZ;
          // tilt correction only makes sense if deltaZ < l_pad
          if ( deltaZ < pad->GetRowSize(fTracklets[trkltIdx].GetZbin()) ) {
            deltaY -= tiltCorr;
          }
          double trkltPosTmpYZ[2] = { fSpacePoints[trkltIdx].fX[0] - tiltCorr, zPosCorr };
          if ( (TMath::Abs(deltaY) < roadY) && (TMath::Abs(deltaZ) < roadZ) )
          {
            //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            RecalcTrkltCov(trkltIdx, tilt, fCandidates[currIdx][iCandidate]->GetSnp(), pad->GetRowSize(fTracklets[trkltIdx].GetZbin()));
            double chi2 = fCandidates[currIdx][iCandidate]->GetPredictedChi2(trkltPosTmpYZ, fSpacePoints[trkltIdx].fCov);
            if (chi2 < fMaxChi2) {
              if (nCurrHypothesis < fNhypothesis) {
                fHypothesis[nCurrHypothesis].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + chi2;
                fHypothesis[nCurrHypothesis].fLayers = fCandidates[currIdx][iCandidate]->GetNlayers();
                fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
                fHypothesis[nCurrHypothesis].fTrackletId = trkltIdx;
                nCurrHypothesis++;
              }
              else {
                std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
                if ( ((chi2 + fCandidates[currIdx][iCandidate]->GetChi2()) / TMath::Max(fCandidates[currIdx][iCandidate]->GetNlayers(), 1)) <
                      (fHypothesis[nCurrHypothesis].fChi2 / TMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
                  fHypothesis[nCurrHypothesis-1].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + chi2;
                  fHypothesis[nCurrHypothesis-1].fLayers = fCandidates[currIdx][iCandidate]->GetNlayers();
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
        fHypothesis[nCurrHypothesis].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + fChi2Penalty;
        fHypothesis[nCurrHypothesis].fLayers = fCandidates[currIdx][iCandidate]->GetNlayers();
        fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
        fHypothesis[nCurrHypothesis].fTrackletId = -1;
        nCurrHypothesis++;
      }
      else {
        std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
        if ( ((fCandidates[currIdx][iCandidate]->GetChi2() + fChi2Penalty) / TMath::Max(fCandidates[currIdx][iCandidate]->GetNlayers(), 1)) <
             (fHypothesis[nCurrHypothesis].fChi2 / TMath::Max(fHypothesis[nCurrHypothesis].fLayers, 1)) ) {
          fHypothesis[nCurrHypothesis-1].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + fChi2Penalty;
          fHypothesis[nCurrHypothesis-1].fLayers = fCandidates[currIdx][iCandidate]->GetNlayers();
          fHypothesis[nCurrHypothesis-1].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis-1].fTrackletId = -1;
        }
      }
      isOK = true;
    } // end candidate loop

    nMatchingTracklets(iLayer) = matchAvailableAll[iLayer].size();
    // in case matching tracklet exists in this layer -> store position information for debugging
    if (nMatchingTracklets(iLayer) > 0) {
      int realTrkltId = matchAvailableAll[iLayer].at(0);
      bool flag = PropagateTrackToBxByBz(fCandidates[currIdx][0], fSpacePoints[realTrkltId].fR, mass, 2.0, kFALSE, 0.8);
      if (flag) {
        flag = AdjustSector(fCandidates[currIdx][0], iLayer);
      }
      if (!flag) {
        Warning("FollowProlongation", "Track parameter at x=%f for track %i at real tracklet x=%f in layer %i cannot be retrieved (pt=%f)", fCandidates[currIdx][0]->GetX(), iTrack, fSpacePoints[realTrkltId].fR, iLayer, fCandidates[currIdx][0]->Pt());
        trackXReal(iLayer) = -9999;
        trackYReal(iLayer) = -9999;
        trackZReal(iLayer) = -9999;
        trackSecReal(iLayer) = -1;
      }
      else {
        trackXReal(iLayer) = fCandidates[currIdx][0]->GetX();
        trackYReal(iLayer) = fCandidates[currIdx][0]->GetY();
        trackZReal(iLayer) = fCandidates[currIdx][0]->GetZ();
        trackSecReal(iLayer) = GetSector(fCandidates[currIdx][0]->GetAlpha());
      }
      double zPosCorrReal = fSpacePoints[realTrkltId].fX[1] + fZCorrCoefNRC * fCandidates[currIdx][0]->GetTgl();
      double yCorrReal = tilt * (zPosCorrReal - fCandidates[currIdx][0]->GetZ());
      double yzPosReal[2] = { fSpacePoints[realTrkltId].fX[0] - yCorrReal, zPosCorrReal };
      RecalcTrkltCov(realTrkltId, tilt, fCandidates[currIdx][0]->GetSnp(), pad->GetRowSize(fTracklets[realTrkltId].GetZbin()));
      chi2Real(iLayer) = fCandidates[currIdx][0]->GetPredictedChi2(yzPosReal, fSpacePoints[realTrkltId].fCov);
      trackletXReal(iLayer) = fSpacePoints[realTrkltId].fR;
      trackletYReal(iLayer) = fSpacePoints[realTrkltId].fX[0] - yCorrReal;
      trackletZReal(iLayer) = zPosCorrReal;
      trackletDetReal(iLayer) = fTracklets[realTrkltId].GetDetector();
      trackletSecReal(iLayer) = fGeo->GetSector(trackletDetReal(iLayer));
    }
    //
    std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
    chi2Update(iLayer) = fHypothesis[0].fChi2 - t->GetChi2(); // only meaningful for ONE candidate!!!
    roadTrkY(iLayer) = roadY; // only meaningful for ONE candidate!!!
    roadTrkZ(iLayer) = roadZ; // only meaningful for ONE candidate!!!
    bool wasTrackStored = false;
    //
    // loop over the best N_candidates hypothesis
    for (int iUpdate = 0; iUpdate < TMath::Min(nCurrHypothesis, kNcandidates); iUpdate++) {
      if (fHypothesis[iUpdate].fCandidateId == -1) {
        // no more candidates
        if (iUpdate == 0) {
          Warning("FollowProlongation", "No valid candidates for track %i in layer %i", iTrack, iLayer);
          nCandidates = 0;
        }
        break;
      }
      nCandidates = iUpdate + 1;
      *fCandidates[nextIdx][iUpdate] = *fCandidates[currIdx][fHypothesis[iUpdate].fCandidateId];
      if (fHypothesis[iUpdate].fTrackletId == -1) {
        // no update for this candidate
        if (findable(iLayer) > 0.5) {
          fCandidates[nextIdx][iUpdate]->SetNmissingConsecLayers(fCandidates[nextIdx][iUpdate]->GetNmissingConsecLayers() + 1);
          if (fCandidates[nextIdx][iUpdate]->GetNmissingConsecLayers() >= fMaxMissingLy) {
            //fCandidates[nextIdx][iUpdate]->SetChi2(fHypothesis[iUpdate].fChi2 + (5. - iLayer) * fChi2Penalty); // FIXME: how to deal with stopped tracks?? penalyze?
            fCandidates[nextIdx][iUpdate]->SetIsStopped();
          }
          else {
            fCandidates[nextIdx][iUpdate]->SetChi2(fHypothesis[iUpdate].fChi2);
          }
        }
        if (iUpdate == 0) {
          *t = *fCandidates[nextIdx][iUpdate];
        }
        continue;
      }
      // best matching tracklet found
      int trkltSec = fGeo->GetSector(fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector());
      if ( trkltSec != GetSector(fCandidates[nextIdx][iUpdate]->GetAlpha())) {
        // if after a matching tracklet was found another sector was searched for tracklets the track needs to be rotated back
        fCandidates[nextIdx][iUpdate]->Rotate( GetAlphaOfSector(trkltSec) );
      }
      if (!PropagateTrackToBxByBz(fCandidates[nextIdx][iUpdate], fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR, mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)){
        Warning("FollowProlongation", "Final track propagation for track %i update %i in layer %i failed", iTrack, iUpdate, iLayer);
        fCandidates[nextIdx][iUpdate]->SetChi2(fCandidates[nextIdx][iUpdate]->GetChi2() + fChi2Penalty);
        if (findable(iLayer) > 0.5) {
          fCandidates[nextIdx][iUpdate]->SetNmissingConsecLayers(fCandidates[nextIdx][iUpdate]->GetNmissingConsecLayers() + 1);
          if (fCandidates[nextIdx][iUpdate]->GetNmissingConsecLayers() >= fMaxMissingLy) {
            fCandidates[nextIdx][iUpdate]->SetIsStopped();
          }
        }
        if (iUpdate == 0) {
          *t = *fCandidates[nextIdx][iUpdate];
        }
        continue;
      }
      RecalcTrkltCov(fHypothesis[iUpdate].fTrackletId, tilt, fCandidates[nextIdx][iUpdate]->GetSnp(), pad->GetRowSize(fTracklets[fHypothesis[iUpdate].fTrackletId].GetZbin()));

      double zPosCorrUpdate = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1] + fZCorrCoefNRC * fCandidates[nextIdx][iUpdate]->GetTgl();
      double yCorr = tilt * (zPosCorrUpdate - fCandidates[nextIdx][iUpdate]->GetZ());
      double trkltPosYZ[2] = { fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0] - yCorr, zPosCorrUpdate };

      trackNoUpdates->Rotate(GetAlphaOfSector(trkltSec));
      PropagateTrackToBxByBz(trackNoUpdates, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR, mass, 2.0, kFALSE, 0.8);

      if (!wasTrackStored) {
        param[iLayer] = *fCandidates[nextIdx][iUpdate];
        paramNoUp[iLayer] = *trackNoUpdates;
        trackNoUpX(iLayer) = trackNoUpdates->GetX();
        trackNoUpY(iLayer) = trackNoUpdates->GetY();
        trackNoUpZ(iLayer) = trackNoUpdates->GetZ();
        trackNoUpYerr(iLayer) = trackNoUpdates->GetSigmaY2();
        trackNoUpZerr(iLayer) = trackNoUpdates->GetSigmaZ2();
        trackNoUpPhi(iLayer) = TMath::ASin(trackNoUpdates->GetSnp());
        trackNoUpSec(iLayer) = GetSector(trackNoUpdates->GetAlpha());
        trackX(iLayer) = fCandidates[nextIdx][iUpdate]->GetX();
        trackY(iLayer) = fCandidates[nextIdx][iUpdate]->GetY();
        trackZ(iLayer) = fCandidates[nextIdx][iUpdate]->GetZ();
        trackPt(iLayer) = fCandidates[nextIdx][iUpdate]->Pt();
        trackYerr(iLayer) = fCandidates[nextIdx][iUpdate]->GetSigmaY2();
        trackZerr(iLayer) = fCandidates[nextIdx][iUpdate]->GetSigmaZ2();
        trackPhi(iLayer) = fCandidates[nextIdx][iUpdate]->GetSnp();
        trackSec(iLayer) = GetSector(fCandidates[nextIdx][iUpdate]->GetAlpha());
        trackletX(iLayer) = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fR;
        trackletY(iLayer) = trkltPosYZ[0];
        trackletZ(iLayer) = trkltPosYZ[1];
        trackletYerr(iLayer) = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov[0];
        trackletYZerr(iLayer) = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov[1];
        trackletZerr(iLayer) = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov[2];
        trackletDy(iLayer) = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fDy;
        trackletDet(iLayer) = fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector();
        roadTrkY(iLayer) = roadY;
        roadTrkZ(iLayer) = roadZ;
        wasTrackStored = true;
      }

      paramOneUp = *trackNoUpdates;
      if (!paramOneUp.Update(trkltPosYZ, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov)) {
        Warning("FollowProlongation", "Failed to update debug track %i with single space point in layer %i", iTrack, iLayer);
      }
      else {
        trackAngle(iLayer) = paramOneUp.GetSnp();
      }

      if (!fCandidates[nextIdx][iUpdate]->Update(trkltPosYZ, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov))
      {
        Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);
        fCandidates[nextIdx][iUpdate]->SetChi2(fCandidates[nextIdx][iUpdate]->GetChi2() + fChi2Penalty);
        if (findable(iLayer) > 0.5) {
          fCandidates[nextIdx][iUpdate]->SetNmissingConsecLayers(fCandidates[nextIdx][iUpdate]->GetNmissingConsecLayers() + 1);
          if (fCandidates[nextIdx][iUpdate]->GetNmissingConsecLayers() >= fMaxMissingLy) {
            fCandidates[nextIdx][iUpdate]->SetIsStopped();
          }
        }
        if (iUpdate == 0) {
          *t = *fCandidates[nextIdx][iUpdate];
        }
        continue;
      }
      fCandidates[nextIdx][iUpdate]->AddTracklet(iLayer, fHypothesis[iUpdate].fTrackletId);
      fCandidates[nextIdx][iUpdate]->SetChi2(fHypothesis[iUpdate].fChi2);
      fCandidates[nextIdx][iUpdate]->SetNmissingConsecLayers(0);
      if (iUpdate == 0) {
        *t = *fCandidates[nextIdx][iUpdate];
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
          update(iLy) = 1 + il;
          nMatching++;
          break;
        }
      }
      if (update(iLy) < 1 && fMCEvent) {
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
              update(iLy) = 50 + il;
              nRelated++;
              break;
            }
            mcPart = motherPart >= 0 ? (AliMCParticle*) fMCEvent->GetTrack(motherPart) : 0;
          }
        }
      }
      if (update(iLy) < 1) {
        update(iLy) = 999;
        nFake++;
        /*
	      printf("FAKE on lr %d for trackID=%d\n",iLy, trackID);
        printf("fake tracklet label[3] = { %i, %i, %i}\n", fSpacePoints[t->GetTracklet(iLy)].fLabel[0], fSpacePoints[t->GetTracklet(iLy)].fLabel[1], fSpacePoints[t->GetTracklet(iLy)].fLabel[2]);
        AliMCParticle *mcPartTrk = (AliMCParticle*) fMCEvent->GetTrack(trackID);
        int levelTrk = 0;
        while (mcPartTrk) {
          int motherPartTrk = mcPartTrk->GetMother();
	        printf("track: Parent %d (level %d)\n",motherPartTrk, levelTrk++);
          mcPartTrk = motherPartTrk >= 0 ? (AliMCParticle*) fMCEvent->GetTrack(motherPartTrk) : 0;
        }
	      for (int il=0; il<3; il++) {
	        int level = 0;
	        if ( (lbTracklet = fSpacePoints[t->GetTracklet(iLy)].fLabel[il]) < 0 ) {
            // no more valid labels
            continue;
          }
          AliMCParticle *mcPartTrklt = (AliMCParticle*) fMCEvent->GetTrack(lbTracklet);
          while (mcPartTrklt) {
            int motherPartTrklt = mcPartTrklt->GetMother();
	          printf("tracklet: Parent %d (level %d)\n",motherPartTrklt, level++);
            mcPartTrklt = motherPartTrklt >= 0 ? (AliMCParticle*) fMCEvent->GetTrack(motherPartTrklt) : 0;
          }
        }
        */
      }
    }
  }

  if (fDebugOutput) {
    double XvMC = 0, YvMC = 0, ZvMC = 0;
    int pdgCode = 0;
    if (fMCEvent) {
      AliMCParticle *mcPartDbg = (AliMCParticle*) fMCEvent->GetTrack(trackID);
      if (mcPartDbg) {
        XvMC = mcPartDbg->Xv();
        YvMC = mcPartDbg->Yv();
        ZvMC = mcPartDbg->Zv();
        pdgCode = mcPartDbg->PdgCode();
      }
    }
    double chi2Total = t->GetChi2();
    int nTracklets = t->GetNtracklets();
    int nLayers = t->GetNlayers();
    int nTrackletsOffline = t->GetNtrackletsOffline();
    AliExternalTrackParam parameterFinal(*t);
    AliExternalTrackParam parameterNoUpdates(*trackNoUpdates);
    (*fStreamer) << "tracksFinal" <<
      "event=" << fNEvents <<                           // event number
      "nTPCtracks=" << nTPCtracks <<                    // total number of TPC tracks for this event
      "trackID=" << trackID <<                          // MC track label
      "iTrack=" << iTrack <<                            // track number in event
      "nTracklets=" << nTracklets <<                    // number of attached tracklets
      "nLayers=" << nLayers <<                          // number of layers in which track was findable
      "nTrackletsOffline=" << nTrackletsOffline <<      // number of attached offline tracklets (if provided)
      "chi2Total=" << chi2Total <<                      // total chi2 of track
      "nRelated=" << nRelated <<                        // number of attached related tracklets
      "nMatching=" << nMatching <<                      // number of attached matching tracklets
      "nFake=" << nFake <<                              // number of attached fake tracklets
      "update.=" << &update <<                          // 0 - no update, 1+x - match for label x, 50+x - related for label x, 999 - fake
      "trackNoUpX.=" << &trackNoUpX <<                  // track x position for track which is not updated with TRD information
      "trackNoUpY.=" << &trackNoUpY <<                  // track y position for track which is not updated with TRD information
      "trackNoUpZ.=" << &trackNoUpZ <<                  // track z position for track which is not updated with TRD information
      "trackNoUpYerr.=" << &trackNoUpYerr <<            // track sigma_y^2 for track which is not updated with TRD information
      "trackNoUpZerr.=" << &trackNoUpZerr <<            // track sigma_z^2 for track which is not updated with TRD information
      "trackNoUpPhi.=" << &trackNoUpPhi <<              // track phi angle for track which is not updated with TRD information
      "trackNoUpSec.=" << &trackNoUpSec <<              // track sector for track which is not updated with TRD information
      "trackX.=" << &trackX <<                          // x position for track (sector coords)
      "trackY.=" << &trackY <<                          // y position for track (sector coords)
      "trackZ.=" << &trackZ <<                          // z position for track (sector coords)
      "trackPt.=" << &trackPt <<                        // track pT
      "trackYerr.=" << &trackYerr <<                    // sigma_y^2 for track
      "trackZerr.=" << &trackZerr <<                    // sigma_z^2 for track
      "trackPhi.=" << &trackPhi <<                      // phi angle of track (TMath::Asin(track.fP[2]))
      "trackSec.=" << &trackSec <<                      // sector of track
      "trackletX.=" << &trackletX <<                    // x position of tracklet used for update (sector coords)
      "trackletY.=" << &trackletY <<                    // y position of tracklet used for update (sector coords)
      "trackletZ.=" << &trackletZ <<                    // z position of tracklet used for update (sector coords)
      "trackletYerr.=" << &trackletYerr <<              // sigma_y^2 for tracklet
      "trackletYZerr.=" << &trackletYZerr <<            // sigma_yz for tracklet
      "trackletZerr.=" << &trackletZerr <<              // sigma_z^2 for tracklet
      "trackletDy.=" << &trackletDy <<                  // deflection for tracklet
      "trackletDet.=" << &trackletDet <<                // detector of tracklet
      "roadY.=" << &roadTrkY <<                         // search road Y (only valid for 1 candidate!)
      "roadZ.=" << &roadTrkZ <<                         // search road Z (only valid for 1 candidate!)
      "findable.=" << &findable <<                      // 0 - not findable, 1 - findable
      "findableMC.=" << &findableMC <<                  // number of MC hits in given layer
      "trackL0.=" << &param[0] <<                       // track parameters of track in layer 0
      "trackL1.=" << &param[1] <<                       // track parameters of track in layer 1
      "trackL2.=" << &param[2] <<                       // track parameters of track in layer 2
      "trackL3.=" << &param[3] <<                       // track parameters of track in layer 3
      "trackL4.=" << &param[4] <<                       // track parameters of track in layer 4
      "trackL5.=" << &param[5] <<                       // track parameters of track in layer 5
      "trackNoUpL0.=" << &paramNoUp[0] <<               // track parameters of track w/o updates in layer 0
      "trackNoUpL1.=" << &paramNoUp[1] <<               // track parameters of track w/o updates in layer 1
      "trackNoUpL2.=" << &paramNoUp[2] <<               // track parameters of track w/o updates in layer 2
      "trackNoUpL3.=" << &paramNoUp[3] <<               // track parameters of track w/o updates in layer 3
      "trackNoUpL4.=" << &paramNoUp[4] <<               // track parameters of track w/o updates in layer 4
      "trackNoUpL5.=" << &paramNoUp[5] <<               // track parameters of track w/o updates in layer 5
      "trackAngle.=" << &trackAngle <<                  // track angle for track w/ single update
      "track.=" << &parameterFinal <<                   // track parameters of final track
      "trackNoUp.=" << &parameterNoUpdates <<           // track parameters of track without any updates
      "nMatchingTracklets.=" << &nMatchingTracklets <<  // number of matching + related tracklets for this track in each layer
      "trackletXReal.=" << &trackletXReal <<            // x position (sector coords) for matching or related tracklet if available, otherwise 0
      "trackletYReal.=" << &trackletYReal <<            // y position (sector coords) for matching or related tracklet if available, otherwise 0
      "trackletZReal.=" << &trackletZReal <<            // z position (sector coords) for matching or related tracklet if available, otherwise 0
      "trackletSecReal.=" << &trackletSecReal <<        // sector number for matching or related tracklet if available, otherwise -1
      "trackletDetReal.=" << &trackletDetReal <<        // detector number for matching or related tracklet if available, otherwise -1
      "trackXReal.=" << &trackXReal <<                  // track x position at tracklet x if matching or related tracklet available, otherwise 0
      "trackYReal.=" << &trackYReal <<                  // track y position at tracklet x if matching or related tracklet available, otherwise 0
      "trackZReal.=" << &trackZReal <<                  // track z position at tracklet x if matching or related tracklet available, otherwise 0
      "trackSecReal.=" << &trackSecReal <<              // track sector if matching or related tracklet exists, otherwise -1
      "chi2Real.=" << &chi2Real <<                      // chi2 for update with matching or related tracklet, otherwise 0
      "chi2Update.=" << &chi2Update <<                  // chi2 for tracklet which was used for update (FIXME only safe for one candidate)
      "XvMC=" << XvMC <<                                // MC production vertex x
      "YvMC=" << YvMC <<                                // MC production vertex y
      "ZvMC=" << ZvMC <<                                // MC production vertex z
      "xPosMC.=" << &xPosMC <<                          // MC truth position when exiting chamber
      "yPosMC.=" << &yPosMC <<                          // MC truth position when exiting chamber
      "zPosMC.=" << &zPosMC <<                          // MC truth position when exiting chamber
      "ptMC.=" << &ptMC <<                              // MC truth pt
      "pdgCode=" << pdgCode <<                          // MC PID
      "\n";
  }

  return true;
}

int AliHLTTRDTracker::GetDetectorNumber(const double zPos, const double alpha, const int layer) const
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
  double alpha     = fGeo->GetAlpha();
  double xTmp      = t->GetX();
  double y         = t->GetY();
  double yMax      = t->GetX() * TMath::Tan(0.5 * alpha);
  double alphaCurr = t->GetAlpha();

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

int AliHLTTRDTracker::GetSector(double alpha) const
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

void AliHLTTRDTracker::RecalcTrkltCov(const int trkltIdx, const double tilt, const double snp, const double rowSize)
{
  //--------------------------------------------------------------------
  // recalculate tracklet covariance taking track phi angle into account
  //--------------------------------------------------------------------
  double t2 = tilt * tilt; // tan^2 (tilt)
  double c2 = 1. / (1. + t2); // cos^2 (tilt)
  double sy2 = GetRPhiRes(snp);
  double sz2 = rowSize * rowSize / 12.;
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

void AliHLTTRDTracker::CheckTrackRefs(const int trackID, TVectorF &findableMC, TVectorF &xPosMC, TVectorF &yPosMC, TVectorF &zPosMC, TVectorF &ptMC) const
{
  //--------------------------------------------------------------------
  // loop over all track references for the input trackID and store
  // number of hits exiting the TRD chamber for each layer
  // in the vector findableMC
  // used to check up to which TRD layer the track was existing
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
    double xLoc = trackReference->LocalX();
    if (!((trackReference->TestBits(0x1 << 18)) || (trackReference->TestBits(0x1 << 17)))) {
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
    xPosMC(layer) = xLoc;
    yPosMC(layer) = trackReference->LocalY();
    zPosMC(layer) = trackReference->Z();
    ptMC(layer) = trackReference->Pt();
    findableMC(layer) += 1;
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

bool AliHLTTRDTracker::IsFindable(const AliHLTTRDTrack *t, const int layer) const
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

  // reject tracks closer than 10 cm to pad plane boundary
  if (yMax - TMath::Abs(t->GetY()) < 10.) {
    return false;
  }
  // reject tracks closer than 10 cm to stack boundary
  if ( (zMax - t->GetZ() < 10.) || (t->GetZ() - zMin < 10.) ) {
    return false;
  }

  return true;
}
