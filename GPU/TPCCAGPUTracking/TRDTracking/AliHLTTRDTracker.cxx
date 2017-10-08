#include <vector>
#include "AliHLTTRDTracker.h"
#include "AliGeomManager.h"
#include "AliTRDgeometry.h"
#include "TDatabasePDG.h"
#include "TTreeStream.h"
#include "TGeoMatrix.h"

#include "AliMCParticle.h"
#include "TVectorF.h"

ClassImp(AliHLTTRDTracker)

// default values taken from AliTRDtrackerV1.cxx
const double AliHLTTRDTracker::fgkX0[kNLayers]    = { 300.2, 312.8, 325.4, 338.0, 350.6, 363.2 };

AliHLTTRDTracker::AliHLTTRDTracker() :
  fIsInitialized(false),
  fTracks(0x0),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fHypothesis(0x0),
  fSpacePoints(0x0),
  fTRDgeometry(0x0),
  fDebugOutput(false),
  fMinPt(1.0),
  fMaxChi2(22.0),
  fChi2Penalty(15),
  fNhypothesis(100),
  fMaskedChambers(0),
  fMCEvent(0),
  fStreamer(0x0)
{
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = fgkX0[iLy];
  }
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fTrackletIndexArray[iDet][0] = -1;
    fTrackletIndexArray[iDet][1] = 0;
  }
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
  fTRDgeometry(0x0),
  fDebugOutput(false),
  fMinPt(1.0),
  fMaxChi2(22.0),
  fChi2Penalty(15),
  fNhypothesis(100),
  fMaskedChambers(0),
  fMCEvent(0),
  fStreamer(0x0)
{
  //--------------------------------------------------------------------
  // Copy constructor (dummy!)
  //--------------------------------------------------------------------
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = fgkX0[iLy];
  }
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fTrackletIndexArray[iDet][0] = tracker.fTrackletIndexArray[iDet][0];
    fTrackletIndexArray[iDet][1] = tracker.fTrackletIndexArray[iDet][1];
  }
  for (int idx=0; idx<(2*kNcandidates); idx++) {
    fCandidates[idx/kNcandidates][idx%kNcandidates] = tracker.fCandidates[idx/kNcandidates][idx%kNcandidates];
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
    delete fTRDgeometry;
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
    fHypothesis[iHypo].fCandidateId = -1;
    fHypothesis[iHypo].fTrackletId = -1;
  }
  for (int idx=0; idx<(2*kNcandidates); idx++) {
    fCandidates[idx/kNcandidates][idx%kNcandidates] = new AliHLTTRDTrack();
  }
  fSpacePoints = new AliHLTTRDSpacePointInternal[fNtrackletsMax];

  fTRDgeometry = new AliTRDgeometry();
  if (!fTRDgeometry) {
    Error("Init", "TRD geometry could not be loaded");
  }

  fTRDgeometry->CreateClusterMatrixArray();
  TGeoHMatrix *matrix = 0x0;
  double loc[3] = { AliTRDgeometry::AnodePos(), 0., 0. };
  double glb[3] = { 0., 0., 0. };
  for (int iLy=0; iLy<kNLayers; iLy++) {
    for (int iSec=0; iSec<kNSectors; iSec++) {
      matrix = fTRDgeometry->GetClusterMatrix(fTRDgeometry->GetDetector(iLy, 2, iSec));
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
    fStreamer = new TTreeSRedirector("debug.root", "recreate");
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
    fSpacePoints[i].fX[0]     = 0.;
    fSpacePoints[i].fX[1]     = 0.;
    fSpacePoints[i].fX[2]     = 0.;
    fSpacePoints[i].fCov[0]   = 0.;
    fSpacePoints[i].fCov[1]   = 0.;
    fSpacePoints[i].fDy       = 0.;
    fSpacePoints[i].fId       = 0;
    fSpacePoints[i].fVolumeId = 0;
  }
  for (int iDet=0; iDet<540; ++iDet) {
    fTrackletIndexArray[iDet][0] = -1;
    fTrackletIndexArray[iDet][1] = 0;
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
    Error("LoadTracklet", "running out of memory for tracklets, skipping tracklet(s). This should actually never happen.");
    return;
  }
  fTracklets[fNTracklets++] = tracklet;
  fTrackletIndexArray[tracklet.GetDetector()][1]++;
}

void AliHLTTRDTracker::DoTracking( AliExternalTrackParam *tracksTPC, int *tracksTPCLab, int nTPCTracks, int *tracksTPCnTrklts )
{
  //--------------------------------------------------------------------
  // Steering function for the tracking
  //--------------------------------------------------------------------

  fNEvents++;

  // sort tracklets and fill index array
  std::sort(fTracklets, fTracklets + fNTracklets);
  int trkltCounter = 0;
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    if (fTrackletIndexArray[iDet][1] != 0) {
      fTrackletIndexArray[iDet][0] = trkltCounter;
      trkltCounter += fTrackletIndexArray[iDet][1];
    }
  }

  // test the correctness of the tracklet index array
  // this can be deleted later...
  if (!IsTrackletSortingOk()) {
    Error("DoTracking", "bug in tracklet index array");
  }

  if (!CalculateSpacePoints()) {
    Error("DoTracking", "space points for at least one chamber could not be calculated");
  }

  delete[] fTracks;
  fNTracks = 0;
  fTracks = new AliHLTTRDTrack[nTPCTracks];

  double piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass(); // pion mass as best guess for all particles

  for (int i=0; i<nTPCTracks; ++i) {
    AliHLTTRDTrack tMI(tracksTPC[i]);
    AliHLTTRDTrack *t = &tMI;
    t->SetTPCtrackId(i);
    t->SetLabel(tracksTPCLab[i]);
    if (tracksTPCnTrklts != 0x0) {
      t->SetNtrackletsOffline(tracksTPCnTrklts[i]);
    }
    t->SetMass(piMass);

    FollowProlongation(t, piMass);
    fTracks[fNTracks++] = *t;
  }

  if (fDebugOutput) {
    (*fStreamer) << "statistics" <<
      "nEvents=" << fNEvents <<
      "nTrackletsTotal=" << fNTracklets <<
      "nTPCtracksTotal=" << nTPCTracks <<
      "\n";
  }
}

bool AliHLTTRDTracker::IsTrackletSortingOk()
{
  //--------------------------------------------------------------------
  // Check the sorting of the tracklet array (paranoia check)
  //--------------------------------------------------------------------
  int nTrklts = 0;
  for (int iDet=0; iDet<kNChambers; iDet++) {
    for (int iTrklt=0; iTrklt<fTrackletIndexArray[iDet][1]; iTrklt++) {
      ++nTrklts;
      int detTracklet = fTracklets[fTrackletIndexArray[iDet][0]+iTrklt].GetDetector();
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
  // from tracklets
  //--------------------------------------------------------------------

  bool result = true;

  for (int iDet=0; iDet<kNChambers; ++iDet) {

    int layer = fTRDgeometry->GetLayer(iDet);
    int stack = fTRDgeometry->GetStack(iDet);

    int nTracklets = fTrackletIndexArray[iDet][1];
    if (nTracklets == 0) {
      continue;
    }

    TGeoHMatrix *matrix = fTRDgeometry->GetClusterMatrix(iDet);
    if (!matrix){
	    Error("CalculateSpacePoints", "invalid TRD cluster matrix, skipping detector  %i", iDet);
      result = false;
	    continue;
    }
    AliTRDpadPlane *padPlane = fTRDgeometry->GetPadPlane(layer, stack);

    for (int iTrklt=0; iTrklt<nTracklets; ++iTrklt) {
      int trkltIdx = fTrackletIndexArray[iDet][0] + iTrklt;
      double xTrklt[3] = { 0. };
      xTrklt[0] = AliTRDgeometry::AnodePos();
      xTrklt[1] = fTracklets[trkltIdx].GetY();
      if (stack == 2) {
        xTrklt[2] = padPlane->GetRowPos(fTracklets[trkltIdx].GetZbin()) -
                        (padPlane->GetRowSize(fTracklets[trkltIdx].GetZbin()))/2. - padPlane->GetRowPos(6);
      }
      else {
        xTrklt[2] = padPlane->GetRowPos(fTracklets[trkltIdx].GetZbin()) -
                        (padPlane->GetRowSize(fTracklets[trkltIdx].GetZbin()))/2. - padPlane->GetRowPos(8);
      }
      matrix->LocalToMaster(xTrklt, fSpacePoints[trkltIdx].fX);
      fSpacePoints[trkltIdx].fId = fTracklets[trkltIdx].GetId();
      //fSpacePoints[trkltIdx].fLabel = TMath::Abs(fTracklets[trkltIdx].GetLabel()); // RS: why abs? 
      for (int i=3;i--;) fSpacePoints[trkltIdx].fLabel[i] = fTracklets[trkltIdx].GetLabel(i);
      fSpacePoints[trkltIdx].fCov[0] = TMath::Power(0.10, 2);
      fSpacePoints[trkltIdx].fCov[1] = TMath::Power(padPlane->GetRowSize(fTracklets[trkltIdx].GetZbin()), 2) / 12.;
      fSpacePoints[trkltIdx].fDy = 0.014 * fTracklets[trkltIdx].GetdY();

      AliGeomManager::ELayerID iLayer = AliGeomManager::ELayerID(AliGeomManager::kTRD1+fTRDgeometry->GetLayer(iDet));
      int modId   = fTRDgeometry->GetSector(iDet) * AliTRDgeometry::kNstack + fTRDgeometry->GetStack(iDet); // global TRD stack number
      unsigned short volId = AliGeomManager::LayerToVolUID(iLayer, modId);
      fSpacePoints[trkltIdx].fVolumeId = volId;
    }
  }
  return result;
}


bool AliHLTTRDTracker::FollowProlongation(AliHLTTRDTrack *t, double mass)
{
  //--------------------------------------------------------------------
  // Propagate TPC track layerwise through TRD and pick up closest
  // tracklets on the way
  // -> return false if prolongation could not be executed fully
  //--------------------------------------------------------------------

  int iTrack = t->GetTPCtrackId(); // for debugging individual tracks
  t->SetChi2(0.);

  // only propagate tracks within TRD acceptance
  if (TMath::Abs(t->Eta()) > 0.9) {
    return false;
  }

  // introduce momentum cut on tracks
  // particles with pT < 0.9 GeV highly unlikely to have matching online TRD tracklets
  if (t->Pt() < fMinPt) {
    return false;
  }

  //if (fDebugOutput) {
  //  AliExternalTrackParam initialParam(*t);
  //  (*fStreamer) << "initialTracks" <<
  //    "iEv=" << fNEvents <<
  //    "iTrk=" << iTrack <<
  //    "track.=" << &initialParam <<
  //    "\n";
  //}

  TVectorF trackFindable(kNLayers);

  // for debugging purposes
  int lbTrack = TMath::Abs(t->GetLabel());
  TVectorF tracklets(kNLayers);
  std::vector<int> matchAvailableAll[kNLayers]; // all available tracklet matches for this track
  std::vector<int> relatives; // all related labels for this track (only up to daughter's daughters...)

  if (fMCEvent) {
    AliMCParticle *mcParticle = (AliMCParticle*) fMCEvent->GetTrack(lbTrack);
    if (mcParticle != 0x0) {
      for (int iDaughter = 0; iDaughter < mcParticle->GetNDaughters(); iDaughter++) {
        int lbDaughter = mcParticle->GetDaughterLabel(iDaughter);
        relatives.push_back(TMath::Abs(lbDaughter));
        AliMCParticle *mcDaughter = (AliMCParticle*) fMCEvent->GetTrack(lbDaughter);
        if (mcDaughter != 0x0) {
          for (int iGrandson = 0; iGrandson < mcDaughter->GetNDaughters(); iGrandson++) {
            relatives.push_back(TMath::Abs(mcDaughter->GetDaughterLabel(iGrandson)));
          }
        }
      }
    }
  }

  if (fMCEvent) {
    // search for matching TRD tracklets
    CountMatches(lbTrack, matchAvailableAll);
  }

  AliTRDpadPlane *pad = 0x0;

  double bZ = GetBz();

  // the vector det holds the numbers of the detectors which are searched for tracklets
  std::vector<int> det;
  std::vector<int>::iterator iDet;

  
  // set input track to first candidate(s)
  *fCandidates[0][0] = *t;
  int nCurrHypothesis = 0;
  int nCandidates = 1;

  // reset struct with hypothesis (necessary in case last prolongation failed, because then the struct is not reset)
  for (int iHypo=0; iHypo<fNhypothesis; iHypo++) {
    fHypothesis[iHypo].fChi2 = 1e4;
    fHypothesis[iHypo].fCandidateId = -1;
    fHypothesis[iHypo].fTrackletId = -1;
  } 

  // for browsing 2D array of candidates
  int currIdx;
  int nextIdx;
  // for tracklet update
  double p[2] = { 0. };
  double cov[3] = { 0. };
  // search window
  double roadY = 0;
  double roadZ = 0;

  // track properties
  AliHLTTRDTrack *trackNoUpdates = new AliHLTTRDTrack();
  *trackNoUpdates = *t;
  TVectorF trkNoUpX(kNLayers);
  TVectorF trkNoUpY(kNLayers);
  TVectorF trkNoUpZ(kNLayers);
  TVectorF trkNoUpYerr(kNLayers);
  TVectorF trkNoUpZerr(kNLayers);
  TVectorF trkNoUpPhi(kNLayers);
  TVectorF trkNoUpSec(kNLayers);
  TVectorF trkX(kNLayers);
  TVectorF trkY(kNLayers);
  TVectorF trkZ(kNLayers);
  TVectorF trkYerr(kNLayers);
  TVectorF trkZerr(kNLayers);
  TVectorF trkPhi(kNLayers);
  TVectorF trkSec(kNLayers);
  // tracklet properties (used for update)
  TVectorF trkltX(kNLayers);
  TVectorF trkltY(kNLayers);
  TVectorF trkltZ(kNLayers);
  TVectorF trkltDy(kNLayers);
  TVectorF trkltDet(kNLayers);
  // other debugging params
  TVectorF roadTrkY(kNLayers);
  TVectorF roadTrkZ(kNLayers);
  TVectorF chi2Up(kNLayers);
  TVectorF nTrkltsInWindow(kNLayers);
  TMatrixF chi2InWindow(kNLayers, 10);
  // tracklet properties (for matching tracklets)
  TVectorF realTrkltX(kNLayers);
  TVectorF realTrkltY(kNLayers);
  TVectorF realTrkltZ(kNLayers);
  TVectorF realTrkltDet(kNLayers);
  TVectorF realTrkltSec(kNLayers);
  TVectorF realTrkX(kNLayers);
  TVectorF realTrkY(kNLayers);
  TVectorF realTrkZ(kNLayers);
  TVectorF realTrkSec(kNLayers);
  TVectorF realChi2(kNLayers);

  for (int iLayer=0; iLayer<kNLayers; ++iLayer) {

    int nTrkltInWindow = 0;

    currIdx = iLayer % 2;
    nextIdx = (iLayer + 1) % 2;

    pad = fTRDgeometry->GetPadPlane(iLayer, 0);
    const float zMaxTRD = pad->GetRowPos(0);

    for (int iCandidate=0; iCandidate<nCandidates; iCandidate++) {

      if (!PropagateTrackToBxByBz(fCandidates[currIdx][iCandidate], fR[iLayer], mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
        Info("FollowProlongation", "track propagation failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        continue;
      }

      // rotate track in new sector in case of sector crossing
      if (!AdjustSector(fCandidates[currIdx][iCandidate])) {
        Info("FollowProlongation", "adjusting sector failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        continue;
      }

      if (!IsFindable(fCandidates[currIdx][iCandidate]->GetY(), fCandidates[currIdx][iCandidate]->GetZ(), fCandidates[currIdx][iCandidate]->GetAlpha(), iLayer )) {
        trackFindable(iLayer) = -1;
      }

      // define search window for tracklets
      roadY = 7. * TMath::Sqrt(fCandidates[currIdx][iCandidate]->GetSigmaY2() + TMath::Power(0.10, 2)) + 2; // add constant to the road for better efficiency
      roadZ = 7. * TMath::Sqrt(fCandidates[currIdx][iCandidate]->GetSigmaZ2() + TMath::Power(9./TMath::Sqrt(12), 2)); // take longest pad length

      if (TMath::Abs(fCandidates[currIdx][iCandidate]->GetZ()) - roadZ >= zMaxTRD ) {
        //Info("FollowProlongation", "track out of TRD acceptance with z=%f in layer %i (eta=%f)", fCandidates[currIdx][iCandidate]->GetZ(), iLayer, fCandidates[currIdx][iCandidate]->Eta());
        return false;
      }

      det.clear();
      if (!FindChambersInRoad(fCandidates[currIdx][iCandidate], roadY, roadZ, iLayer, det, zMaxTRD)) {
        Error("FollowProlongation", "finding chambers in road failed for track %i candidate %i in layer %i", iTrack, iCandidate, iLayer);
        continue;
      }

      // look for tracklets in chamber(s)
      bool wasTrackRotated = false;

      for (iDet = det.begin(); iDet != det.end(); ++iDet) {
        int detToSearch = *iDet;
        int stackToSearch = (detToSearch % 30) / kNLayers;
        int sectorToSearch = fTRDgeometry->GetSector(detToSearch);
        if (fTRDgeometry->IsHole(iLayer, stackToSearch, sectorToSearch) || detToSearch == 538) {
          // skip PHOS hole and non-existing chamber 17_4_4
          continue;
        }
        if (sectorToSearch != GetSector(fCandidates[currIdx][iCandidate]->GetAlpha()) && !wasTrackRotated) {
          float alphaToSearch = 2.0 * TMath::Pi() / (float) kNSectors * ((float) sectorToSearch + 0.5);
          if (!fCandidates[currIdx][iCandidate]->Rotate(alphaToSearch)) {
            Error("FollowProlongation", "track could not be rotated in tracklet coordinate system");
            break;
          }
          wasTrackRotated = true; // tracks need to be rotated max once per layer
        }
        if (sectorToSearch != GetSector(fCandidates[currIdx][iCandidate]->GetAlpha())) {
          Error("FollowProlongation", "track is in sector %i and sector %i is searched for tracklets",
                    GetSector(fCandidates[currIdx][iCandidate]->GetAlpha()), sectorToSearch);
          continue;
        }

        // first propagate track to x of tracklet
        for (int iTrklt=0; iTrklt<fTrackletIndexArray[detToSearch][1]; ++iTrklt) {
          int trkltIdx = fTrackletIndexArray[detToSearch][0] + iTrklt;
          Double_t trackYZ[2] = { 9999, 9999 }; // local y and z position of the track at the tracklet x
          Double_t xTracklet = fSpacePoints[trkltIdx].fX[0];
          if (!fCandidates[currIdx][iCandidate]->GetYZAt(xTracklet, bZ, trackYZ)) {
            Warning("FollowProlongation", "Track parameter at tracklet x cannot be retrieved");
            continue;
          }
          if ( (TMath::Abs(fSpacePoints[trkltIdx].fX[1] - trackYZ[0]) < roadY) &&
                  (TMath::Abs(fSpacePoints[trkltIdx].fX[2] - trackYZ[1]) < roadZ) )
          {
            //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
            p[0] = fSpacePoints[trkltIdx].fX[1];
            p[1] = fSpacePoints[trkltIdx].fX[2];
            cov[0] = fSpacePoints[trkltIdx].fCov[0];
            cov[1] = 0;
            cov[2] = fSpacePoints[trkltIdx].fCov[1];
            double chi2 = fCandidates[currIdx][iCandidate]->GetPredictedChi2(p, cov);
            if (chi2 < fMaxChi2) {
              if (nCurrHypothesis >= (fNhypothesis-1)) {
                std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
                if ( (chi2 + fCandidates[currIdx][iCandidate]->GetChi2()) < fHypothesis[nCurrHypothesis].fChi2) {
                  fHypothesis[nCurrHypothesis].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + chi2;
                  fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
                  fHypothesis[nCurrHypothesis].fTrackletId = trkltIdx;
                }
              }
              else {
                fHypothesis[nCurrHypothesis].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + chi2;
                fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
                fHypothesis[nCurrHypothesis].fTrackletId = trkltIdx;
                nCurrHypothesis++;
              }
              if (nTrkltInWindow < 10) {
		Bool_t fnd = kFALSE;
		for (int il=0;il<3;il++) {
		  int lbl = fSpacePoints[trkltIdx].fLabel[il];
		  if (lbl<0) break; // no more valid labels
		  if (lbl==lbTrack ||
		      std::find(relatives.begin(), relatives.end(), lbl) == relatives.end() ) {
		    fnd = kTRUE;
		    break;
		  }
		}
		// RS: why do we register chi2 only if it is not correct and not "related"? 
		if (!fnd) chi2InWindow[iLayer][nTrkltInWindow++] = chi2;
              }
            }
            nTrkltsInWindow(iLayer) += 1;
          }
        }
      } // end adding hypothesis
      // add no update to hypothesis list
      if (nCurrHypothesis >= (fNhypothesis-1)) {
        std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
        if ( (fCandidates[currIdx][iCandidate]->GetChi2() + fChi2Penalty) < fHypothesis[nCurrHypothesis].fChi2) {
          fHypothesis[nCurrHypothesis].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + fChi2Penalty;
          fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
          fHypothesis[nCurrHypothesis].fTrackletId = -1;
        }
      }
      else {
        fHypothesis[nCurrHypothesis].fChi2 = fCandidates[currIdx][iCandidate]->GetChi2() + fChi2Penalty;
        fHypothesis[nCurrHypothesis].fCandidateId = iCandidate;
        fHypothesis[nCurrHypothesis].fTrackletId = -1;
        nCurrHypothesis++;
      }
    } // end candidate loop

    // in case matching tracklet exists in this layer -> store position information for debugging
    if (matchAvailableAll[iLayer].size() > 0) {
      realTrkltX(iLayer) = fSpacePoints[matchAvailableAll[iLayer].at(0)].fX[0];
      realTrkltY(iLayer) = fSpacePoints[matchAvailableAll[iLayer].at(0)].fX[1];
      realTrkltZ(iLayer) = fSpacePoints[matchAvailableAll[iLayer].at(0)].fX[2];
      realTrkltDet(iLayer) =  fTracklets[matchAvailableAll[iLayer].at(0)].GetDetector();
      realTrkltSec(iLayer) = fTRDgeometry->GetSector(realTrkltDet[iLayer]);
      double trackYZtmp[2] = { 9999, 9999 }; // local y and z position of the track at the tracklet x
      fCandidates[currIdx][0]->GetYZAt(realTrkltX(iLayer), bZ, trackYZtmp);
      realTrkX(iLayer) = realTrkltX(iLayer);
      realTrkY(iLayer) = trackYZtmp[0];
      realTrkZ(iLayer) = trackYZtmp[1];
      realTrkSec(iLayer) = GetSector(fCandidates[currIdx][0]->GetAlpha());
      double realP[2] = { 0 };
      double realCov[3] = { 0 };
      realP[0] = fSpacePoints[matchAvailableAll[iLayer].at(0)].fX[1];
      realP[1] = fSpacePoints[matchAvailableAll[iLayer].at(0)].fX[2];
      realCov[0] = fSpacePoints[matchAvailableAll[iLayer].at(0)].fCov[0];
      realCov[1] = 0;
      realCov[2] = fSpacePoints[matchAvailableAll[iLayer].at(0)].fCov[1];
      realChi2(iLayer) = fCandidates[currIdx][0]->GetPredictedChi2(realP, realCov);
    }
    //

    std::sort(fHypothesis, fHypothesis+nCurrHypothesis, Hypothesis_Sort);
    chi2Up(iLayer) = fHypothesis[0].fChi2 - t->GetChi2();
    for (int iUpdate = 0; iUpdate < kNcandidates; iUpdate++) {
      if (fHypothesis[iUpdate].fCandidateId == -1) {
        // no more candidates
        if (iUpdate == 0) {
          Warning("FollowProlongation", "no valid candidates for track %i in layer %i", iTrack, iLayer);
          return false;
        }
        break;
      }
      nCandidates = iUpdate + 1;
      *fCandidates[nextIdx][iUpdate] = *fCandidates[currIdx][fHypothesis[iUpdate].fCandidateId];
      if (fHypothesis[iUpdate].fTrackletId == -1) {
        // no update for this candidate
        fCandidates[nextIdx][iUpdate]->SetChi2(fHypothesis[iUpdate].fChi2);
        if (iUpdate == 0) {
          *t = *fCandidates[nextIdx][iUpdate];
        }
        continue;
      }
      // best matching tracklet found
      if (fTRDgeometry->GetSector(fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector()) != GetSector(fCandidates[nextIdx][iUpdate]->GetAlpha())) {
        fCandidates[nextIdx][iUpdate]->Rotate( 2. * TMath::Pi() / (float) kNSectors * ((float) fTRDgeometry->GetSector(fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector()) + 0.5));
      }
      p[0] = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[1];
      p[1] = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[2];
      cov[0] = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov[0];
      cov[1] = 0;
      cov[2] = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fCov[1];

      if (!PropagateTrackToBxByBz(fCandidates[nextIdx][iUpdate], fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0], mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)){
        Error("FollowProlongation", "Final track propagation for update failed");
        continue;
      }

      PropagateTrackToBxByBz(trackNoUpdates, fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0], mass, 2.0, kFALSE, 0.8);
      trackNoUpdates->Rotate( 2. * TMath::Pi() / (float) kNSectors * ((float) fTRDgeometry->GetSector(fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector()) + 0.5));
      trkNoUpX(iLayer) = trackNoUpdates->GetX();
      trkNoUpY(iLayer) = trackNoUpdates->GetY();
      trkNoUpZ(iLayer) = trackNoUpdates->GetZ();
      trkNoUpYerr(iLayer) = trackNoUpdates->GetSigmaY2();
      trkNoUpZerr(iLayer) = trackNoUpdates->GetSigmaZ2();
      trkNoUpPhi(iLayer) = TMath::ASin(trackNoUpdates->GetSnp());
      trkNoUpSec(iLayer) = GetSector(trackNoUpdates->GetAlpha());
      trkX(iLayer) = fCandidates[nextIdx][iUpdate]->GetX();
      trkY(iLayer) = fCandidates[nextIdx][iUpdate]->GetY();
      trkZ(iLayer) = fCandidates[nextIdx][iUpdate]->GetZ();
      trkYerr(iLayer) = fCandidates[nextIdx][iUpdate]->GetSigmaY2();
      trkZerr(iLayer) = fCandidates[nextIdx][iUpdate]->GetSigmaZ2();
      trkPhi(iLayer) = TMath::ASin(fCandidates[nextIdx][iUpdate]->GetSnp());
      trkSec(iLayer) = GetSector(fCandidates[nextIdx][iUpdate]->GetAlpha());
      trkltX(iLayer) = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0];
      trkltY(iLayer) = p[0];
      trkltZ(iLayer) = p[1];
      trkltDy(iLayer) = fSpacePoints[fHypothesis[iUpdate].fTrackletId].fDy;
      trkltDet(iLayer) = fTracklets[fHypothesis[iUpdate].fTrackletId].GetDetector();
      roadTrkY(iLayer) = roadY;
      roadTrkZ(iLayer) = roadZ;

      if (!fCandidates[nextIdx][iUpdate]->Update(p, cov)) {
        Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);
        continue;
      }
      fCandidates[nextIdx][iUpdate]->AddTracklet(iLayer, fHypothesis[iUpdate].fTrackletId);
      int nTrkltsCurr = fCandidates[nextIdx][iUpdate]->GetNtracklets();
      if (nTrkltsCurr >= 6) {
        printf("nTracklets (%i) larger than possible. iUpdate(%i), iLayer(%i), iTrack(%i)\n", nTrkltsCurr, iUpdate, iLayer, iTrack);
      }
      fCandidates[nextIdx][iUpdate]->SetNtracklets(nTrkltsCurr+1);
      fCandidates[nextIdx][iUpdate]->SetChi2(fHypothesis[iUpdate].fChi2);
      if (iUpdate == 0) {
        *t = *fCandidates[nextIdx][iUpdate];
      }
      if (TMath::Abs(fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0] - fR[iLayer]) > 5) {
        Error("FollowProlongation", "tracklet from wrong layer added to track. fR[iLayer]=%f, xTracklet=%f, trkltIdx=%i",
                  fR[iLayer], fSpacePoints[fHypothesis[iUpdate].fTrackletId].fX[0], fHypothesis[iUpdate].fTrackletId);
      }
    } // end update loop
    // reset struct with hypothesis
    for (int iHypothesis=0; iHypothesis<=nCurrHypothesis; iHypothesis++) {
      fHypothesis[iHypothesis].fChi2 = 1e4;
      fHypothesis[iHypothesis].fCandidateId = -1;
      fHypothesis[iHypothesis].fTrackletId = -1;
    }
    nCurrHypothesis = 0;
  } // end layer loop
  for (int iLy = 0; iLy < kNLayers; iLy++) {
    tracklets(iLy) = 0;
    if (fMCEvent && t->GetTracklet(iLy) != -1) {
      tracklets(iLy) = 1;
      int lbTracklet;
      for (int il=0;il<3;il++) {
	if ( (lbTracklet=fSpacePoints[t->GetTracklet(iLy)].fLabel[il])<0 ) break; // no more valid labels
 	if (lbTracklet == lbTrack) {
	  tracklets(iLy) = 64; // RS: exact match, do we need to flag non-first matching label as 64+il ? 
	  break;
	}
      }
      if (tracklets(iLy)<2) { // no exact match, check in related labels
	for (int il=0;il<3;il++) {
	  if ( (lbTracklet=fSpacePoints[t->GetTracklet(iLy)].fLabel[il])<0 ) break; // no more valid labels
	  AliMCParticle *mcPart = (AliMCParticle*) fMCEvent->GetTrack(lbTracklet);
	  while (mcPart) {
	    int motherPart = mcPart->GetMother();
	    if (motherPart == lbTrack) {
	      tracklets(iLy) = 8; // RS: related match, do we need to flag non-first matching label as 8+il ? 
	      break;
	    }
	    mcPart = motherPart >= 0 ? (AliMCParticle*) fMCEvent->GetTrack(motherPart) : 0; 
	  }
	}
      }
    }
  }
  if (fDebugOutput) {
    TVectorF matchInLayer(kNLayers);
    for (int lyIdx=0; lyIdx<kNLayers; lyIdx++) {
      if (matchAvailableAll[lyIdx].size() > 0) {
        matchInLayer(lyIdx) = 1;
      }
    }
    double chi2Track = t->GetChi2();
    int nTracklets = t->GetNtracklets();
    int nTrackletsOffline = t->GetNtrackletsOffline();
    AliExternalTrackParam parameterFinal(*t);
    AliExternalTrackParam parameterNoUpdates(*trackNoUpdates);
    (*fStreamer) << "tracksFinal" <<
      "ev=" << fNEvents <<
      "iTrk=" << iTrack <<
      "nTracklets=" << nTracklets <<
      "nTrackletsOffline=" << nTrackletsOffline <<
      "chi2Trk=" << chi2Track <<
      "update.=" << &tracklets <<
      "matchAvailable.=" << &matchInLayer <<
      "trkltXReal.=" << &realTrkltX <<
      "trkltYReal.=" << &realTrkltY <<
      "trkltZReal.=" << &realTrkltZ <<
      "trkltSecReal.=" << &realTrkltSec <<
      "trkltDetReal.=" << &realTrkltDet <<
      "trkXReal.=" << &realTrkX <<
      "trkYReal.=" << &realTrkY <<
      "trkZReal.=" << &realTrkZ <<
      "trkSecReal.=" << &realTrkSec <<
      "chi2.=" << &realChi2 <<
      "chi2Up.=" << &chi2Up <<
      "nTrkltsInWindow.=" << &nTrkltsInWindow <<
      "trkNoUpX.=" << &trkNoUpX <<
      "trkNoUpY.=" << &trkNoUpY <<
      "trkNoUpZ.=" << &trkNoUpZ <<
      "trkNoUpYerr.=" << &trkNoUpYerr <<
      "trkNoUpZerr.=" << &trkNoUpZerr <<
      "trkNoUpPhi.=" << &trkNoUpPhi <<
      "trkNoUpSec.=" << &trkNoUpSec <<
      "trkX.=" << &trkX <<
      "trkY.=" << &trkY <<
      "trkZ.=" << &trkZ <<
      "trkYerr.=" << &trkYerr <<
      "trkZerr.=" << &trkZerr <<
      "trkPhi.=" << &trkPhi <<
      "trkSec.=" << &trkSec <<
      "trkltX.=" << &trkltX <<
      "trkltY.=" << &trkltY <<
      "trkltZ.=" << &trkltZ <<
      "trkltDy.=" << &trkltDy <<
      "trkltDet.=" << &trkltDet <<
      "roadY.=" << &roadTrkY <<
      "roadZ.=" << &roadTrkZ <<
      "findable.=" << &trackFindable <<
      "track.=" << &parameterFinal <<
      "trackNoUp.=" << &parameterNoUpdates <<
      "chi2InWindow.=" << &chi2InWindow <<
      "\n";
  }

  return true;
}

int AliHLTTRDTracker::GetDetectorNumber(const double zPos, double alpha, int layer)
{
  //--------------------------------------------------------------------
  // if track position is within chamber return the chamber number
  // otherwise return -1
  //--------------------------------------------------------------------
  int stack = fTRDgeometry->GetStack(zPos, layer);
  if (stack < 0) {
    return -1;
  }
  double alphaTmp = (alpha > 0) ? alpha : alpha + 2. * TMath::Pi();
  int sector = 18. * alphaTmp / (2. * TMath::Pi());

  return fTRDgeometry->GetDetector(layer, stack, sector);
}

bool AliHLTTRDTracker::AdjustSector(AliHLTTRDTrack *t)
{
  //--------------------------------------------------------------------
  // rotate track in new sector if necessary
  //--------------------------------------------------------------------
  double alpha     = fTRDgeometry->GetAlpha();
  double y         = t->GetY();
  double yMax      = t->GetX() * TMath::Tan(0.5 * alpha);
  double alphaCurr = t->GetAlpha();

  if (TMath::Abs(y) > 2. * yMax) {
    Info("AdjustSector", "Track %i with pT = %f crossing two sector boundaries at x = %f", t->GetTPCtrackId(), t->Pt(), t->GetX());
    return false;
  }

  if (y > yMax) {

    if (!t->Rotate(alphaCurr+alpha)) {
      return false;
    }
  }
  else if (y < -yMax) {
    if (!t->Rotate(alphaCurr-alpha)) {
      return false;
    }
  }
  return true;
}

int AliHLTTRDTracker::GetSector(double alpha)
{
  //--------------------------------------------------------------------
  // TRD sector number for reference system alpha
  //--------------------------------------------------------------------
  if (alpha < 0) {
    alpha += 2. * TMath::Pi();
  }
  return (int) (alpha * kNSectors / (2. * TMath::Pi()));
}

void AliHLTTRDTracker::CountMatches(int trkLbl, std::vector<int> *matches)
{
  // search in all TRD chambers for matching tracklets
  for (int k = 0; k < kNChambers; k++) {
    int layer = fTRDgeometry->GetLayer(k);
    for (int iTrklt = 0; iTrklt < fTrackletIndexArray[k][1]; iTrklt++) {
      int trkltIdx = fTrackletIndexArray[k][0] + iTrklt;
      for (int il=0;il<3;il++) {
	int lb = fSpacePoints[trkltIdx].fLabel[il];
	if (lb<0) break; // no more valid labels
	if (lb == trkLbl) {
	  matches[layer].push_back(trkltIdx);
	  break;
	}
      }
    }
  }
}

bool AliHLTTRDTracker::FindChambersInRoad(AliHLTTRDTrack *t, float roadY, float roadZ, int iLayer, std::vector<int> &det, float zMax)
{
  //--------------------------------------------------------------------
  // determine initial chamber where the track ends up
  // add more chambers of the same sector if track is close to edge of the chamber
  //--------------------------------------------------------------------

  const float yMax    = TMath::Abs(fTRDgeometry->GetCol0(iLayer));

  // where the track initially ends up
  int initialStack  = -1;
  int initialSector = -1;
  int detector      = -1;

  detector = GetDetectorNumber(t->GetZ(), t->GetAlpha(), iLayer);
  if (detector != -1) {
    det.push_back(detector);
    initialStack = fTRDgeometry->GetStack(detector);
    initialSector = fTRDgeometry->GetSector(detector);
    AliTRDpadPlane *padTmp = fTRDgeometry->GetPadPlane(iLayer, initialStack);
    int lastPadRow = -1;
    float zCenter = 0.;
    if (initialStack == 2) {
      lastPadRow = 11;
      zCenter = padTmp->GetRowPos(6);
    }
    else {
      lastPadRow = 15;
      zCenter = padTmp->GetRowPos(8);
    }
    if ( ( t->GetZ() + roadZ ) > padTmp->GetRowPos(0) || ( t->GetZ() - roadZ ) < padTmp->GetRowPos(lastPadRow) ) {
      int addStack = t->GetZ() > zCenter ? initialStack - 1 : initialStack + 1;
      if (addStack < kNStacks && addStack > -1) {
        det.push_back(fTRDgeometry->GetDetector(iLayer, addStack, initialSector));
      }
    }
  }
  else {
    if (TMath::Abs(t->GetZ()) > zMax) {
      t->GetZ() > 0 ? // shift track in z so it is in the TRD acceptance
        detector = GetDetectorNumber( 280 /*t->GetZ()- roadZ*/, t->GetAlpha(), iLayer) :
        detector = GetDetectorNumber( -280 /*t->GetZ()+ roadZ*/, t->GetAlpha(), iLayer); //FIXME otherwise for a large road the search might start in stack 1 or 3 instead of 0 or 4
      if (detector != -1) {
        det.push_back(detector);
        initialStack = fTRDgeometry->GetStack(detector);
        initialSector = fTRDgeometry->GetSector(detector);
      }
      else {
        Error("FindChambersInRoad", "outer detector cannot be found although track was shifted in z");
        return false;
      }
    }
    else {
      // track in between two stacks, add both surrounding chambers
      detector = GetDetectorNumber(t->GetZ()+4.0, t->GetAlpha(), iLayer);
      if (detector != -1) {
        det.push_back(detector);
        initialStack = fTRDgeometry->GetStack(detector);
        initialSector = fTRDgeometry->GetSector(detector);
      }
      else {
        Error("FindChambersInRoad", "detector cannot be found although track was shifted in positive z");
        return false;
      }
      detector = GetDetectorNumber(t->GetZ()-4.0, t->GetAlpha(), iLayer);
      if (detector != -1) {
        det.push_back(detector);
      }
      else {
        Error("FindChambersInRoad", "detector cannot be found although track was shifted in negative z");
        return false;
      }
    }
  }
  // add chamber(s) from neighbouring sector in case the track is close to the boundary
  if ( ( TMath::Abs(t->GetY()) + roadY ) > yMax ) {
    const int nStacksToSearch = det.size();
    for (int idx = 0; idx < nStacksToSearch; ++idx) {
      int currStack = fTRDgeometry->GetStack(det.at(idx));
      if (t->GetY() > 0) {
        int newSector = initialSector + 1;
        if (newSector == kNSectors) {
          newSector = 0;
        }
        det.push_back(fTRDgeometry->GetDetector(iLayer, currStack, newSector));
      }
      else {
        int newSector = initialSector - 1;
        if (newSector == -1) {
          newSector = kNSectors - 1;
        }
        det.push_back(fTRDgeometry->GetDetector(iLayer, currStack, newSector));
      }
    }
  }
  return true;
}

bool AliHLTTRDTracker::IsFindable(float y, float z, float alpha, int layer)
{
  //--------------------------------------------------------------------
  // Is track position inside active area of the TRD
  //--------------------------------------------------------------------

  int detector = GetDetectorNumber(z, alpha, layer);

  // reject tracks between stacks
  if (detector < 0) {
    return false;
  }

  // reject tracks inside masked chambers
  if (std::find(fMaskedChambers.begin(), fMaskedChambers.end(), detector) != fMaskedChambers.end()) {
    return false;
  }

  AliTRDpadPlane *pp = fTRDgeometry->GetPadPlane(layer, fTRDgeometry->GetStack(detector));
  float yMax = TMath::Abs(pp->GetColPos(0));

  // reject tracks closer than 5 cm to pad plane boundary
  if (yMax - TMath::Abs(y) < 5.) {
    return false;
  }

  return true;
}
