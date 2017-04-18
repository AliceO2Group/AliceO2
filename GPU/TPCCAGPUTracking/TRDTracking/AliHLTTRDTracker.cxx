#include <vector>
#include "AliHLTTRDTracker.h"
#include "AliGeomManager.h"
#include "AliTRDgeometry.h"
#include "TDatabasePDG.h"
#include "TTreeStream.h"
#include "TGeoMatrix.h"

ClassImp(AliHLTTRDTracker)

// default values taken from AliTRDtrackerV1.cxx
const Double_t AliHLTTRDTracker::fgkX0[kNLayers]    = { 300.2, 312.8, 325.4, 338.0, 350.6, 363.2 };

AliHLTTRDTracker::AliHLTTRDTracker() :
  fIsInitialized(false),
  fTracks(0x0),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fSpacePoints(0x0),
  fTRDgeometry(0x0),
  fDebugOutput(false),
  fMinPt(1.0),
  fMaxChi2(35.0),
  fStreamer(0x0)
{
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = fgkX0[iLy];
  }
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fTrackletIndexArray[iDet][0] = -1;
    fTrackletIndexArray[iDet][1] = 0;
  }
  //Default constructor
}

AliHLTTRDTracker::AliHLTTRDTracker(const AliHLTTRDTracker &tracker) :
  fIsInitialized(false),
  fTracks(0x0),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fSpacePoints(0x0),
  fTRDgeometry(0x0),
  fDebugOutput(false),
  fMinPt(1.0),
  fMaxChi2(35.0),
  fStreamer(0x0)
{
  //Copy constructor
  //Dummy!
  for (int iLy=0; iLy<kNLayers; iLy++) {
    fR[iLy] = fgkX0[iLy];
  }
  for (int iDet=0; iDet<kNChambers; ++iDet) {
    fTrackletIndexArray[iDet][0] = tracker.fTrackletIndexArray[iDet][0];
    fTrackletIndexArray[iDet][1] = tracker.fTrackletIndexArray[iDet][1];
  }
}

AliHLTTRDTracker & AliHLTTRDTracker::operator=(const AliHLTTRDTracker &tracker){
  //Assignment operator
  this->~AliHLTTRDTracker();
  new(this) AliHLTTRDTracker(tracker);
  return *this;
}

AliHLTTRDTracker::~AliHLTTRDTracker()
{
  //Destructor
  if (fIsInitialized) {
    delete[] fTracklets;
    delete[] fTracks;
    delete[] fSpacePoints;
    delete fTRDgeometry;
    if (fDebugOutput) {
      delete fStreamer;
    }
  }
}


void AliHLTTRDTracker::Init()
{
  if(!AliGeomManager::GetGeometry()){
    Error("Init", "Could not get geometry.");
  }

  fTracklets = new AliHLTTRDTrackletWord[fNtrackletsMax];
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
  fNTracklets = 0;
  for (int i=0; i<fNtrackletsMax; ++i) {
    fTracklets[i] = 0x0;
    fSpacePoints[i].fX[0]     = 0.;
    fSpacePoints[i].fX[1]     = 0.;
    fSpacePoints[i].fX[2]     = 0.;
    fSpacePoints[i].fCov[0]   = 0.;
    fSpacePoints[i].fCov[1]   = 0.;
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
  if (fNTracklets >= fNtrackletsMax ) {
    Error("LoadTracklet", "running out of memory for tracklets, skipping tracklet(s). This should actually never happen.");
    return;
  }
  fTracklets[fNTracklets++] = tracklet;
  fTrackletIndexArray[tracklet.GetDetector()][1]++;
}

void AliHLTTRDTracker::DoTracking( AliExternalTrackParam *tracksTPC, int *tracksTPCLab, int nTPCTracks )
{
  //--------------------------------------------------------------------
  // This functions reconstructs TRD tracks
  // using TPC tracks as seeds
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

  CalculateSpacePoints();

  delete[] fTracks;
  fNTracks = 0;
  fTracks = new AliHLTTRDTrack[nTPCTracks];

  double piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass(); // pion mass as best guess for all particles

  for (int i=0; i<nTPCTracks; ++i) {
    AliHLTTRDTrack tMI(tracksTPC[i]);
    AliHLTTRDTrack *t = &tMI;
    t->SetTPCtrackId(i);
    t->SetLabel(tracksTPCLab[i]);
    t->SetMass(piMass);

    int result = FollowProlongation(t, piMass);
    t->SetNtracklets(result);
    fTracks[fNTracks++] = *t;

    double pT = t->Pt();
    double alpha = t->GetAlpha();

    if (fDebugOutput) {
      (*fStreamer) << "tracksFinal" <<
        "ev=" << fNEvents <<
        "iTrk=" << i <<
        "pT=" << pT <<
        "alpha=" << alpha <<
        "nTrackletsAttached=" << result <<
        "track.=" << t <<
        "\n";
    }
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


void AliHLTTRDTracker::CalculateSpacePoints()
{
  //--------------------------------------------------------------------
  // This functions calculates the TRD space points
  // in sector tracking coordinates for all tracklets
  //--------------------------------------------------------------------

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
	    continue;
    }
    AliTRDpadPlane *padPlane = fTRDgeometry->GetPadPlane(layer, stack);

    for (int iTrklt=0; iTrklt<nTracklets; ++iTrklt) {
      int trkltIdx = fTrackletIndexArray[iDet][0] + iTrklt;
      double xTrklt[3] = { 0. };
      xTrklt[0] = AliTRDgeometry::AnodePos();
      xTrklt[1] = fTracklets[trkltIdx].GetY();
      if (stack == 2) {
        xTrklt[2] = padPlane->GetRowPos(fTracklets[trkltIdx].GetZbin()) - (padPlane->GetRowSize(fTracklets[trkltIdx].GetZbin()))/2. - padPlane->GetRowPos(6);
      }
      else {
        xTrklt[2] = padPlane->GetRowPos(fTracklets[trkltIdx].GetZbin()) - (padPlane->GetRowSize(fTracklets[trkltIdx].GetZbin()))/2. - padPlane->GetRowPos(8);
      }
      matrix->LocalToMaster(xTrklt, fSpacePoints[trkltIdx].fX);
      fSpacePoints[trkltIdx].fId = fTracklets[trkltIdx].GetId();
      fSpacePoints[trkltIdx].fCov[0] = 0.03;
      fSpacePoints[trkltIdx].fCov[1] = padPlane->GetRowPos(fTracklets[trkltIdx].GetZbin()) / TMath::Sqrt(12);

      AliGeomManager::ELayerID iLayer = AliGeomManager::ELayerID(AliGeomManager::kTRD1+fTRDgeometry->GetLayer(iDet));
      int modId   = fTRDgeometry->GetSector(iDet) * AliTRDgeometry::kNstack + fTRDgeometry->GetStack(iDet); // global TRD stack number
      unsigned short volId = AliGeomManager::LayerToVolUID(iLayer, modId);
      fSpacePoints[trkltIdx].fVolumeId = volId;
    }
  }
}


int AliHLTTRDTracker::FollowProlongation(AliHLTTRDTrack *t, double mass)
{
  //--------------------------------------------------------------------
  // This function propagates found tracks with pT > fMinPt
  // through the TRD and picks up the closest tracklet in each
  // layer on the way.
  // the tracks are currently not updated with the assigned tracklets
  // returns variable result = number of assigned tracklets
  //--------------------------------------------------------------------

  int result = 0;
  int iTrack = t->GetTPCtrackId(); // for debugging individual tracks

  // only propagate tracks within TRD acceptance
  if (TMath::Abs(t->Eta()) > 0.9) {
    return result;
  }

  // introduce momentum cut on tracks
  // particles with pT < 0.9 GeV highly unlikely to have matching online TRD tracklets
  if (t->Pt() < fMinPt) {
    return result;
  }

  if (fDebugOutput) {
    double xDebug = t->GetX();
    double eta = t->Eta();
    double pt = t->Pt();
    AliExternalTrackParam debugParam(*t);
    (*fStreamer) << "initialTracks" <<
      "x=" << xDebug <<
      "pt=" << pt <<
      "eta=" << eta <<
      "track.=" << &debugParam <<
      "\n";
  }

  // define some constants
  const int nCols   = fTRDgeometry->Colmax();

  AliTRDpadPlane *pad = 0x0;

  // the vector det holds the numbers of the detectors which are searched for tracklets
  std::vector<int> det;
  std::vector<int>::iterator iDet;

  for (int iLayer=0; iLayer<kNLayers; ++iLayer) {

    det.clear();

    // where the trackl initially ends up
    int initialStack  = -1;
    int initialSector = -1;
    int detector      = -1;

    pad = fTRDgeometry->GetPadPlane(iLayer, 0);

    const float zMaxTRD = pad->GetRowPos(0);
    const float yMax    = pad->GetColPos(nCols-1) + pad->GetColSize(nCols-1);

    if (!PropagateTrackToBxByBz(t, fR[iLayer], mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
      return result;
    }

    if (abs(t->GetZ()) >= (zMaxTRD + 10.)) {
      Warning("FollowProlongation", "track out of TRD acceptance with z=%f in layer %i (eta=%f)", t->GetZ(), iLayer, t->Eta());
      return result;
    }

    // rotate track in new sector in case of sector crossing
    if (!AdjustSector(t)) {
      return result;
    }

    // debug purposes only -> store track information
    if (fDebugOutput) {
      double sigmaY = TMath::Sqrt(t->GetSigmaY2());
      double sigmaZ = TMath::Sqrt(t->GetSigmaZ2());
      AliExternalTrackParam param(*t);
      (*fStreamer) << "trackInfoLayerwise" <<
        "iEv=" << fNEvents <<
        "iTrk=" << iTrack <<
        "layer=" << iLayer <<
        "sigmaY=" << sigmaY <<
        "sigmaZ=" << sigmaZ <<
        "nUpdates=" << result <<
        "track.=" << &param <<
        "\n";
    }

    // determine initial chamber where the track ends up
    // add more chambers of the same sector if track is close to edge of the chamber
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
      if ( t->GetZ() > (padTmp->GetRowPos(0) - 10.) || t->GetZ() < (padTmp->GetRowPos(lastPadRow) + 10) ) {
        int extStack = t->GetZ() > zCenter ? initialStack - 1 : initialStack + 1;
        if (extStack < kNStacks && extStack > -1) {
          det.push_back(fTRDgeometry->GetDetector(iLayer, extStack, initialSector));
        }
      }
    }
    else {
      if (TMath::Abs(t->GetZ()) > zMaxTRD) {
        t->GetZ() > 0 ? // shift track in z so it is in the TRD acceptance
          detector = GetDetectorNumber(t->GetZ()-10., t->GetAlpha(), iLayer) :
          detector = GetDetectorNumber(t->GetZ()+10., t->GetAlpha(), iLayer);
        if (detector != -1) {
          det.push_back(detector);
          initialStack = fTRDgeometry->GetStack(detector);
          initialSector = fTRDgeometry->GetSector(detector);
        }
        else {
          Error("FollowProlongation", "outer detector cannot be found although track %i was shifted in z", iTrack);
          return result;
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
          Error("FollowProlongation", "detector cannot be found although track %i was shifted in positive z", iTrack);
          return result;
        }
        detector = GetDetectorNumber(t->GetZ()-4.0, t->GetAlpha(), iLayer);
        if (detector != -1) {
          det.push_back(detector);
        }
        else {
          Error("FollowProlongation", "detector cannot be found although track %i was shifted in negative z", iTrack);
          return result;
        }
      }
    }

    // add chamber(s) from neighbouring sector in case the track is close to the boundary
    if ( TMath::Abs(t->GetY()) > (yMax - 10.) ) {
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

    if (fDebugOutput) {
      int nDetToSearch = det.size();
      (*fStreamer) << "chambersToSearch" <<
        "nChambers=" << nDetToSearch <<
        "layer=" << iLayer <<
        "iEv=" << fNEvents <<
        "\n";
    }

    // define search window for tracklets
    double deltaY = 7. * TMath::Sqrt(t->GetSigmaY2() + TMath::Power(0.03, 2)) + 2; // add constant to the road for better efficiency
    double deltaZ = 7. * TMath::Sqrt(t->GetSigmaZ2() + TMath::Power(9./TMath::Sqrt(12), 2));

    if (fDebugOutput) {
      (*fStreamer) << "searchWindow" <<
        "layer=" << iLayer <<
        "dY=" << deltaY <<
        "dZ=" << deltaZ <<
        "\n";
    }

    // look for tracklets in chamber(s)
    double bestGuessChi2 = fMaxChi2; // TODO define meaningful chi2 cut
    int bestGuessIdx = -1;
    int bestGuessDet = -1;
    double p[2] = { 0. };
    double cov[3] = { 0. };
    bool wasTrackRotated = false;
    for (iDet = det.begin(); iDet != det.end(); ++iDet) {
      int detToSearch = *iDet;
      int stackToSearch = (detToSearch % 30) / kNLayers;
      int stackToSearchGlobal = detToSearch / kNLayers; // global stack number (0..89)
      int sectorToSearch = fTRDgeometry->GetSector(detToSearch);
      if (fTRDgeometry->IsHole(iLayer, stackToSearch, sectorToSearch) || detToSearch == 538) {
        // skip PHOS hole and non-existing chamber 17_4_4
        continue;
      }
      if (sectorToSearch != initialSector && !wasTrackRotated) {
        float alphaToSearch = 2.0 * TMath::Pi() / (float) kNSectors * ((float) sectorToSearch + 0.5);
        if (!t->Rotate(alphaToSearch)) {
          return result;
        }
        wasTrackRotated = true; // tracks need to be rotated max once per layer
      }

      // TODO
      // - get parameter at x of chamber
      // - for each tracklet: get parameter at x of tracklet and compare y and z at this position
      double xOfChamberLoc[3] = {AliTRDgeometry::AnodePos(), 0., 0.};
      double xOfChamberGlb[3];
      TGeoHMatrix *matrix = fTRDgeometry->GetClusterMatrix(detToSearch);
      if (!matrix) {
        Error("FollowProlongation", "invalid TRD cluster matrix, skipping detector %i", detToSearch);
        continue;
      }
      matrix->LocalToMaster(xOfChamberLoc, xOfChamberGlb);
      if (TMath::Abs(xOfChamberGlb[0] - t->GetX()) > 1.) {
        if (!PropagateTrackToBxByBz(t, xOfChamberGlb[0], mass, 1.0, kFALSE, 0.8)) {
          Error("FollowProlongation", "track propagation failed while fine tuning track parameter for chamber");
          return result;
        }
      }

      for (int iTrklt=0; iTrklt<fTrackletIndexArray[detToSearch][1]; ++iTrklt) {
        int trkltIdx = fTrackletIndexArray[detToSearch][0] + iTrklt;
        Double_t trackYZ[2] = { 9999, 9999 }; // local y and z position of the track at the tracklet x
        Double_t xTracklet = fSpacePoints[trkltIdx].fX[0];
        Double_t bZ = GetBz(&xTracklet);
        if (!t->GetYZAt(xTracklet, bZ, trackYZ)) {
          Warning("FollowProlongation", "Track parameter at tracklet x cannot be retrieved");
          continue;
        }
        if ( (TMath::Abs(fSpacePoints[trkltIdx].fX[1] - trackYZ[0]) < deltaY) && (TMath::Abs(fSpacePoints[trkltIdx].fX[2] - trackYZ[1]) < deltaZ) )
        {
          //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
          p[0] = fSpacePoints[trkltIdx].fX[1];
          p[1] = fSpacePoints[trkltIdx].fX[2];
          if (TMath::Abs(p[0]) > 1000) {
            Error("FollowProlongation", "impossible y-value of tracklet: y=%.12f", p[0]);
            continue;
          }
          cov[0] = TMath::Power(0.03, 2);
          cov[1] = 0;
          cov[2] = TMath::Power(9.,2) / 12.;
          double chi2 = t->GetPredictedChi2(p, cov);
          if (chi2 < bestGuessChi2) {
            bestGuessChi2 = chi2;
            bestGuessIdx = trkltIdx;
            bestGuessDet = detToSearch;
          }
          float dX = xTracklet - fSpacePoints[trkltIdx].fX[0]; // zero by definition
          float dY = trackYZ[0] - fSpacePoints[trkltIdx].fX[1];
          float dZ = trackYZ[1] - fSpacePoints[trkltIdx].fX[2];
          if (fDebugOutput) {
            (*fStreamer) << "residuals" <<
              "iEv=" << fNEvents <<
              "iTrk=" << iTrack <<
              "layer=" << iLayer <<
              "iTrklt=" << iTrklt <<
              "stack=" << stackToSearchGlobal <<
              "det=" << detToSearch <<
              "dX=" << dX <<
              "dY=" << dY <<
              "dZ=" << dZ <<
              "\n";
          }
          if (TMath::Abs(dZ) > 160) {
            Error("FollowProlongation", "impossible dZ-value of tracklet: track z = %f, tracklet z = %f, detToSearch = %i", t->GetZ(), fSpacePoints[trkltIdx].fX[2], detToSearch);
          }
        }
      }
    }
    if (bestGuessIdx != -1 ) {
      // best matching tracklet found
      p[0] = fSpacePoints[bestGuessIdx].fX[1];
      p[1] = fSpacePoints[bestGuessIdx].fX[2];
      cov[0] = TMath::Power(fSpacePoints[bestGuessIdx].fCov[0], 2);
      cov[1] = 0;
      cov[2] = TMath::Power(fSpacePoints[bestGuessIdx].fCov[1], 2);
      t->AddTracklet(iLayer, bestGuessIdx);
      //FIXME uncomment following lines to update track with tracklet information
      //if (!t->Update(p, cov)) {
      //  Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);
      //  return result;
      //}
      ++result;
      if (TMath::Abs(fSpacePoints[bestGuessIdx].fX[0] - fR[iLayer]) > 5) {
        Error("FollowProlongation", "tracklet from wrong layer added to track. fR[iLayer]=%f, xTracklet=%f", fR[iLayer], fSpacePoints[bestGuessIdx].fX[0]);
      }
    }
  }
  return result;
}

int AliHLTTRDTracker::GetDetectorNumber(const double zPos, double alpha, int layer)
{
  int stack = fTRDgeometry->GetStack(zPos, layer);
  if (stack < 0) {
    Info("GetDetectorNumber", "Stack determination failed for layer %i, alpha=%f, z=%f", layer, alpha, zPos);
    return -1;
  }
  double alphaTmp = (alpha > 0) ? alpha : alpha + 2. * TMath::Pi();
  int sector = 18. * alphaTmp / (2. * TMath::Pi());

  return fTRDgeometry->GetDetector(layer, stack, sector);
}

bool AliHLTTRDTracker::AdjustSector(AliHLTTRDTrack *t)
{
  // rotate track in new sector if necessary
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
