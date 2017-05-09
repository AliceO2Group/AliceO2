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
      fSpacePoints[trkltIdx].fLabel = fTracklets[trkltIdx].GetLabel();
      fSpacePoints[trkltIdx].fCov[0] = TMath::Power(0.07, 2);
      fSpacePoints[trkltIdx].fCov[1] = TMath::Power(padPlane->GetRowSize(fTracklets[trkltIdx].GetZbin()), 2) / 12.;

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

    // where the track initially ends up
    int initialStack  = -1;
    int initialSector = -1;
    int detector      = -1;

    pad = fTRDgeometry->GetPadPlane(iLayer, 0);

    const float zMaxTRD = pad->GetRowPos(0);
    const float yMax    = pad->GetColPos(nCols-1) + pad->GetColSize(nCols-1);

    if (!PropagateTrackToBxByBz(t, fR[iLayer], mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
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

    // define search window for tracklets
    double deltaY = 7. * TMath::Sqrt(t->GetSigmaY2() + TMath::Power(0.05, 2)) + 2; // add constant to the road for better efficiency
    double deltaZ = 7. * TMath::Sqrt(t->GetSigmaZ2() + TMath::Power(9./TMath::Sqrt(12), 2)); // take longest pad length first

    if (abs(t->GetZ()) - deltaZ >= zMaxTRD ) {
      Info("FollowProlongation", "track out of TRD acceptance with z=%f in layer %i (eta=%f)", t->GetZ(), iLayer, t->Eta());
      return result;
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
      if ( ( t->GetZ() + deltaZ ) > padTmp->GetRowPos(0) || ( t->GetZ() - deltaZ ) < padTmp->GetRowPos(lastPadRow) ) {
        int extStack = t->GetZ() > zCenter ? initialStack - 1 : initialStack + 1;
        if (extStack < kNStacks && extStack > -1) {
          det.push_back(fTRDgeometry->GetDetector(iLayer, extStack, initialSector));
        }
      }
    }
    else {
      if (TMath::Abs(t->GetZ()) > zMaxTRD) {
        t->GetZ() > 0 ? // shift track in z so it is in the TRD acceptance
          detector = GetDetectorNumber(t->GetZ()- deltaZ, t->GetAlpha(), iLayer) :
          detector = GetDetectorNumber(t->GetZ()+ deltaZ, t->GetAlpha(), iLayer);
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
    if ( ( TMath::Abs(t->GetY()) + deltaY ) > yMax ) {
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
      if (sectorToSearch != GetSector(t->GetAlpha())) {
        Error("FollowProlongation", "track is in sector %i and sector %i is searched for tracklets", GetSector(t->GetAlpha()), sectorToSearch);
      }

      // - first propagate track to x of chamber
      // - for each tracklet: get track parameter at x of tracklet and compare y and z at this position
      /*
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
          Error("FollowProlongation", "track propagation failed while fine tuning track parameter for chamber. Abandoning track");
          return result;
        }
      }
      */
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
          cov[0] = fSpacePoints[trkltIdx].fCov[0];
          cov[1] = 0;
          cov[2] = fSpacePoints[trkltIdx].fCov[1];
          double chi2 = t->GetPredictedChi2(p, cov);
          if (chi2 < bestGuessChi2) {
            bestGuessChi2 = chi2;
            bestGuessIdx = trkltIdx;
            bestGuessDet = detToSearch;
          }
          if (fDebugOutput) {
            float dX = xTracklet - fSpacePoints[trkltIdx].fX[0]; // zero by definition
            float dY = trackYZ[0] - fSpacePoints[trkltIdx].fX[1];
            float dZ = trackYZ[1] - fSpacePoints[trkltIdx].fX[2];
            float trkErrorY = t->GetSigmaY2();
            float trkErrorZ = t->GetSigmaZ2();
            float trkltErrorY = fSpacePoints[trkltIdx].fCov[0];
            float trkltErrorZ = fSpacePoints[trkltIdx].fCov[1];
            float errorY = TMath::Sqrt(trkErrorY * trkErrorY + trkltErrorY * trkltErrorY);
            float errorZ = TMath::Sqrt(trkErrorZ * trkErrorZ + trkltErrorZ * trkltErrorZ);
            float pullY = dY / errorY;
            float pullZ = dZ / errorZ;
            bool updateInPrevLayer = false;
            if (iLayer != 0) {
              if (t->GetTracklet(iLayer-1) == -1) {
                updateInPrevLayer = true;
              }
            }
            bool correctUpdateInPrevLayer = false;
            if (updateInPrevLayer) {
              if (fSpacePoints[t->GetTracklet(iLayer-1)].fLabel == t->GetLabel()) {
                correctUpdateInPrevLayer = true;
              }
            }
            int charge = t->Charge();
            int trkLabel = t->GetLabel();
            int trkltLabel = fSpacePoints[trkltIdx].fLabel;
            bool match = (trkLabel == trkltLabel) ? true : false;
            AliExternalTrackParam parameter(*t);
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
              "updateInPrevLayer=" << updateInPrevLayer <<
              "correctUpdateInPrevLayer=" << correctUpdateInPrevLayer <<
              "charge=" << charge <<
              "trkLabel=" << trkLabel <<
              "trkltLabel=" << trkLabel <<
              "match=" << match <<
              "trkErrorY=" << trkErrorY <<
              "trkErrorZ=" << trkErrorZ <<
              "trkltErrorY=" << trkErrorY <<
              "trkltErrorZ=" << trkErrorZ <<
              "errorY=" << errorY <<
              "errorZ=" << errorZ <<
              "pullY=" << pullY <<
              "pullZ=" << pullZ <<
              "chi2=" << chi2 <<
              "parameter.=" << &parameter <<
              "\n";
          }
        }
      }
    }
    if (bestGuessIdx != -1 ) {
      // best matching tracklet found
      if (fTRDgeometry->GetSector(fTracklets[bestGuessIdx].GetDetector()) != GetSector(t->GetAlpha())) {
        t->Rotate( 2. * TMath::Pi() / (float) kNSectors * ((float) fTRDgeometry->GetSector(fTracklets[bestGuessIdx].GetDetector()) + 0.5));
      }
      p[0] = fSpacePoints[bestGuessIdx].fX[1];
      p[1] = fSpacePoints[bestGuessIdx].fX[2];
      cov[0] = fSpacePoints[bestGuessIdx].fCov[0];
      cov[1] = 0;
      cov[2] = fSpacePoints[bestGuessIdx].fCov[1];

      if (!PropagateTrackToBxByBz(t, fSpacePoints[bestGuessIdx].fX[0], mass, 2.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)){
        Error("FollowProlongation", "Final track propagation for update failed\n");
        continue;
      }

      if (fDebugOutput) {
        float trkltX = fSpacePoints[bestGuessIdx].fX[0];
        float trkltY = fSpacePoints[bestGuessIdx].fX[1];
        float trkltZ = fSpacePoints[bestGuessIdx].fX[2];
        float trkltErrY2 = cov[0];
        float trkltErrZ2 = cov[2];
        float xResidual = trkltX - t->GetX();
        float yResidual = trkltY - t->GetY();
        float zResidual = trkltZ - t->GetZ();
        float yPull = yResidual / TMath::Sqrt(t->GetSigmaY2() + trkltErrY2);
        float zPull = zResidual / TMath::Sqrt(t->GetSigmaZ2() + trkltErrZ2);
        bool prevUpdate = false;
        if (iLayer != 0) {
          if (t->GetTracklet(iLayer-1) != -1) {
            prevUpdate = true;
          }
        }
        AliExternalTrackParam parameterUpdate(*t);
        (*fStreamer) << "updates" <<
          "iEv=" << fNEvents <<
          "iTrk=" << iTrack <<
          "layer=" << iLayer <<
          "trkltX=" << trkltX <<
          "trkltY=" << trkltY <<
          "trkltZ=" << trkltZ <<
          "trkltErrY2=" << trkltErrY2 <<
          "trkltErrZ2=" << trkltErrZ2 <<
          "xResidual=" << xResidual <<
          "yResidual=" << yResidual <<
          "zResidual=" << zResidual <<
          "yPull=" << yPull <<
          "zPull=" << zPull <<
          "chi2=" << bestGuessChi2 <<
          "prevUpdate=" << prevUpdate <<
          "parameter.=" << &parameterUpdate <<
          "\n";
      }

      if (!t->Update(p, cov)) {
        Warning("FollowProlongation", "Failed to update track %i with space point in layer %i", iTrack, iLayer);
        continue;;
      }
      t->AddTracklet(iLayer, bestGuessIdx);
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

int AliHLTTRDTracker::GetSector(double alpha)
{
  if (alpha < 0) {
    alpha += 2. * TMath::Pi();
  }
  return (int) (alpha * kNSectors / (2. * TMath::Pi()));
}
