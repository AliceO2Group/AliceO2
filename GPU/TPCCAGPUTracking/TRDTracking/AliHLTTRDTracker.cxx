#include <vector>
#include "AliHLTTRDTracker.h"
#include "AliGeomManager.h"
#include "AliTRDgeometry.h"
#include "TDatabasePDG.h"
#include "TTreeStream.h"
#include "TGeoMatrix.h"

ClassImp(AliHLTTRDTracker)

AliHLTTRDTracker::AliHLTTRDTracker() :
  fTracks(0x0),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fSpacePoints(0x0),
  fTRDgeometry(0x0),
  fEnableDebugOutput(false),
  fStreamer(0x0)
{
  for (int iDet=0; iDet<540; ++iDet) {
    fTrackletIndexArray[iDet][0] = -1;
    fTrackletIndexArray[iDet][1] = 0;
  }
  //Default constructor
}

AliHLTTRDTracker::AliHLTTRDTracker(const AliHLTTRDTracker &tracker) :
  fTracks(0x0),
  fNTracks(0),
  fNEvents(0),
  fTracklets(0x0),
  fNtrackletsMax(1000),
  fNTracklets(0),
  fSpacePoints(0x0),
  fTRDgeometry(0x0),
  fEnableDebugOutput(false),
  fStreamer(0x0)
{
  //Copy constructor
  //Dummy!
  for (int iDet=0; iDet<540; ++iDet) {
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
  delete[] fTracklets;
  delete[] fTracks;
  delete[] fSpacePoints;
  delete fTRDgeometry;
  if (fEnableDebugOutput) {
    delete fStreamer;
  }
}


void AliHLTTRDTracker::Init()
{
  fTracklets = new AliHLTTRDTrackletWord[fNtrackletsMax];
  fSpacePoints = new AliHLTTRDSpacePointInternal[fNtrackletsMax];

  fTRDgeometry = new AliTRDgeometry();
  if (!fTRDgeometry) {
    Error("Init", "TRD geometry could not be loaded\n");
  }

  if (fEnableDebugOutput) {
    fStreamer = new TTreeSRedirector("debug.root", "recreate");
  }
}

void AliHLTTRDTracker::Reset()
{
  fNTracklets = 0;
  for (int i=0; i<fNtrackletsMax; ++i) {
    fTracklets[i] = 0x0;
    fSpacePoints[i].fX[0]     = 0.;
    fSpacePoints[i].fX[1]     = 0.;
    fSpacePoints[i].fX[2]     = 0.;
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
    Error("LoadTracklet", "running out of memory for tracklets, skipping tracklet(s). This should actually never happen.\n");
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
  for (int iDet=0; iDet<540; ++iDet) {
    if (fTrackletIndexArray[iDet][1] != 0) {
      fTrackletIndexArray[iDet][0] = trkltCounter;
      trkltCounter += fTrackletIndexArray[iDet][1];
    }
  }

  // test the correctness of the tracklet index array
  // this can be deleted later...
  if (!IsTrackletSortingOk()) {
    Error("DoTracking", "bug in tracklet index array\n");
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
    t->SetLabel(0); // use for monte carlo tracks later TODO: set correct label
    t->SetMass(piMass);

    int result = FollowProlongation(t, piMass);
    t->SetNtracklets(result);
    fTracks[fNTracks++] = *t;

    double pT = t->Pt();
    double alpha = t->GetAlpha();

    if (fEnableDebugOutput) {
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

  if (fEnableDebugOutput) {
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
  for (int iDet=0; iDet<540; iDet++) {
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

  for (int iDet=0; iDet<540; ++iDet) {

    int layer = fTRDgeometry->GetLayer(iDet);
    int stack = fTRDgeometry->GetStack(iDet);

    int nTracklets = fTrackletIndexArray[iDet][1];
    if (nTracklets == 0) {
      continue;
    }

    TGeoHMatrix *matrix = fTRDgeometry->GetClusterMatrix(iDet);
    if (!matrix){
	    Error("CalculateSpacePoints", "invalid TRD cluster matrix, skipping detector  %i\n", iDet);
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

      AliGeomManager::ELayerID iLayer = AliGeomManager::ELayerID(AliGeomManager::kTRD1+fTRDgeometry->GetLayer(iDet));
      int modId   = fTRDgeometry->GetSector(iDet) * AliTRDgeometry::kNstack + fTRDgeometry->GetStack(iDet);
      unsigned short volId = AliGeomManager::LayerToVolUID(iLayer, modId);
      fSpacePoints[trkltIdx].fVolumeId = volId;
    }
  }
}


int AliHLTTRDTracker::FollowProlongation(AliHLTTRDTrack *t, double mass)
{
  //--------------------------------------------------------------------
  // This function propagates found tracks with pT > 1.0 GeV
  // through the TRD and picks up the closest tracklet in each
  // layer on the way.
  // returns variable result = number of assigned tracklets
  //--------------------------------------------------------------------

  int result = 0;
  int iTrack = t->GetTPCtrackId(); // for debugging individual tracks

  // introduce momentum cut on tracks
  // particles with pT < 0.9 GeV highly unlikely to have matching online TRD tracklets
  if (t->Pt() < 1.0) {
    return result;
  }

  // define some constants
  const int nSector = fTRDgeometry->Nsector();
  const int nStack  = fTRDgeometry->Nstack();
  const int nLayer  = fTRDgeometry->Nlayer();

  AliTRDpadPlane *pad = 0x0;

  // the vector det holds the numbers of the detectors which are searched for tracklets
  std::vector<int> det;
  std::vector<int>::iterator iDet;

  for (int iLayer=0; iLayer<nLayer; ++iLayer) {

    det.clear();

    // where the trackl initially ends up
    int initialStack  = -1;
    int initialSector = -1;
    int detector      = -1;

    pad = fTRDgeometry->GetPadPlane(iLayer, 0);

    const float zMaxTRD = pad->GetRowPos(0);
    const float yMax    = pad->GetColPos(143) + pad->GetColSize(143);
    const float xLayer  = fTRDgeometry->GetTime0(iLayer);

    if (!PropagateTrackToBxByBz(t, xLayer, mass, 5.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/) || (abs(t->GetZ()) >= (zMaxTRD + 10.)) ) {
      return result;
    }

    // rotate track in new sector in case of sector crossing
    if (!AdjustSector(t)) {
      return result;
    }

    // debug purposes only -> store track information
    double sigmaY = TMath::Sqrt(t->GetSigmaY2());
    double sigmaZ = TMath::Sqrt(t->GetSigmaZ2());
    AliExternalTrackParam param(*t);
    if (fEnableDebugOutput) {
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
        if ( !(initialStack == 0 && t->GetZ() > 0) && !(initialStack == nStack-1 && t->GetZ() < 0) ) {
          // track not close to outer end of TRD -> add neighbouring stack
          if (t->GetZ() > zCenter) {
            det.push_back(fTRDgeometry->GetDetector(iLayer, initialStack-1, initialSector));
          }
          else {
            det.push_back(fTRDgeometry->GetDetector(iLayer, initialStack+1, initialSector));
          }
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
          if (newSector == nSector) {
            newSector = 0;
          }
          det.push_back(fTRDgeometry->GetDetector(iLayer, currStack, newSector));
        }
        else {
          int newSector = initialSector - 1;
          if (newSector == -1) {
            newSector = nSector - 1;
          }
          det.push_back(fTRDgeometry->GetDetector(iLayer, currStack, newSector));
        }
      }
    }

    if (fEnableDebugOutput) {
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

    if (fEnableDebugOutput) {
      (*fStreamer) << "searchWindow" <<
        "layer=" << iLayer <<
        "dY=" << deltaY <<
        "dZ=" << deltaZ <<
        "\n";
    }

    // look for tracklets in chamber(s)
    double bestGuessChi2 = 100.; // TODO define meaningful chi2 cut 
    int bestGuessIdx = -1;
    int bestGuessDet = -1;
    double p[2] = { 0. };
    double cov[3] = { 0. };
    bool wasTrackRotated = false;
    for (iDet = det.begin(); iDet != det.end(); ++iDet) {
      int detToSearch = *iDet;
      int stackToSearch = detToSearch / nLayer; // global stack number
      int sectorToSearch = fTRDgeometry->GetSector(detToSearch);
      if (sectorToSearch != initialSector && !wasTrackRotated) {
        float alphaToSearch = 2.0 * TMath::Pi() / (float) nSector * ((float) sectorToSearch + 0.5);
        t->Rotate(alphaToSearch);
        wasTrackRotated = true; // tracks need to be rotated max once per layer
      }
      for (int iTrklt=0; iTrklt<fTrackletIndexArray[detToSearch][1]; ++iTrklt) {
        int trkltIdx = fTrackletIndexArray[detToSearch][0] + iTrklt;
        if ((fSpacePoints[trkltIdx].fX[1] < t->GetY() + deltaY) && (fSpacePoints[trkltIdx].fX[1] > t->GetY() - deltaY) &&
            (fSpacePoints[trkltIdx].fX[2] < t->GetZ() + deltaZ) && (fSpacePoints[trkltIdx].fX[2] > t->GetZ() - deltaZ))
        {
          //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
          p[0] = fSpacePoints[trkltIdx].fX[1];
          p[1] = fSpacePoints[trkltIdx].fX[2];
          if (TMath::Abs(p[0]) > 1000) {
            Error("FollowProlongation", "impossible y-value of tracklet: y=%.12f\n", p[0]);
            return result;
          }
          cov[0] = TMath::Power(0.03, 2);
          cov[1] = 0;
          cov[2] = TMath::Power(9./TMath::Sqrt(12), 2);
          double chi2 = t->GetPredictedChi2(p, cov);
          if (chi2 < bestGuessChi2) {
            bestGuessChi2 = chi2;
            bestGuessIdx = trkltIdx;
            bestGuessDet = detToSearch;
          }
          float dY = t->GetY() - fSpacePoints[trkltIdx].fX[1];
          float dZ = t->GetZ() - fSpacePoints[trkltIdx].fX[2];
          if (fEnableDebugOutput) {
            (*fStreamer) << "residuals" <<
              "iEv=" << fNEvents <<
              "iTrk=" << iTrack <<
              "layer=" << iLayer <<
              "iTrklt=" << iTrklt <<
              "stack=" << stackToSearch <<
              "det=" << detToSearch <<
              "dY=" << dY <<
              "dZ=" << dZ <<
              "\n";
          }
          if (TMath::Abs(dZ) > 160) {
            Error("FollowProlongation", "impossible dZ-value of tracklet: track z = %f, tracklet z = %f, detToSearch = %i\n", t->GetZ(), fSpacePoints[trkltIdx].fX[2], detToSearch);
          }
        }
      }
    }
    if (bestGuessIdx != -1 ) {
      // best matching tracklet found
      p[0] = fSpacePoints[bestGuessIdx].fX[1];
      p[1] = fSpacePoints[bestGuessIdx].fX[2];
      cov[0] = TMath::Power(0.03, 2);
      cov[1] = 0;
      cov[2] = TMath::Power(9./TMath::Sqrt(12), 2);
      t->Update(p, cov);
      ++result;
    }
  }
  // after propagation, propagate track back to inner radius of TPC
  float xInnerParam = 83.65;
  if (!PropagateTrackToBxByBz(t, xInnerParam, mass, 5.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
    Warning("FollowProlongation", "Back propagation for track %i failed", iTrack);
    return result;
  }

  // rotate track back in old sector in case of sector crossing
  AdjustSector(t);

  return result;
}

// helper function for event display -> later not needed anymore
void AliHLTTRDTracker::Rotate(const double alpha, const double * const loc, double *glb)
{
  glb[0] = loc[0] * TMath::Cos(alpha) - loc[1] * TMath::Sin(alpha);
  glb[1] = loc[0] * TMath::Sin(alpha) + loc[1] * TMath::Cos(alpha);
  glb[2] = loc[2];
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
