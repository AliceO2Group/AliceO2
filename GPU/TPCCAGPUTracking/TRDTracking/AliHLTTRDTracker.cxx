#include "AliHLTTRDTracker.h"
#include "AliGeomManager.h"
#include "AliTRDgeometry.h"
#include "TDatabasePDG.h"
#include <vector>
#include <iostream>
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
  delete fStreamer;
}


void AliHLTTRDTracker::Init()
{
  fTracklets = new AliHLTTRDTrackletWord[fNtrackletsMax];
  fSpacePoints = new AliHLTTRDSpacePointInternal[fNtrackletsMax];

  fTRDgeometry = new AliTRDgeometry();
  if (!fTRDgeometry) {
    Error("Init", "TRD geometry could not be loaded\n");
  }

  fStreamer = new TTreeSRedirector("debug.root", "recreate");
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
  //fNtrackletsInEvent = nTrklts;
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
  for (int iDet=0; iDet<540; ++iDet) {
    if (fTrackletIndexArray[iDet][0] != -1) {
      for (int iTrklt=0; iTrklt<fTrackletIndexArray[iDet][1]; ++iTrklt) {
        if (fTracklets[fTrackletIndexArray[iDet][0]+iTrklt].GetHCId()/2 != iDet) {
          printf("Bug in fTrackletIndexArray\n");
          printf("fTrackletIndexArray[%i][0]: %i\n", iDet, fTrackletIndexArray[iDet][0]);
          printf("HCId of tracklet: %i\n", fTracklets[fTrackletIndexArray[iDet][0]+iTrklt].GetHCId());
          Error("DoTracking", "bug in tracklet index array\n");
        }
      }
    }
  }
  CalculateSpacePoints();
  delete[] fTracks;
  fNTracks = 0;
  fTracks = new AliHLTTRDTrack[nTPCTracks];
  double piMass = TDatabasePDG::Instance()->GetParticle(211)->Mass(); //why pion mass??

  for (int i=0; i<nTPCTracks; ++i) {
    AliHLTTRDTrack tMI(tracksTPC[i]);
    AliHLTTRDTrack *t = &tMI;
    t->SetTPCtrackId(i);
    t->SetLabel(0); // for monte carlo tracks still TODO: set correct label
    t->SetMass(piMass);

    int result = FollowProlongation(t, piMass);
    //FindResiduals(t, piMass);
    t->SetNtracklets(result);
    /*
    if (result != 0) {
      printf("AliHLTTRDTracker:: add track with %i atttached tracklets\n", result);
    }
    */
    fTracks[fNTracks++] = *t;

    (*fStreamer) << "FollowProlongation" << "nTrackletsAttached=" << result << "\n";
  }
  (*fStreamer) << "Counter" << "nTrackletsTotal=" << fNTracklets << "nTPCtracksTotal=" << nTPCTracks << "\n";
}


void AliHLTTRDTracker::CalculateSpacePoints()
{
  // calculate the tracklet space points in sector tracking coordinates for all tracklets
  for (int iDet=0; iDet<540; ++iDet) {
    int stack = fTRDgeometry->GetStack(iDet);
    int layer = fTRDgeometry->GetLayer(iDet);
    int nTracklets = fTrackletIndexArray[iDet][1];
    if (nTracklets == 0) continue;
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
  // propagate TPC track through TRD and pick up
  // closest tracklet in each layer on the way
  // returns number of assigned tracklets

  //TODO ignore tracks ending up in PHOS hole

  int result = 0;
  int nSector = fTRDgeometry->Nsector();
  int nStack = fTRDgeometry->Nstack();

  for (int iLayer=0; iLayer<fTRDgeometry->Nlayer(); ++iLayer) {
    int stack[2] = { -1, -1 };
    int det[6] = { -1, -1, -1, -1, -1, -1 }; // look for tracklets in max 4 trd chambers //FIXME use proper index and start at 0, only 4 detectors needed
    int detCnt = 0; // number of chambers to be searched for tracklets

    float xLayer = fTRDgeometry->GetTime0(iLayer);
    if (!PropagateTrackToBxByBz(t, xLayer, mass, 5.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/) || abs(t->GetZ()) > 350.)
      return result;

    // debug purposes only
    double sigmaY = TMath::Sqrt(t->GetSigmaY2());
    double sigmaZ = TMath::Sqrt(t->GetSigmaZ2());
    switch (iLayer) {
      case 0:
        (*fStreamer) << "Layer0" << "sigmaY=" << sigmaY << "sigmaZ=" << sigmaZ << "nUpdates=" << result << "\n";
        break;
      case 1:
        (*fStreamer) << "Layer1" << "sigmaY=" << sigmaY << "sigmaZ=" << sigmaZ << "nUpdates=" << result << "\n";
        break;
      case 2:
        (*fStreamer) << "Layer2" << "sigmaY=" << sigmaY << "sigmaZ=" << sigmaZ << "nUpdates=" << result << "\n";
        break;
      case 3:
        (*fStreamer) << "Layer3" << "sigmaY=" << sigmaY << "sigmaZ=" << sigmaZ << "nUpdates=" << result << "\n";
        break;
      case 4:
        (*fStreamer) << "Layer4" << "sigmaY=" << sigmaY << "sigmaZ=" << sigmaZ << "nUpdates=" << result << "\n";
        break;
      case 5:
        (*fStreamer) << "Layer5" << "sigmaY=" << sigmaY << "sigmaZ=" << sigmaZ << "nUpdates=" << result << "\n";
        break;
    }

    // rotate track in new sector in case of sector crossing
    if (abs(t->GetY()) > (TMath::Tan(TMath::DegToRad()*10) * xLayer) ) {
      if (abs(t->GetY()) > ((TMath::Tan(TMath::DegToRad()*10) * xLayer) * ( 1. + 2 * TMath::Cos(TMath::DegToRad() * 20))) ) {
        Warning("FollowProlongation", "Stop following track crossing two sector boundaries");
        return result;
      }
      double rotAngle = t->GetAlpha();
      if (t->GetY() > 0) {
        rotAngle += 2. * TMath::Pi() / nSector;
      }
      else if (t->GetY() < 0) {
        rotAngle -= 2. * TMath::Pi() / nSector;
      }
      t->Rotate(rotAngle);
    }

    // determine chamber(s) where the track ends up
    int sector = TMath::Nint(nSector * (TMath::Pi() + t->GetAlpha()) / (2. * TMath::Pi()) - 0.5);
    stack[0] = fTRDgeometry->GetStack(t->GetZ(), iLayer);
    if (stack[0] < 0) {
      if (abs(t->GetZ()) > 300) { // shift track in z in case track ends up at the very end of the TRD
        double zNew = t->GetZ() > 0 ? t->GetZ() - 20. : t->GetZ() + 20.;
        stack[0] = fTRDgeometry->GetStack(zNew, iLayer);
        if (stack[0] < 0) {
          Info("FollowProlongation", "Determining stack failed twice in outer chamber: z=%f, layer=%i, stack=%i\n", t->GetZ(), iLayer, stack[0]);
          return result;
        }
      }
      else { //track ends up in between stacks of one sector -> search in both neighbouring chambers
        stack[0] = fTRDgeometry->GetStack((t->GetZ() - 10.), iLayer);
        stack[1] = fTRDgeometry->GetStack((t->GetZ() + 10.), iLayer);
        if (stack[0] < 0 || stack[1] < 0) {
          Error("FollowProlongation", "gap between stacks of one sector too large");
          printf("Determining stack failed twice in inner chamber(s): z=%f, layer=%i\n", t->GetZ(), iLayer);
          printf("stack[0] = %i, stack[1] = %i\n", stack[0], stack[1]);
          return result;
        }
        det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[1], sector);
      }
    }
    det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[0], sector);

    // define search window for tracklets
    double deltaY = 7. * TMath::Sqrt(t->GetSigmaY2() + TMath::Power(0.03, 2));
    double deltaZ = 7. * TMath::Sqrt(t->GetSigmaZ2() + TMath::Power(9./TMath::Sqrt(12), 2));

    if (abs(t->GetZ()) + deltaZ > fTRDgeometry->GetChamberLength(iLayer, stack[0]) / 2. ) {
      if (stack[1] != -1) {
        continue;
      }
      else {
        if (stack[0] == 2) {
          stack[1] = t->GetZ() > 0 ? stack[0] - 1 : stack[0] + 1;
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[1], sector);
        }
        else {
          AliTRDpadPlane *pp = fTRDgeometry->GetPadPlane(iLayer, stack[0]);
          if (t->GetZ() > (pp->GetRowEnd() + abs(pp->GetRow0() - pp->GetRowEnd()))) {
            stack[1] = stack[0] - 1;
          }
          else {
            stack[1] = stack[0] + 1;
          }
          if (stack[1] < 0 || stack[1] > nStack - 1) {
            continue;
          }
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[1], sector);
        }
      }
    }

    if (abs(t->GetY()) + deltaY > fTRDgeometry->GetChamberWidth(iLayer) / 2.) {
      // TODO: check whether this makes sense
    /*
      if (t->GetY() < 0) {
        if (sector == 0) {
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[0], nSector-1);
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[1], nSector-1);
        }
        else {
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[0], sector-1);
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[1], sector-1);
        }
      }
      else {
        if (sector == nSector-1) {
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[0], 0);
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[1], 0);
        }
        else {
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[0], sector+1);
          det[detCnt++] = fTRDgeometry->GetDetector(iLayer, stack[1], sector+1);
        }
      }
    */
    }

    // look for tracklets in chamber(s)
    double bestGuessChi2 = 10000.;
    int bestGuessIdx = -1;
    double p[2] = { 0. };
    double cov[3] = { 0. };
    for (int iDet=0; iDet<detCnt; ++iDet) {
      for (int iTrklt=0; iTrklt<fTrackletIndexArray[det[iDet]][1]; ++iTrklt) {
        int trkltIdx = fTrackletIndexArray[det[iDet]][0] + iTrklt;
        if ((fSpacePoints[trkltIdx].fX[1] < t->GetY() + deltaY) && (fSpacePoints[trkltIdx].fX[1] > t->GetY() - deltaY) &&
            (fSpacePoints[trkltIdx].fX[2] < t->GetZ() + deltaZ) && (fSpacePoints[trkltIdx].fX[2] > t->GetZ() - deltaZ))
        {
          //tracklet is in windwow: get predicted chi2 for update and store tracklet index if best guess
          p[0] = fSpacePoints[trkltIdx].fX[1];
          p[1] = fSpacePoints[trkltIdx].fX[2];
          cov[0] = TMath::Power(0.03, 2);
          cov[1] = 0;
          cov[2] = TMath::Power(9./TMath::Sqrt(12), 2);
          //double chi2 = ((AliExternalTrackParam) *t).GetPredictedChi2(p, cov);
          double chi2 = t->GetPredictedChi2(p, cov);
          if (chi2 < bestGuessChi2) {
            bestGuessChi2 = chi2;
            bestGuessIdx = trkltIdx;
          }
          float dY = t->GetY() - fSpacePoints[trkltIdx].fX[1];
          float dZ = t->GetZ() - fSpacePoints[trkltIdx].fX[2];
          (*fStreamer) << "deltaY" << "dY=" << dY << "\n";
          (*fStreamer) << "deltaZ" << "dZ=" << dZ << "\n";
        }
      }
    }
    if (bestGuessIdx != -1 && bestGuessChi2 < 100 /* TODO define meaningful chi2 cut */ ) {
      // best matching tracklet found
      p[0] = fSpacePoints[bestGuessIdx].fX[1];
      p[1] = fSpacePoints[bestGuessIdx].fX[2];
      cov[0] = TMath::Power(0.03, 2);
      cov[1] = 0;
      cov[2] = TMath::Power(9./TMath::Sqrt(12), 2);
      ((AliExternalTrackParam) *t).Update(p, cov);
      ++result;
      //printf("Closest tracklet number %i found in layer %i with deltaX = %f, deltaY = %f, deltaZ = %f\n", result, iLayer, t->GetX()-xLayer, t->GetY()-p[0], t->GetZ()-p[1]);
      //printf("t->GetX()=%f, xLayer=%f\n", t->GetX(), xLayer);
      //printf("t->GetY()=%f, p[0]=%f\n", t->GetY(), p[0]);
      //printf("t->GetZ()=%f, p[1]=%f\n", t->GetZ(), p[1]);
    }
  }
  // after propagation, propagate track back to inner radius of TPC
  float xInnerParam = 83.65;
  if (!PropagateTrackToBxByBz(t, xInnerParam, mass, 5.0 /*max step*/, kFALSE /*rotateTo*/, 0.8 /*maxSnp*/)) {
    Warning("FollowProlongation", "Back propagation for track failed");
    return result;
  }

  // rotate track back in old sector in case of sector crossing
  if (abs(t->GetY()) > (TMath::Tan(TMath::DegToRad()*10) * xInnerParam) ) {
    if (abs(t->GetY()) > ((TMath::Tan(TMath::DegToRad()*10) * xInnerParam) * ( 1. + 2 * TMath::Cos(TMath::DegToRad() * 20))) ) {
      Warning("FollowProlongation", "Stop following track crossing two sector boundaries during inward propagation");
      return result;
    }
    double rotAngle = t->GetAlpha();
    if (t->GetY() > 0) {
      rotAngle += 2. * TMath::Pi() / nSector;
    }
    else if (t->GetY() < 0) {
      rotAngle -= 2. * TMath::Pi() / nSector;
    }
    t->Rotate(rotAngle);
  }

  return result;
}

void AliHLTTRDTracker::FindResiduals(AliHLTTRDTrack *t, double mass)
{
  //TODO
}
