// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTrackerDebug.h
/// \brief For performance analysis + error parametrization of the TRD tracker

/// \author Ole Schmidt

#ifndef GPUTRDTRACKERDEBUG_H
#define GPUTRDTRACKERDEBUG_H

#if defined(ENABLE_GPUTRDDEBUG) && defined(GPUCA_ALIROOT_LIB)

#include "TVectorF.h"
#include "TTreeStream.h"
#include "GPUTRDTrack.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTRDTrackerDebug
{
 public:
  GPUTRDTrackerDebug() : fStreamer(0x0) {}
  ~GPUTRDTrackerDebug() { delete fStreamer; }

  void CreateStreamer()
  {
    GPUInfo("Creating streamer for debugging");
    fStreamer = new TTreeSRedirector("TRDhlt.root", "recreate");
  }

  int GetSector(float alpha)
  {
    if (alpha < 0) {
      alpha += 2. * M_PI;
    }
    return (int)(alpha * 18 / (2. * M_PI));
  }

  void ExpandVectors()
  {
    fTrackX.ResizeTo(6);
    fTrackY.ResizeTo(6);
    fTrackZ.ResizeTo(6);
    fTrackPhi.ResizeTo(6);
    fTrackLambda.ResizeTo(6);
    fTrackPt.ResizeTo(6);
    fTrackQPt.ResizeTo(6);
    fTrackSector.ResizeTo(6);
    fTrackYerr.ResizeTo(6);
    fTrackZerr.ResizeTo(6);
    fTrackNoUpX.ResizeTo(6);
    fTrackNoUpY.ResizeTo(6);
    fTrackNoUpZ.ResizeTo(6);
    fTrackNoUpPhi.ResizeTo(6);
    fTrackNoUpLambda.ResizeTo(6);
    fTrackNoUpPt.ResizeTo(6);
    fTrackNoUpSector.ResizeTo(6);
    fTrackNoUpYerr.ResizeTo(6);
    fTrackNoUpZerr.ResizeTo(6);
    fTrackletX.ResizeTo(6);
    fTrackletY.ResizeTo(6);
    fTrackletZ.ResizeTo(6);
    ;
    fTrackletYcorr.ResizeTo(6);
    fTrackletZcorr.ResizeTo(6);
    fTrackletY2err.ResizeTo(6);
    fTrackletYZerr.ResizeTo(6);
    fTrackletZ2err.ResizeTo(6);
    fTrackletDy.ResizeTo(6);
    fTrackletDet.ResizeTo(6);
    fRoadY.ResizeTo(6);
    fRoadZ.ResizeTo(6);
    fTrackletXReal.ResizeTo(6);
    fTrackletYReal.ResizeTo(6);
    fTrackletZReal.ResizeTo(6);
    ;
    fTrackletYcorrReal.ResizeTo(6);
    fTrackletZcorrReal.ResizeTo(6);
    fTrackletSecReal.ResizeTo(6);
    fTrackletDetReal.ResizeTo(6);
    fTrackXReal.ResizeTo(6);
    fTrackYReal.ResizeTo(6);
    fTrackZReal.ResizeTo(6);
    fTrackSecReal.ResizeTo(6);
    fChi2Update.ResizeTo(6);
    fChi2Real.ResizeTo(6);
    fNmatchesAvail.ResizeTo(6);
    fFindable.ResizeTo(6);
    fFindableMC.ResizeTo(6);
    fUpdates.ResizeTo(6);
  }

  void Reset()
  {
    fTrackX.Zero();
    fTrackY.Zero();
    fTrackZ.Zero();
    fTrackPhi.Zero();
    fTrackLambda.Zero();
    fTrackPt.Zero();
    fTrackQPt.Zero();
    fTrackSector.Zero();
    fTrackYerr.Zero();
    fTrackZerr.Zero();
    fTrackNoUpX.Zero();
    fTrackNoUpY.Zero();
    fTrackNoUpZ.Zero();
    fTrackNoUpPhi.Zero();
    fTrackNoUpLambda.Zero();
    fTrackNoUpPt.Zero();
    fTrackNoUpSector.Zero();
    fTrackNoUpYerr.Zero();
    fTrackNoUpZerr.Zero();
    fTrackletX.Zero();
    fTrackletY.Zero();
    fTrackletZ.Zero();
    ;
    fTrackletYcorr.Zero();
    fTrackletZcorr.Zero();
    fTrackletY2err.Zero();
    fTrackletYZerr.Zero();
    fTrackletZ2err.Zero();
    fTrackletDy.Zero();
    fTrackletDet.Zero();
    fRoadY.Zero();
    fRoadZ.Zero();
    fTrackletXReal.Zero();
    fTrackletYReal.Zero();
    fTrackletZReal.Zero();
    ;
    fTrackletYcorrReal.Zero();
    fTrackletZcorrReal.Zero();
    fTrackletSecReal.Zero();
    fTrackletDetReal.Zero();
    fTrackXReal.Zero();
    fTrackYReal.Zero();
    fTrackZReal.Zero();
    fTrackSecReal.Zero();
    fChi2Update.Zero();
    fChi2Real.Zero();
    fNmatchesAvail.Zero();
    fFindable.Zero();
    fFindableMC.Zero();
    fUpdates.Zero();
    fEv = 0;
    fNTPCtracks = 0;
    fTrk = 0;
    mTrackId = 0;
    fPtTPC = 0.f;
    fNtrklts = 0;
    fNtrkltsRef = 0;
    fNtrkltsRefMatch = 0;
    fNtrkltsRefRelated = 0;
    fNtrkltsRefFake = 0;
    fTrackIDref = -1;
    fNlayers = 0;
    fChi2 = 0.f;
    fNmatch = 0;
    fNfake = 0;
    fNrelated = 0;
    fXvMC = 0;
    fYvMC = 0;
    fZvMC = 0;
    fPdgCode = 0;
  }

  // general information
  void SetGeneralInfo(int iEv, int nTPCtracks, int iTrk, int trackId, float pt)
  {
    fEv = iEv;
    fNTPCtracks = nTPCtracks;
    fTrk = iTrk;
    mTrackId = trackId;
    fPtTPC = pt;
  }
  void SetTrackProperties(int nMatch = 0, int nFake = 0, int nRelated = 0)
  {
    fNmatch = nMatch;
    fNfake = nFake;
    fNrelated = nRelated;
  }

  // track parameters
  void SetTrackParameter(const GPUTRDTrack& trk, int ly)
  {
    fTrackX(ly) = trk.getX();
    fTrackY(ly) = trk.getY();
    fTrackZ(ly) = trk.getZ();
    fTrackPhi(ly) = trk.getSnp();
    fTrackLambda(ly) = trk.getTgl();
    fTrackPt(ly) = trk.getPt();
    fTrackQPt(ly) = trk.getQ2Pt();
    fTrackSector(ly) = GetSector(trk.getAlpha());
    fTrackYerr(ly) = trk.getSigmaY2();
    fTrackZerr(ly) = trk.getSigmaZ2();
  }
  void SetTrackParameterNoUp(const GPUTRDTrack& trk, int ly)
  {
    fTrackNoUpX(ly) = trk.getX();
    fTrackNoUpY(ly) = trk.getY();
    fTrackNoUpZ(ly) = trk.getZ();
    fTrackNoUpPhi(ly) = trk.getSnp();
    fTrackNoUpLambda(ly) = trk.getTgl();
    fTrackNoUpPt(ly) = trk.getPt();
    fTrackNoUpSector(ly) = GetSector(trk.getAlpha());
    fTrackNoUpYerr(ly) = trk.getSigmaY2();
    fTrackNoUpZerr(ly) = trk.getSigmaZ2();
  }
  void SetTrackParameterReal(const GPUTRDTrack& trk, int ly)
  {
    fTrackXReal(ly) = trk.getX();
    fTrackYReal(ly) = trk.getY();
    fTrackZReal(ly) = trk.getZ();
    fTrackSecReal(ly) = GetSector(trk.getAlpha());
  }
  void SetTrack(const GPUTRDTrack& trk)
  {
    fChi2 = trk.GetChi2();
    fNlayers = trk.GetNlayers();
    fNtrklts = trk.GetNtracklets();
    fNtrkltsRef = trk.GetNtrackletsOffline(0);
    fNtrkltsRefMatch = trk.GetNtrackletsOffline(1);
    fNtrkltsRefRelated = trk.GetNtrackletsOffline(2);
    fNtrkltsRefFake = trk.GetNtrackletsOffline(3);
    fTrackIDref = trk.GetLabelOffline();
    for (int iLy = 0; iLy < 6; iLy++) {
      if (trk.GetIsFindable(iLy)) {
        fFindable(iLy) = 1;
      }
    }
  }

  // tracklet parameters
  void SetRawTrackletPosition(const float fX, const float* fYZ, int ly)
  {
    fTrackletX(ly) = fX;
    fTrackletY(ly) = fYZ[0];
    fTrackletZ(ly) = fYZ[1];
  }
  void SetCorrectedTrackletPosition(const My_Float* fYZ, int ly)
  {
    fTrackletYcorr(ly) = fYZ[0];
    fTrackletZcorr(ly) = fYZ[1];
  }
  void SetTrackletCovariance(const My_Float* fCov, int ly)
  {
    fTrackletY2err(ly) = fCov[0];
    fTrackletYZerr(ly) = fCov[1];
    fTrackletZ2err(ly) = fCov[2];
  }
  void SetTrackletProperties(const float dy, const int det, int ly)
  {
    fTrackletDy(ly) = dy;
    fTrackletDet(ly) = det;
  }
  void SetRawTrackletPositionReal(float fX, float* fYZ, int ly)
  {
    fTrackletXReal(ly) = fX;
    fTrackletYReal(ly) = fYZ[0];
    fTrackletZReal(ly) = fYZ[1];
  }
  void SetCorrectedTrackletPositionReal(My_Float* fYZ, int ly)
  {
    fTrackletYcorrReal(ly) = fYZ[0];
    fTrackletZcorrReal(ly) = fYZ[1];
  }
  void SetTrackletPropertiesReal(const int det, int ly)
  {
    fTrackletSecReal(ly) = det / 30;
    fTrackletDetReal(ly) = det;
  }

  // update information
  void SetChi2Update(float chi2, int ly) { fChi2Update(ly) = chi2; }
  void SetChi2Real(float chi2, int ly) { fChi2Real(ly) = chi2; }

  // other infos
  void SetRoad(float roadY, float roadZ, int ly)
  {
    fRoadY(ly) = roadY;
    fRoadZ(ly) = roadZ;
  }
  void SetUpdates(int* up)
  {
    for (int iLy = 0; iLy < 6; iLy++) {
      fUpdates(iLy) = up[iLy];
    }
  }
  void SetNmatchAvail(size_t i, int ly) { fNmatchesAvail(ly) = (int)i; };
  void SetFindableMC(bool* findableMC)
  {
    for (int iLy = 0; iLy < 6; iLy++) {
      fFindableMC(iLy) = findableMC[iLy];
    }
  }
  void SetMCinfo(float xv, float yv, float zv, int pdg)
  {
    fXvMC = xv;
    fYvMC = yv;
    fZvMC = zv;
    fPdgCode = pdg;
  }

  void Output()
  {
    (*fStreamer) << "tracksFinal"
                 << "event=" << fEv <<                     // event number
      "nTPCtracks=" << fNTPCtracks <<                      // total number of TPC tracks for this event
      "iTrack=" << fTrk <<                                 // track index in event
      "trackID=" << mTrackId <<                            // TPC MC track label
      "trackPtTPC=" << fPtTPC <<                           // track pT before any propagation
      "trackX.=" << &fTrackX <<                            // x-pos of track (layerwise)
      "trackY.=" << &fTrackY <<                            // y-pos of track (layerwise)
      "trackZ.=" << &fTrackZ <<                            // z-pos of track (layerwise)
      "trackPhi.=" << &fTrackPhi <<                        // phi angle of track (track.fP[2])
      "trackLambda.=" << &fTrackLambda <<                  // lambda angle of track (track.fP[3])
      "trackQPt.=" << &fTrackQPt <<                        // track q/pT (track.fP[4])
      "trackPt.=" << &fTrackPt <<                          // track pT (layerwise)
      "trackYerr.=" << &fTrackYerr <<                      // sigma_y^2 for track
      "trackZerr.=" << &fTrackZerr <<                      // sigma_z^2 for track
      "trackSec.=" << &fTrackSector <<                     // TRD sector of track
      "trackNoUpX.=" << &fTrackNoUpX <<                    // x-pos of track w/o updates (layerwise)
      "trackNoUpY.=" << &fTrackNoUpY <<                    // y-pos of track w/o updates (layerwise)
      "trackNoUpZ.=" << &fTrackNoUpZ <<                    // z-pos of track w/o updates (layerwise)
      "trackNoUpPhi.=" << &fTrackNoUpPhi <<                // phi angle of track w/o updates (track.fP[2])
      "trackNoUpLambda.=" << &fTrackNoUpLambda <<          // lambda angle of track w/o updates (track.fP[3])
      "trackNoUpPt.=" << &fTrackNoUpPt <<                  // track pT w/o updates (layerwise)
      "trackNoUpYerr.=" << &fTrackNoUpYerr <<              // sigma_y^2 for track w/o updates
      "trackNoUpZerr.=" << &fTrackNoUpZerr <<              // sigma_z^2 for track w/o updates
      "trackNoUpSec.=" << &fTrackNoUpSector <<             // TRD sector of track w/o updates
      "trackletX.=" << &fTrackletX <<                      // x position of tracklet used for update (sector coords)
      "trackletY.=" << &fTrackletYcorr <<                  // y position of tracklet used for update (sector coords, tilt corrected position)
      "trackletZ.=" << &fTrackletZcorr <<                  // z position of tracklet used for update (sector coords, tilt corrected position)
      "trackletYRaw.=" << &fTrackletY <<                   // y position of tracklet used for update (sector coords)
      "trackletZRaw.=" << &fTrackletZ <<                   // z position of tracklet used for update (sector coords)
      "trackletYerr.=" << &fTrackletY2err <<               // sigma_y^2 for tracklet
      "trackletYZerr.=" << &fTrackletYZerr <<              // sigma_yz for tracklet
      "trackletZerr.=" << &fTrackletZ2err <<               // sigma_z^2 for tracklet
      "trackletDy.=" << &fTrackletDy <<                    // deflection for tracklet
      "trackletDet.=" << &fTrackletDet <<                  // TRD chamber of tracklet
      "trackXReal.=" << &fTrackXReal <<                    // x-pos for track at first found tracklet radius w/ matching MC label
      "trackYReal.=" << &fTrackYReal <<                    // y-pos for track at first found tracklet radius w/ matching MC label
      "trackZReal.=" << &fTrackZReal <<                    // z-pos for track at first found tracklet radius w/ matching MC label
      "trackSecReal.=" << &fTrackSecReal <<                // TRD sector for track at first found tracklet w/ matching MC label
      "trackletXReal.=" << &fTrackletXReal <<              // x position (sector coords) for matching or related tracklet if available, otherwise 0
      "trackletYReal.=" << &fTrackletYcorrReal <<          // y position (sector coords, tilt correctet position) for matching or related tracklet if available, otherwise 0
      "trackletZReal.=" << &fTrackletZcorrReal <<          // z position (sector coords, tilt correctet position) for matching or related tracklet if available, otherwise 0
      "trackletYRawReal.=" << &fTrackletYReal <<           // y position (sector coords) for matching or related tracklet if available, otherwise 0
      "trackletZRawReal.=" << &fTrackletZReal <<           // z position (sector coords) for matching or related tracklet if available, otherwise 0
      "trackletSecReal.=" << &fTrackletSecReal <<          // sector number for matching or related tracklet if available, otherwise -1
      "trackletDetReal.=" << &fTrackletDetReal <<          // detector number for matching or related tracklet if available, otherwise -1
      "chi2Update.=" << &fChi2Update <<                    // chi2 for update
      "chi2Real.=" << &fChi2Real <<                        // chi2 for first tracklet w/ matching MC label
      "chi2Total=" << fChi2 <<                             // total chi2 for track
      "nLayers=" << fNlayers <<                            // number of layers in which track was findable
      "nTracklets=" << fNtrklts <<                         // number of attached tracklets
      "nTrackletsOffline=" << fNtrkltsRef <<               // number of attached offline tracklets
      "nTrackletsOfflineMatch=" << fNtrkltsRefMatch <<     // number of attached offline tracklets
      "nTrackletsOfflineRelated=" << fNtrkltsRefRelated << // number of attached offline tracklets
      "nTrackletsOfflineFake=" << fNtrkltsRefFake <<       // number of attached offline tracklets
      "labelRef=" << fTrackIDref <<                        // TRD MC track label from offline, if provided
      "roadY.=" << &fRoadY <<                              // search road width in Y
      "roadZ.=" << &fRoadZ <<                              // search road width in Z
      "findable.=" << &fFindable <<                        // whether or not track was in active TRD volume (layerwise)
      "findableMC.=" << &fFindableMC <<                    // whether or not a MC hit existed inside the TRD for the track (layerwise)
      "update.=" << &fUpdates <<                           // layerwise tracklet attachment (0 - no tracklet, [1-3] matching tracklet, [4-6] related tracklet, 9 fake tracklet)
      "nRelated=" << fNrelated <<                          // number of attached related tracklets
      "nMatching=" << fNmatch <<                           // number of attached matching tracklets
      "nFake=" << fNfake <<                                // number of attached fake tracklets
      "nMatchingTracklets.=" << &fNmatchesAvail <<         // number of matching + related tracklets for this track in each layer
      "XvMC=" << fXvMC <<                                  // MC production vertex x
      "YvMC=" << fYvMC <<                                  // MC production vertex y
      "ZvMC=" << fZvMC <<                                  // MC production vertex z
      "pdgCode=" << fPdgCode <<                            // MC PID
      "\n";
  }

 private:
  int fEv;
  int fNTPCtracks;
  int fTrk;
  int mTrackId;
  float fPtTPC;
  int fNtrklts;
  int fNtrkltsRef;
  int fNtrkltsRefMatch;
  int fNtrkltsRefRelated;
  int fNtrkltsRefFake;
  int fTrackIDref;
  int fNlayers;
  float fChi2;
  int fNmatch;
  int fNfake;
  int fNrelated;
  TVectorF fNmatchesAvail;
  TVectorF fTrackX;
  TVectorF fTrackY;
  TVectorF fTrackZ;
  TVectorF fTrackPhi;
  TVectorF fTrackLambda;
  TVectorF fTrackPt;
  TVectorF fTrackQPt;
  TVectorF fTrackSector;
  TVectorF fTrackYerr;
  TVectorF fTrackZerr;
  TVectorF fTrackNoUpX;
  TVectorF fTrackNoUpY;
  TVectorF fTrackNoUpZ;
  TVectorF fTrackNoUpPhi;
  TVectorF fTrackNoUpLambda;
  TVectorF fTrackNoUpPt;
  TVectorF fTrackNoUpSector;
  TVectorF fTrackNoUpYerr;
  TVectorF fTrackNoUpZerr;
  TVectorF fTrackletX;
  TVectorF fTrackletY;
  TVectorF fTrackletZ;
  TVectorF fTrackletYcorr;
  TVectorF fTrackletZcorr;
  TVectorF fTrackletY2err;
  TVectorF fTrackletYZerr;
  TVectorF fTrackletZ2err;
  TVectorF fTrackletDy;
  TVectorF fTrackletDet;
  TVectorF fTrackXReal;
  TVectorF fTrackYReal;
  TVectorF fTrackZReal;
  TVectorF fTrackSecReal;
  TVectorF fTrackletXReal;
  TVectorF fTrackletYReal;
  TVectorF fTrackletZReal;
  TVectorF fTrackletYcorrReal;
  TVectorF fTrackletZcorrReal;
  TVectorF fTrackletSecReal;
  TVectorF fTrackletDetReal;
  TVectorF fChi2Update;
  TVectorF fChi2Real;
  TVectorF fRoadY;
  TVectorF fRoadZ;
  TVectorF fFindable;
  TVectorF fFindableMC;
  TVectorF fUpdates;
  float fXvMC;
  float fYvMC;
  float fZvMC;
  int fPdgCode;

  TTreeSRedirector* fStreamer;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTRDTrackerDebug
{
 public:
  GPUd() void CreateStreamer() {}
  GPUd() void ExpandVectors() {}
  GPUd() void Reset() {}

  // general information
  GPUd() void SetGeneralInfo(int iEv, int nTPCtracks, int iTrk, int trackId, float pt) {}
  GPUd() void SetTrackProperties(int nMatch = 0, int nFake = 0, int nRelated = 0) {}

  // track parameters
  GPUd() void SetTrackParameter(const GPUTRDTrack& trk, int ly) {}
  GPUd() void SetTrackParameterNoUp(const GPUTRDTrack& trk, int ly) {}
  GPUd() void SetTrackParameterReal(const GPUTRDTrack& trk, int ly) {}
  GPUd() void SetTrack(const GPUTRDTrack& trk) {}

  // tracklet parameters
  GPUd() void SetRawTrackletPosition(const float fX, const float* fYZ, int ly) {}
  GPUd() void SetCorrectedTrackletPosition(const My_Float* fYZ, int ly) {}
  GPUd() void SetTrackletCovariance(const My_Float* fCov, int ly) {}
  GPUd() void SetTrackletProperties(const float dy, const int det, int ly) {}
  GPUd() void SetRawTrackletPositionReal(float fX, float* fYZ, int ly) {}
  GPUd() void SetCorrectedTrackletPositionReal(My_Float* fYZ, int ly) {}
  GPUd() void SetTrackletPropertiesReal(const int det, int ly) {}

  // update information
  GPUd() void SetChi2Update(float chi2, int ly) {}
  GPUd() void SetChi2Real(float chi2, int ly) {}

  // other infos
  GPUd() void SetRoad(float roadY, float roadZ, int ly) {}
  GPUd() void SetUpdates(int* up) {}
  GPUd() void SetNmatchAvail(size_t i, int ly) {}
  GPUd() void SetFindable(bool* findable) {}
  GPUd() void SetFindableMC(bool* findableMC) {}
  GPUd() void SetMCinfo(float xv, float yv, float zv, int pdg) {}
  GPUd() void Output() {}
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif // GPUTRDTRACKERDEBUG_H
