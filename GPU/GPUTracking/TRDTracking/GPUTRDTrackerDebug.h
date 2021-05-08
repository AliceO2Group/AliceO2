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
#include "GPULogging.h"
#include "GPUTRDTrack.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <class T>
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
    fChi2Update.ResizeTo(6);
    fFindable.ResizeTo(6);
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
    fChi2Update.Zero();
    fFindable.Zero();
    fEv = 0;
    fNTPCtracks = 0;
    fTrk = 0;
    fPtTPC = 0.f;
    fNtrklts = 0;
    fNlayers = 0;
    fChi2 = 0.f;
  }

  // general information
  void SetGeneralInfo(int iEv, int nTPCtracks, int iTrk, float pt)
  {
    fEv = iEv;
    fNTPCtracks = nTPCtracks;
    fTrk = iTrk;
    fPtTPC = pt;
  }

  // track parameters
  void SetTrackParameter(const T& trk, int ly)
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
  void SetTrackParameterNoUp(const T& trk, int ly)
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
  void SetTrack(const T& trk)
  {
    fChi2 = trk.getChi2();
    fNlayers = trk.getNlayers();
    fNtrklts = trk.getNtracklets();
    for (int iLy = 0; iLy < 6; iLy++) {
      if (trk.getIsFindable(iLy)) {
        fFindable(iLy) = 1;
      }
    }
  }

  // tracklet parameters
  void SetRawTrackletPosition(const float fX, const float fY, const float fZ, int ly)
  {
    fTrackletX(ly) = fX;
    fTrackletY(ly) = fY;
    fTrackletZ(ly) = fZ;
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

  // update information
  void SetChi2Update(float chi2, int ly) { fChi2Update(ly) = chi2; }

  // other infos
  void SetRoad(float roadY, float roadZ, int ly)
  {
    fRoadY(ly) = roadY;
    fRoadZ(ly) = roadZ;
  }

  void Output()
  {
    (*fStreamer) << "tracksFinal"
                 << "event=" << fEv <<                     // event number
      "nTPCtracks=" << fNTPCtracks <<                      // total number of TPC tracks for this event
      "iTrack=" << fTrk <<                                 // track index in event
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
      "chi2Update.=" << &fChi2Update <<                    // chi2 for update
      "chi2Total=" << fChi2 <<                             // total chi2 for track
      "nLayers=" << fNlayers <<                            // number of layers in which track was findable
      "nTracklets=" << fNtrklts <<                         // number of attached tracklets
      "roadY.=" << &fRoadY <<                              // search road width in Y
      "roadZ.=" << &fRoadZ <<                              // search road width in Z
      "findable.=" << &fFindable <<                        // whether or not track was in active TRD volume (layerwise)
      "\n";
  }

 private:
  int fEv;
  int fNTPCtracks;
  int fTrk;
  float fPtTPC;
  int fNlayers;
  float fChi2;
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
  TVectorF fChi2Update;
  TVectorF fRoadY;
  TVectorF fRoadZ;
  TVectorF fFindable;

  TTreeSRedirector* fStreamer;
};
template class GPUTRDTrackerDebug<GPUTRDTrack>;
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#else

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <class T>
class GPUTRDTrackerDebug
{
 public:
  GPUd() void CreateStreamer() {}
  GPUd() void ExpandVectors() {}
  GPUd() void Reset() {}

  // general information
  GPUd() void SetGeneralInfo(int iEv, int nTPCtracks, int iTrk, float pt) {}

  // track parameters
  GPUd() void SetTrackParameter(const T& trk, int ly) {}
  GPUd() void SetTrackParameterNoUp(const T& trk, int ly) {}
  GPUd() void SetTrack(const T& trk) {}

  // tracklet parameters
  GPUd() void SetRawTrackletPosition(const float fX, const float fY, const float fZ, int ly) {}
  GPUd() void SetCorrectedTrackletPosition(const My_Float* fYZ, int ly) {}
  GPUd() void SetTrackletCovariance(const My_Float* fCov, int ly) {}
  GPUd() void SetTrackletProperties(const float dy, const int det, int ly) {}

  // update information
  GPUd() void SetChi2Update(float chi2, int ly) {}
  GPUd() void SetChi2YZPhiUpdate(float chi2, int ly) {}

  // other infos
  GPUd() void SetRoad(float roadY, float roadZ, int ly) {}
  GPUd() void SetFindable(bool* findable) {}
  GPUd() void Output() {}
};
#ifndef GPUCA_ALIROOT_LIB
template class GPUTRDTrackerDebug<GPUTRDTrackGPU>;
#endif
#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
template class GPUTRDTrackerDebug<GPUTRDTrack>;
#endif
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
#endif // GPUTRDTRACKERDEBUG_H
