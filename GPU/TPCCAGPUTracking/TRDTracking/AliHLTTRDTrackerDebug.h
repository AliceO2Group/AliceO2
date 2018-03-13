#ifdef ENABLE_HLTTRDDEBUG
# ifndef ALIHLTTRDTRACKERDEBUG_H
# define ALIHLTTRDTRACKERDEBUG_H

#include "TVectorF.h"
#include "TTreeStream.h"

class AliHLTTRDTrackerDebug
{
  public:
    AliHLTTRDTrackerDebug() { fStreamer = new TTreeSRedirector("TRDhlt.root", "recreate"); }
    ~AliHLTTRDTrackerDebug() { delete fStreamer; }

    int GetSector(float alpha) { if (alpha < 0) { alpha += 2. * TMath::Pi(); } return (int) (alpha * 18 / (2. * TMath::Pi())); }

    void ExpandVectors() { fTrackX.ResizeTo(6); fTrackY.ResizeTo(6); fTrackZ.ResizeTo(6); fTrackPhi.ResizeTo(6); fTrackLambda.ResizeTo(6); fTrackPt.ResizeTo(6);
                            fTrackSector.ResizeTo(6); fTrackYerr.ResizeTo(6); fTrackZerr.ResizeTo(6); fTrackNoUpX.ResizeTo(6); fTrackNoUpY.ResizeTo(6); fTrackNoUpZ.ResizeTo(6);
                            fTrackNoUpPhi.ResizeTo(6); fTrackNoUpLambda.ResizeTo(6); fTrackNoUpPt.ResizeTo(6); fTrackNoUpSector.ResizeTo(6); fTrackNoUpYerr.ResizeTo(6);
                            fTrackNoUpZerr.ResizeTo(6); fTrackletX.ResizeTo(6); fTrackletY.ResizeTo(6); fTrackletZ.ResizeTo(6); ; fTrackletYcorr.ResizeTo(6); fTrackletZcorr.ResizeTo(6);
                            fTrackletY2err.ResizeTo(6); fTrackletYZerr.ResizeTo(6); fTrackletZ2err.ResizeTo(6); fTrackletDy.ResizeTo(6); fTrackletDet.ResizeTo(6);
                            fRoadY.ResizeTo(6); fRoadZ.ResizeTo(6); fTrackletXReal.ResizeTo(6); fTrackletYReal.ResizeTo(6); fTrackletZReal.ResizeTo(6); ; fTrackletYcorrReal.ResizeTo(6); fTrackletZcorrReal.ResizeTo(6);
                            fTrackletSecReal.ResizeTo(6); fTrackletDetReal.ResizeTo(6); fTrackXReal.ResizeTo(6); fTrackYReal.ResizeTo(6); fTrackZReal.ResizeTo(6); fTrackSecReal.ResizeTo(6);
                            fChi2Update.ResizeTo(6); fChi2Real.ResizeTo(6); fNmatchesAvail.ResizeTo(6); fFindable.ResizeTo(6); fFindableMC.ResizeTo(6); fUpdates.ResizeTo(6);
                         }

    void Reset() { fTrackX.Zero(); fTrackY.Zero(); fTrackZ.Zero(); fTrackPhi.Zero(); fTrackLambda.Zero(); fTrackPt.Zero();
                    fTrackSector.Zero(); fTrackYerr.Zero(); fTrackZerr.Zero(); fTrackNoUpX.Zero(); fTrackNoUpY.Zero(); fTrackNoUpZ.Zero();
                    fTrackNoUpPhi.Zero(); fTrackNoUpLambda.Zero(); fTrackNoUpPt.Zero(); fTrackNoUpSector.Zero(); fTrackNoUpYerr.Zero();
                    fTrackNoUpZerr.Zero(); fTrackletX.Zero(); fTrackletY.Zero(); fTrackletZ.Zero(); ; fTrackletYcorr.Zero(); fTrackletZcorr.Zero();
                    fTrackletY2err.Zero(); fTrackletYZerr.Zero(); fTrackletZ2err.Zero(); fTrackletDy.Zero(); fTrackletDet.Zero();
                    fRoadY.Zero(); fRoadZ.Zero(); fTrackletXReal.Zero(); fTrackletYReal.Zero(); fTrackletZReal.Zero(); ; fTrackletYcorrReal.Zero(); fTrackletZcorrReal.Zero();
                    fTrackletSecReal.Zero(); fTrackletDetReal.Zero(); fTrackXReal.Zero(); fTrackYReal.Zero(); fTrackZReal.Zero(); fTrackSecReal.Zero();
                    fChi2Update.Zero(); fChi2Real.Zero(); fNmatchesAvail.Zero(); fFindable.Zero(); fFindableMC.Zero(); fUpdates.Zero();
                    fEv = 0; fNTPCtracks = 0; fTrk = 0; fTrackId = 0; fNtrklts = 0; fNtrkltsRef = 0; fNlayers = 0; fChi2 = 0; fNmatch = 0; fNfake = 0; fNrelated = 0;
                    fXvMC = 0; fYvMC = 0; fZvMC = 0; fPdgCode = 0; fParam.Reset(); fParamNoUp.Reset();
                 }

    // general information
    void SetGeneralInfo(int iEv, int nTPCtracks, int iTrk, int trackId)
            { fEv = iEv; fNTPCtracks = nTPCtracks; fTrk = iTrk; fTrackId = trackId; }
    void SetTrackProperties(int nMatch = 0, int nFake = 0, int nRelated = 0)
            { fNmatch = nMatch; fNfake = nFake; fNrelated = nRelated; }

    // track parameters
    void SetTrackParameter(const AliHLTTRDTrack &trk, int ly)
            { fTrackX(ly) = trk.GetX(); fTrackY(ly) = trk.GetY(); fTrackZ(ly) = trk.GetZ(); fTrackPhi(ly) = trk.GetSnp(); fTrackLambda(ly) = trk.GetTgl(); fTrackPt(ly) = trk.Pt(); fTrackSector(ly) = GetSector(trk.GetAlpha());
                fTrackYerr(ly) = trk.GetSigmaY2(); fTrackZerr(ly) = trk.GetSigmaZ2(); }
    void SetTrackParameterNoUp(const AliHLTTRDTrack &trk, int ly)
            { fTrackNoUpX(ly) = trk.GetX(); fTrackNoUpY(ly) = trk.GetY(); fTrackNoUpZ(ly) = trk.GetZ(); fTrackNoUpPhi(ly) = trk.GetSnp(); fTrackNoUpLambda(ly) = trk.GetTgl();
              fTrackNoUpPt(ly) = trk.Pt(); fTrackNoUpSector(ly) = GetSector(trk.GetAlpha()); fTrackNoUpYerr(ly) = trk.GetSigmaY2(); fTrackNoUpZerr(ly) = trk.GetSigmaZ2(); }
    void SetTrackParameterReal(const AliHLTTRDTrack &trk, int ly) { fTrackXReal(ly) = trk.GetX(); fTrackYReal(ly) = trk.GetY(); fTrackZReal(ly) = trk.GetZ(); fTrackSecReal(ly) = GetSector(trk.GetAlpha()); }
    void SetTrack(const AliHLTTRDTrack &trk) { fParam = trk; fChi2 = trk.GetChi2(); fNlayers = trk.GetNlayers(); fNtrklts = trk.GetNtracklets(); fNtrkltsRef = trk.GetNtrackletsOffline();
                                                for (int iLy=0; iLy<6; iLy++) { if (trk.GetIsFindable(iLy)) fFindable(iLy) = 1; } }
    void SetTrackNoUp(const AliHLTTRDTrack &trk) { fParamNoUp = trk; }

    // tracklet parameters
    void SetRawTrackletPosition(const float fX, const float (&fYZ)[2], int ly) { fTrackletX(ly) = fX; fTrackletY(ly) = fYZ[0]; fTrackletZ(ly) = fYZ[1]; }
    void SetCorrectedTrackletPosition(const double (&fYZ)[2], int ly) { fTrackletYcorr(ly) = fYZ[0]; fTrackletZcorr(ly) = fYZ[1]; }
    void SetTrackletCovariance(const double *fCov, int ly) { fTrackletY2err(ly) = fCov[0]; fTrackletYZerr(ly) = fCov[1]; fTrackletZ2err(ly) = fCov[2]; }
    void SetTrackletProperties(const float dy, const int det, int ly) { fTrackletDy(ly) = dy; fTrackletDet(ly) = det; }
    void SetRawTrackletPositionReal(float fX, float *fYZ, int ly) { fTrackletXReal(ly) = fX; fTrackletYReal(ly) = fYZ[0]; fTrackletZReal(ly) = fYZ[1]; }
    void SetCorrectedTrackletPositionReal(double *fYZ, int ly) { fTrackletYcorrReal(ly) = fYZ[0]; fTrackletZcorrReal(ly) = fYZ[1]; }
    void SetTrackletPropertiesReal(const int sec, const int det, int ly) { fTrackletSecReal(ly) = sec; fTrackletDetReal(ly) = det; }

    // update information
    void SetChi2Update(float chi2, int ly) { fChi2Update(ly) = chi2; }
    void SetChi2Real(float chi2, int ly) { fChi2Real(ly) = chi2; }

    // other infos
    void SetRoad(float roadY, float roadZ, int ly) { fRoadY(ly) = roadY; fRoadZ(ly) = roadZ; }
    void SetUpdates(int *up) { for (int iLy=0; iLy<6; iLy++) { fUpdates(iLy) = up[iLy]; } }
    void SetNmatchAvail(size_t i, int ly) { fNmatchesAvail(ly) = (int) i; };
    void SetFindableMC(bool *findableMC) { for (int iLy=0; iLy<6; iLy++) { fFindableMC(iLy) = findableMC[iLy]; } }
    void SetMCinfo(float xv, float yv, float zv, int pdg) { fXvMC = xv; fYvMC = yv; fZvMC = zv; fPdgCode = pdg; }

    void Output() {
      (*fStreamer) << "tracksFinal" <<
        "event=" << fEv <<                           // event number
        "nTPCtracks=" << fNTPCtracks <<              // total number of TPC tracks for this event
        "iTrack=" << fTrk <<                         // track number in event
        "trackID=" << fTrackId <<                    // MC track label
        "trackX.=" << &fTrackX <<
        "trackY.=" << &fTrackY <<
        "trackZ.=" << &fTrackZ <<
        "trackPhi.=" << &fTrackPhi <<                // phi angle of track (track.fP[2])
        "trackLambda.=" << &fTrackLambda <<          // lambda angle of track (track.fP[3])
        "trackPt.=" << &fTrackPt <<                  // track pT
        "trackYerr.=" << &fTrackYerr <<              // sigma_y^2 for track
        "trackZerr.=" << &fTrackZerr <<              // sigma_z^2 for track
        "trackSec.=" << &fTrackSector <<             // sector of track
        "trackNoUpX.=" << &fTrackNoUpX <<
        "trackNoUpY.=" << &fTrackNoUpY <<
        "trackNoUpZ.=" << &fTrackNoUpZ <<
        "trackNoUpPhi.=" << &fTrackNoUpPhi <<        // phi angle of track (track.fP[2])
        "trackNoUpLambda.=" << &fTrackNoUpLambda <<  // lambda angle of track (track.fP[3])
        "trackNoUpPt.=" << &fTrackNoUpPt <<          // track pT
        "trackNoUpYerr.=" << &fTrackNoUpYerr <<      // sigma_y^2 for track
        "trackNoUpZerr.=" << &fTrackNoUpZerr <<      // sigma_z^2 for track
        "trackNoUpSec.=" << &fTrackNoUpSector <<     // sector of track
        "trackletX.=" << &fTrackletX <<              // x position of tracklet used for update (sector coords)
        "trackletY.=" << &fTrackletYcorr <<          // y position of tracklet used for update (sector coords, tilt corrected position)
        "trackletZ.=" << &fTrackletZcorr <<          // z position of tracklet used for update (sector coords, tilt corrected position)
        "trackletYRaw.=" << &fTrackletY <<           // y position of tracklet used for update (sector coords)
        "trackletZRaw.=" << &fTrackletZ <<           // z position of tracklet used for update (sector coords)
        "trackletYerr.=" << &fTrackletY2err <<       // sigma_y^2 for tracklet
        "trackletYZerr.=" << &fTrackletYZerr <<      // sigma_yz for tracklet
        "trackletZerr.=" << &fTrackletZ2err <<       // sigma_z^2 for tracklet
        "trackletDy.=" << &fTrackletDy <<            // deflection for tracklet
        "trackletDet.=" << &fTrackletDet <<          // detector of tracklet
        "trackXReal.=" << &fTrackXReal <<
        "trackYReal.=" << &fTrackYReal <<
        "trackZReal.=" << &fTrackZReal <<
        "trackSecReal.=" << &fTrackSecReal <<
        "trackletXReal.=" << &fTrackletXReal <<       // x position (sector coords) for matching or related tracklet if available, otherwise 0
        "trackletYReal.=" << &fTrackletYcorrReal <<       // y position (sector coords, tilt correctet position) for matching or related tracklet if available, otherwise 0
        "trackletZReal.=" << &fTrackletZcorrReal <<       // z position (sector coords, tilt correctet position) for matching or related tracklet if available, otherwise 0
        "trackletYRawReal.=" << &fTrackletYReal << // y position (sector coords) for matching or related tracklet if available, otherwise 0
        "trackletZRawReal.=" << &fTrackletZReal << // z position (sector coords) for matching or related tracklet if available, otherwise 0
        "trackletSecReal.=" << &fTrackletSecReal <<   // sector number for matching or related tracklet if available, otherwise -1
        "trackletDetReal.=" << &fTrackletDetReal <<   // detector number for matching or related tracklet if available, otherwise -1
        "chi2Update.=" << &fChi2Update <<
        "chi2Real.=" << &fChi2Real <<
        "chi2Total=" << fChi2 <<
        "nLayers=" << fNlayers <<
        "nTracklets=" << fNtrklts <<
        "track.=" << &fParam <<
        "trackNoUp.=" << &fParamNoUp <<
        "roadY.=" << &fRoadY <<
        "roadZ.=" << &fRoadZ <<
        "findable.=" << &fFindable <<
        "findableMC.=" << &fFindableMC <<
        "update.=" << &fUpdates <<
        "nRelated=" << fNrelated <<
        "nMatching=" << fNmatch <<
        "nFake=" << fNfake <<
        "nMatchingTracklets.=" << &fNmatchesAvail <<  // number of matching + related tracklets for this track in each layer
        "XvMC=" << fXvMC <<                          // MC production vertex x
        "YvMC=" << fYvMC <<                          // MC production vertex y
        "ZvMC=" << fZvMC <<                          // MC production vertex z
        "pdgCode=" <<fPdgCode <<                     // MC PID
        "\n";
    }

  private:
    int fEv;
    int fNTPCtracks;
    int fTrk;
    int fTrackId;
    int fNtrklts;
    int fNtrkltsRef;
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
    AliExternalTrackParam fParam;
    AliExternalTrackParam fParamNoUp;
    float fXvMC;
    float fYvMC;
    float fZvMC;
    int fPdgCode;

    TTreeSRedirector *fStreamer;
};

# endif
#else
# ifndef ALIHLTTRDTRACKERDEBUG_H
# define ALIHLTTRDTRACKERDEBUG_H

class AliHLTTRDTrackerDebug
{
  public:
    void ExpandVectors() {}
    void Reset() {}

    // general information
    void SetGeneralInfo(int iEv, int nTPCtracks, int iTrk, int trackId) {}
    void SetTrackProperties(int nMatch = 0, int nFake = 0, int nRelated = 0) {}

    // track parameters
    void SetTrackParameter(const AliHLTTRDTrack &trk, int ly) {}
    void SetTrackParameterNoUp(const AliHLTTRDTrack &trk, int ly) {}
    void SetTrackParameterReal(const AliHLTTRDTrack &trk, int ly) {}
    void SetTrack(const AliHLTTRDTrack &trk) {}
    void SetTrackNoUp(const AliHLTTRDTrack &trk) {}

    // tracklet parameters
    void SetRawTrackletPosition(const float fX, const float (&fYZ)[2], int ly) {}
    void SetCorrectedTrackletPosition(const double (&fYZ)[2], int ly) {}
    void SetTrackletCovariance(const double *fCov, int ly) {}
    void SetTrackletProperties(const float dy, const int det, int ly) {}
    void SetRawTrackletPositionReal(float fX, float *fYZ, int ly) {}
    void SetCorrectedTrackletPositionReal(double *fYZ, int ly) {}
    void SetTrackletPropertiesReal(const int sec, const int det, int ly) {}

    // update information
    void SetChi2Update(float chi2, int ly) {}
    void SetChi2Real(float chi2, int ly) {}

    // other infos
    void SetRoad(float roadY, float roadZ, int ly) {}
    void SetUpdates(int *up) {}
    void SetNmatchAvail(size_t i, int ly) {}
    void SetFindable(bool *findable) {}
    void SetFindableMC(bool *findableMC) {}
    void SetMCinfo(float xv, float yv, float zv, int pdg) {}
    void Output() {}
};

# endif
#endif
