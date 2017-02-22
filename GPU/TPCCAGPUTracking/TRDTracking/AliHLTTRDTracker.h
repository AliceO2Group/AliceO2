#ifndef ALIHLTTRDTRACKER_H
#define ALIHLTTRDTRACKER_H
/* Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


class TTree;
class TTreeSRedirector;
class AliESDEvent;
class AliESDtrack;

#include "AliHLTTRDTrackletWord.h"
#include "AliHLTTRDTrack.h"

#include "AliTracker.h"


//-------------------------------------------------------------------------
class AliHLTTRDTracker : public AliTracker {
public:

  // struct to hold the information on the space points
  struct AliHLTTRDSpacePointInternal {
    double fX[3];
    int fId;
    unsigned short fVolumeId;
  };

  void Init();
  void Reset();
  void StartLoadTracklets(const int nTrklts);
  void LoadTracklet(const AliHLTTRDTrackletWord &tracklet);
  void DoTracking(AliExternalTrackParam *tracksTPC, int *tracksTPCLab, int nTPCTracks);
  void CalculateSpacePoints();
  int FollowProlongation(AliHLTTRDTrack *t, double mass);
  void EnableDebugOutput() { fEnableDebugOutput = true; }
  void SetPtThreshold(float minPt) { fMinPt = minPt; }
  void Rotate(const double alpha, const double * const loc, double *glb);
  int GetDetectorNumber(const double zPos, double alpha, int layer);
  bool AdjustSector(AliHLTTRDTrack *t);

  // for testing
  bool IsTrackletSortingOk();
  float GetPtThreshold() { return fMinPt; }


  AliHLTTRDTrack *Tracks() const { return fTracks;}
  int NTracks() const { return fNTracks;}
  AliHLTTRDSpacePointInternal *SpacePoints() const { return fSpacePoints; }

  //----- Functions to be overridden from AliTracker -----
  Int_t Clusters2Tracks(AliESDEvent *event) { return 0; }
  Int_t PropagateBack(AliESDEvent *event) { return 0; }
  Int_t RefitInward(AliESDEvent *event) { return 0; }
  Int_t LoadClusters(TTree *) { return 0; }
  void UnloadClusters() { return; }
  AliCluster *GetCluster(Int_t index) { return 0x0; }
  AliCluster *GetCluster(Int_t index) const { return 0x0; }


  AliHLTTRDTracker();
  virtual ~AliHLTTRDTracker();


protected:


  AliHLTTRDTrack *fTracks;                    // array of trd-updated tracks
  int fNTracks;                               // number of TPC tracks to be matched
  int fNEvents;                               // number of processed events
  AliHLTTRDTrackletWord *fTracklets;          // array of all tracklets, later sorted by HCId
  int fNtrackletsMax;                         // max number of tracklets
  int fNTracklets;                            // total number of tracklets in event
  int fTrackletIndexArray[540][2];            // index of first tracklet of each detector [iDet][0]
                                              // and number of tracklets in detector [iDet][1]
  AliHLTTRDSpacePointInternal *fSpacePoints;  // array with tracklet coordinates in global tracking frame
  AliTRDgeometry *fTRDgeometry;               // TRD geometry
  bool fEnableDebugOutput;                    // store debug output
  float fMinPt;
  TTreeSRedirector *fStreamer;                // debug output stream

private:
  AliHLTTRDTracker(const AliHLTTRDTracker &tracker);
  AliHLTTRDTracker & operator=(const AliHLTTRDTracker &tracker);
  ClassDef(AliHLTTRDTracker,0)   //HLT ITS tracker
};

#endif
