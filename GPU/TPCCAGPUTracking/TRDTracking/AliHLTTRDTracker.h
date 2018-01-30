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
#include "AliMCEvent.h"


//-------------------------------------------------------------------------
class AliHLTTRDTracker : public AliTracker {
public:

  enum EHLTTRDTracker {
    kNLayers = 6,
    kNStacks = 5,
    kNSectors = 18,
    kNChambers = 540,
    kNcandidates = 1
  };

  // struct to hold the information on the space points
  struct AliHLTTRDSpacePointInternal {
    double fR;                // x position (7mm above anode wires)
    double fX[2];             // y and z position (sector coordinates)
    double fCov[3];           // sigma_y^2, sigma_yz, sigma_z^2
    double fDy;               // deflection over drift length
    int fId;                  // index
    int fLabel[3];            // MC labels
    unsigned short fVolumeId; // basically derived from TRD chamber number
  };

  enum Relation_t { kNoTracklet = 0, kNoMatch, kRelated, kEqual };

  struct Hypothesis {
    double fChi2;
    int fLayers;
    int fCandidateId;
    int fTrackletId;
  };

  static bool Hypothesis_Sort(const Hypothesis &lhs, const Hypothesis &rhs) {
    if (lhs.fLayers < 1 || rhs.fLayers < 1) {
      return ( lhs.fChi2 < rhs.fChi2 );
    }
    else {
      return ( (lhs.fChi2/lhs.fLayers) < (rhs.fChi2/rhs.fLayers) );
    }
  }

  void Init();
  void Reset();
  void StartLoadTracklets(const int nTrklts);
  void LoadTracklet(const AliHLTTRDTrackletWord &tracklet);
  void DoTracking(AliExternalTrackParam *tracksTPC, int *tracksTPClab, int nTPCtracks, int *tracksTPCnTrklts = 0x0);
  bool CalculateSpacePoints();
  bool FollowProlongation(AliHLTTRDTrack *t, int nTPCtracks);
  int GetDetectorNumber(const double zPos, const double alpha, const int layer) const;
  bool AdjustSector(AliHLTTRDTrack *t, const int layer) const;
  int GetSector(double alpha) const;
  float GetAlphaOfSector(const int sec) const;
  void CountMatches(const int trackID, std::vector<int> *matches) const;
  void CheckTrackRefs(const int trackID, TVectorF &findableMC) const;
  void FindChambersInRoad(const AliHLTTRDTrack *t, const float roadY, const float roadZ, const int iLayer, std::vector<int> &det, const float zMax) const;
  bool IsFindable(const AliHLTTRDTrack *t, const int layer) const;

  // settings
  void SetMCEvent(AliMCEvent* mc) {fMCEvent = mc;}
  void EnableDebugOutput() { fDebugOutput = true; }
  void SetPtThreshold(float minPt) { fMinPt = minPt; }
  void SetMaxEta(float maxEta) { fMaxEta = maxEta; }
  void SetChi2Threshold(float maxChi2) { fMaxChi2 = maxChi2; }
  void SetMaxMissingLayers(int ly) {fMaxMissingLy = ly; }

  float GetPtThreshold() const { return fMinPt; }
  float GetMaxEta() const { return fMaxEta; }
  float GetChi2Threshold() const { return fMaxChi2; }
  int   GetMaxMissingLayers() const { return fMaxMissingLy; }

  // for testing
  bool IsTrackletSortingOk() const;

  AliHLTTRDTrack *Tracks() const { return fTracks;}
  int NTracks() const { return fNTracks;}
  AliHLTTRDSpacePointInternal *SpacePoints() const { return fSpacePoints; }

  //----- Functions to be overwritten from AliTracker -----
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

  static const double fgkX0[kNLayers];        // default values of anode wires
  static const double fgkXshift;              // online tracklets evaluated above anode wire

  double *fR;                                 // rough radial position of each TRD layer
  bool fIsInitialized;                        // flag is set upon initialization
  AliHLTTRDTrack *fTracks;                    // array of trd-updated tracks
  int fNTracks;                               // number of TPC tracks to be matched
  int fNEvents;                               // number of processed events
  AliHLTTRDTrackletWord *fTracklets;          // array of all tracklets, later sorted by HCId
  int fNtrackletsMax;                         // max number of tracklets
  int fNTracklets;                            // total number of tracklets in event
  int *fNtrackletsInChamber;                  // number of tracklets in each chamber
  int *fTrackletIndexArray;                   // index of first tracklet for each chamber
  Hypothesis *fHypothesis;                    // array with multiple track hypothesis
  AliHLTTRDTrack *fCandidates[2][kNcandidates];     // array of tracks for multiple hypothesis tracking
  AliHLTTRDSpacePointInternal *fSpacePoints;  // array with tracklet coordinates in global tracking frame
  AliTRDgeometry *fGeo;                       // TRD geometry
  bool fDebugOutput;                          // store debug output
  float fMinPt;                               // min pt of TPC tracks for tracking
  float fMaxEta;                              // TPC tracks with higher eta are ignored
  float fMaxChi2;                             // max chi2 for tracklets
  int fMaxMissingLy;                          // max number of missing layers per track
  double fChi2Penalty;                        // chi2 added to the track for no update
  double fZCorrCoefNRC;                       // tracklet z-position depends linearly on track dip angle
  int fNhypothesis;                           // number of track hypothesis per layer
  std::vector<int> fMaskedChambers;           // vector holding bad TRD chambers
  AliMCEvent* fMCEvent;                       //! externaly supplied optional MC event
  TTreeSRedirector *fStreamer;                // debug output stream

private:
  AliHLTTRDTracker(const AliHLTTRDTracker &tracker);
  AliHLTTRDTracker & operator=(const AliHLTTRDTracker &tracker);

  ClassDef(AliHLTTRDTracker,0)   //HLT ITS tracker
};

#endif
