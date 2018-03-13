#ifndef ALIHLTTRDTRACKER_H
#define ALIHLTTRDTRACKER_H
/* Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


class TTree;
class AliHLTTRDTrackerDebug;
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
    kNChambers = 540
  };

  // struct to hold the information on the space points
  struct AliHLTTRDSpacePointInternal {
    float fR;                // x position (7mm above anode wires)
    float fX[2];             // y and z position (sector coordinates)
    double fCov[3];           // sigma_y^2, sigma_yz, sigma_z^2
    float fDy;               // deflection over drift length
    int fId;                  // index
    int fLabel[3];            // MC labels
    unsigned short fVolumeId; // basically derived from TRD chamber number
  };

  enum Relation_t { kNoTracklet = 0, kNoMatch, kRelated, kEqual };

  struct Hypothesis {
    float fChi2;
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
  int GetDetectorNumber(const float zPos, const float alpha, const int layer) const;
  bool AdjustSector(AliHLTTRDTrack *t, const int layer) const;
  int GetSector(float alpha) const;
  float GetAlphaOfSector(const int sec) const;
  float GetRPhiRes(float snp) const { return (0.04*0.04+0.33*0.33*(snp-0.126)*(snp-0.126)); }
  void RecalcTrkltCov(const int trkltIdx, const float tilt, const float snp, const float rowSize);
  void CountMatches(const int trackID, std::vector<int> *matches) const;
  void CheckTrackRefs(const int trackID, bool *findableMC) const;
  void FindChambersInRoad(const AliHLTTRDTrack *t, const float roadY, const float roadZ, const int iLayer, std::vector<int> &det, const float zMax) const;
  bool IsGeoFindable(const AliHLTTRDTrack *t, const int layer) const;

  // settings
  void SetMCEvent(AliMCEvent* mc)       { fMCEvent = mc;}
  void EnableDebugOutput()              { fDebugOutput = true; }
  void SetPtThreshold(float minPt)      { fMinPt = minPt; }
  void SetMaxEta(float maxEta)          { fMaxEta = maxEta; }
  void SetChi2Threshold(float chi2)     { fMaxChi2 = chi2; }
  void SetChi2Penalty(float chi2)       { fChi2Penalty = chi2; }
  void SetMaxMissingLayers(int ly)      { fMaxMissingLy = ly; }
  void SetNCandidates(int n)            { if (!fIsInitialized) fNCandidates = n; else Error("SetNCandidates", "Cannot change fNCandidates after initialization"); }

  AliMCEvent * GetMCEvent()   const { return fMCEvent; }
  bool  GetIsDebugOutputOn()  const { return fDebugOutput; }
  float GetPtThreshold()      const { return fMinPt; }
  float GetMaxEta()           const { return fMaxEta; }
  float GetChi2Threshold()    const { return fMaxChi2; }
  float GetChi2Penalty()      const { return fChi2Penalty; }
  int   GetMaxMissingLayers() const { return fMaxMissingLy; }
  int   GetNCandidates()      const { return fNCandidates; }
  void PrintSettings() const;

  // output
  AliHLTTRDTrack *Tracks()                    const { return fTracks;}
  int NTracks()                               const { return fNTracks;}
  AliHLTTRDSpacePointInternal *SpacePoints()  const { return fSpacePoints; }

  //----- Functions to be overwritten from AliTracker -----
  Int_t Clusters2Tracks(AliESDEvent *event) { return 0; }
  Int_t PropagateBack(AliESDEvent *event)   { return 0; }
  Int_t RefitInward(AliESDEvent *event)     { return 0; }
  Int_t LoadClusters(TTree *)               { return 0; }
  void UnloadClusters()                     { return; }
  AliCluster *GetCluster(Int_t index)       { return 0x0; }
  AliCluster *GetCluster(Int_t index) const { return 0x0; }


  AliHLTTRDTracker();
  virtual ~AliHLTTRDTracker();

protected:

  static const float fgkX0[kNLayers];        // default values of anode wires
  static const float fgkXshift;              // online tracklets evaluated above anode wire

  float *fR;                                  // rough radial position of each TRD layer
  bool fIsInitialized;                        // flag is set upon initialization
  AliHLTTRDTrack *fTracks;                    // array of trd-updated tracks
  int fNCandidates;                           // max. track hypothesis per layer
  int fNTracks;                               // number of TPC tracks to be matched
  int fNEvents;                               // number of processed events
  AliHLTTRDTrackletWord *fTracklets;          // array of all tracklets, later sorted by HCId
  int fNtrackletsMax;                         // max number of tracklets
  int fNTracklets;                            // total number of tracklets in event
  int *fNtrackletsInChamber;                  // number of tracklets in each chamber
  int *fTrackletIndexArray;                   // index of first tracklet for each chamber
  Hypothesis *fHypothesis;                    // array with multiple track hypothesis
  AliHLTTRDTrack *fCandidates;                // array of tracks for multiple hypothesis tracking
  AliHLTTRDSpacePointInternal *fSpacePoints;  // array with tracklet coordinates in global tracking frame
  AliTRDgeometry *fGeo;                       // TRD geometry
  bool fDebugOutput;                          // store debug output
  float fMinPt;                               // min pt of TPC tracks for tracking
  float fMaxEta;                              // TPC tracks with higher eta are ignored
  float fMaxChi2;                             // max chi2 for tracklets
  int fMaxMissingLy;                          // max number of missing layers per track
  float fChi2Penalty;                         // chi2 added to the track for no update
  float fZCorrCoefNRC;                        // tracklet z-position depends linearly on track dip angle
  int fNhypothesis;                           // number of track hypothesis per layer
  std::vector<int> fMaskedChambers;           // vector holding bad TRD chambers
  AliMCEvent* fMCEvent;                       //! externaly supplied optional MC event
  AliHLTTRDTrackerDebug *fDebug;              // debug output

private:
  AliHLTTRDTracker(const AliHLTTRDTracker &tracker);
  AliHLTTRDTracker & operator=(const AliHLTTRDTracker &tracker);

  ClassDef(AliHLTTRDTracker,0)   //HLT ITS tracker
};

#endif
