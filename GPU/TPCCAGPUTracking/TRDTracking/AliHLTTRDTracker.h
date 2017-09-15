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
    double fX[3];             // 3d position in sector coordinates
    double fCov[2];           // sigma_y^2, sigma_yz, sigma_z^2
    double fDy;               // deflection over drift length
    int fId;                  // index
    int fLabel;               // MC label
    unsigned short fVolumeId; // basically derived from TRD chamber number
  };

  enum Relation_t { kNoTracklet = 0, kNoMatch, kRelated, kEqual }; 

  struct Hypothesis {
    double fChi2;
    int fCandidateId;
    int fTrackletId;
  };

  static bool Hypothesis_Sort(const Hypothesis &lhs, const Hypothesis &rhs) {
    return (lhs.fChi2 < rhs.fChi2);
  }

  void Init();
  void Reset();
  void StartLoadTracklets(const int nTrklts);
  void LoadTracklet(const AliHLTTRDTrackletWord &tracklet);
  void DoTracking(AliExternalTrackParam *tracksTPC, int *tracksTPCLab, int nTPCTracks, int *tracksTPCnTrklts = 0x0);
  bool CalculateSpacePoints();
  bool FollowProlongation(AliHLTTRDTrack *t, double mass);
  void EnableDebugOutput() { fDebugOutput = true; }
  void SetPtThreshold(float minPt) { fMinPt = minPt; }
  void SetChi2Threshold(float maxChi2) { fMaxChi2 = maxChi2; }
  int GetDetectorNumber(const double zPos, double alpha, int layer);
  bool AdjustSector(AliHLTTRDTrack *t);
  int GetSector(double alpha);
  void CountMatches(int trkLbl, std::vector<int> *matches);
  bool FindChambersInRoad(AliHLTTRDTrack *t, float roadY, float roadZ, int iLayer, std::vector<int> &det, float zMax);
  bool IsFindable(float y, float z, float alpha, int layer);

  // for testing
  bool IsTrackletSortingOk();
  float GetPtThreshold() { return fMinPt; }
  float GetChi2Threshold() { return fMaxChi2; }


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

  void SetMCEvent(AliMCEvent* mc) {fMCEvent = mc;}

protected:

  static const double fgkX0[kNLayers];        // default values of anode wires

  double fR[kNLayers];                        // rough radial position of each TRD layer
  bool fIsInitialized;                        // flag is set upon initialization
  AliHLTTRDTrack *fTracks;                    // array of trd-updated tracks
  int fNTracks;                               // number of TPC tracks to be matched
  int fNEvents;                               // number of processed events
  AliHLTTRDTrackletWord *fTracklets;          // array of all tracklets, later sorted by HCId
  int fNtrackletsMax;                         // max number of tracklets
  int fNTracklets;                            // total number of tracklets in event
  int fTrackletIndexArray[kNChambers][2];     // index of first tracklet of each detector [iDet][0]
                                              // and number of tracklets in detector [iDet][1]
  Hypothesis *fHypothesis;                    // array with multiple track hypothesis
  AliHLTTRDTrack *fCandidates[2][kNcandidates];     // array of tracks for multiple hypothesis tracking
  AliHLTTRDSpacePointInternal *fSpacePoints;  // array with tracklet coordinates in global tracking frame
  AliTRDgeometry *fTRDgeometry;               // TRD geometry
  bool fDebugOutput;                          // store debug output
  float fMinPt;                               // min pt of TPC tracks for tracking
  float fMaxChi2;                             // max chi2 for tracklets
  double fChi2Penalty;                        // chi2 added to the track for no update
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
