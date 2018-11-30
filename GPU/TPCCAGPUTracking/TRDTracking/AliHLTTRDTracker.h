#ifndef ALIHLTTRDTRACKER_H
#define ALIHLTTRDTRACKER_H
/* Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include "AliTPCCommonDef.h"
#include "AliHLTTRDDef.h"
#include "AliHLTTPCCADef.h"
#include "AliHLTTRDTrackerDebug.h"

class AliHLTTRDTrackletWord;
class AliHLTTRDGeometry;
class AliExternalTrackParam;
class AliMCEvent;
class AliHLTTPCGMMerger;

//-------------------------------------------------------------------------
class AliHLTTRDTracker {
 public:

#ifndef HLTCA_GPUCODE
  AliHLTTRDTracker();
  AliHLTTRDTracker(const AliHLTTRDTracker &tracker) CON_DELETE;
  AliHLTTRDTracker & operator=(const AliHLTTRDTracker &tracker) CON_DELETE;
  ~AliHLTTRDTracker();
#endif

  enum EHLTTRDTracker {
    kNLayers = 6,
    kNStacks = 5,
    kNSectors = 18,
    kNChambers = 540
  };

  // struct to hold the information on the space points
  struct AliHLTTRDSpacePointInternal {
    float fR;                 // x position (7mm above anode wires)
    float fX[2];              // y and z position (sector coordinates)
    My_Float fCov[3];         // sigma_y^2, sigma_yz, sigma_z^2
    float fDy;                // deflection over drift length
    int fId;                  // index
    int fLabel[3];            // MC labels
    unsigned short fVolumeId; // basically derived from TRD chamber number
  };

  struct Hypothesis {
    int fLayers;
    int fCandidateId;
    int fTrackletId;
    float fChi2;

    Hypothesis() : fLayers(0), fCandidateId(-1), fTrackletId(-1), fChi2(9999.f) {}
  };

  GPUd() size_t SetPointersBase(void* base, int maxThreads = 1, bool doConstruct = false);
  GPUd() size_t SetPointersTracklets(void* base);
  GPUd() size_t SetPointersTracks(void *base, int nTracks);

  GPUd() bool Init(AliHLTTRDGeometry *geo = nullptr);
  GPUd() void Reset();
  GPUd() void StartLoadTracklets(const int nTrklts);
  GPUd() void LoadTracklet(const AliHLTTRDTrackletWord &tracklet);
  GPUd() void DoTracking(HLTTRDTrack *tracksTPC, int *tracksTPClab, int nTPCtracks, int *tracksTPCnTrklts = 0x0, int *tracksTRDlabel = 0x0);
  GPUd() void DoTrackingThread(HLTTRDTrack *tracksTPC, int *tracksTPClab, int nTPCtracks, int iTrk, int threadId, int *tracksTPCnTrklts = 0x0, int *tracksTRDlabel = 0x0);
  GPUd() bool CalculateSpacePoints();
  GPUd() bool FollowProlongation(HLTTRDPropagator *prop, HLTTRDTrack *t, int nTPCtracks, int threadId);
  GPUd() int GetDetectorNumber(const float zPos, const float alpha, const int layer) const;
  GPUd() bool AdjustSector(HLTTRDPropagator *prop, HLTTRDTrack *t, const int layer) const;
  GPUd() int GetSector(float alpha) const;
  GPUd() float GetAlphaOfSector(const int sec) const;
  GPUd() float GetRPhiRes(float snp) const { return (0.04f*0.04f+0.33f*0.33f*(snp-0.126f)*(snp-0.126f)); } // parametrization obtained from track-tracklet residuals
  GPUd() void RecalcTrkltCov(const int trkltIdx, const float tilt, const float snp, const float rowSize, My_Float (&cov)[3]);
  void CountMatches(const int trackID, std::vector<int> *matches) const;
  GPUd() void CheckTrackRefs(const int trackID, bool *findableMC) const;
  GPUd() void FindChambersInRoad(const HLTTRDTrack *t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const;
  GPUd() bool IsGeoFindable(const HLTTRDTrack *t, const int layer, const float alpha) const;
  GPUd() void  SwapTracklets(const int left, const int right);
  GPUd() int   PartitionTracklets(const int left, const int right);
  GPUd() void  SwapHypothesis(const int left, const int right);
  GPUd() int   PartitionHypothesis(const int left, const int right);
  GPUd() void  Quicksort(const int left, const int right, const int size, const int type = 0);
  GPUd() void  PrintSettings() const;
  bool IsInitialized() const {return fIsInitialized;}

  // settings
  GPUd() void SetMCEvent(AliMCEvent* mc)       { fMCEvent = mc;}
  GPUd() void EnableDebugOutput()              { fDebugOutput = true; }
  GPUd() void SetPtThreshold(float minPt)      { fMinPt = minPt; }
  GPUd() void SetMaxEta(float maxEta)          { fMaxEta = maxEta; }
  GPUd() void SetChi2Threshold(float chi2)     { fMaxChi2 = chi2; }
  GPUd() void SetChi2Penalty(float chi2)       { fChi2Penalty = chi2; }
  GPUd() void SetMaxMissingLayers(int ly)      { fMaxMissingLy = ly; }
  GPUd() void SetNCandidates(int n);

  GPUd() AliMCEvent * GetMCEvent()   const { return fMCEvent; }
  GPUd() bool  GetIsDebugOutputOn()  const { return fDebugOutput; }
  GPUd() float GetPtThreshold()      const { return fMinPt; }
  GPUd() float GetMaxEta()           const { return fMaxEta; }
  GPUd() float GetChi2Threshold()    const { return fMaxChi2; }
  GPUd() float GetChi2Penalty()      const { return fChi2Penalty; }
  GPUd() int   GetMaxMissingLayers() const { return fMaxMissingLy; }
  GPUd() int   GetNCandidates()      const { return fNCandidates; }

  // output
  GPUd() HLTTRDTrack *Tracks()                       const { return fTracks;}
  GPUd() int NTracks()                               const { return fNTracks;}
  GPUd() AliHLTTRDSpacePointInternal *SpacePoints()  const { return fSpacePoints; }
  GPUd() void DumpTracks();

 protected:

  void* fBaseDataPtr;                         // pointer to allocated memory block with base data
  void* fTrackletsDataPtr;                    // pointer to allocated memory block with tracklet data
  void* fTracksDataPtr;                       // pointer to allocated memory block with tracks for I/O
  float *fR;                                  // rough radial position of each TRD layer
  bool fIsInitialized;                        // flag is set upon initialization
  HLTTRDTrack *fTracks;                       // array of trd-updated tracks
  int fNCandidates;                           // max. track hypothesis per layer
  int fNTracks;                               // number of TPC tracks to be matched
  int fNEvents;                               // number of processed events
  AliHLTTRDTrackletWord *fTracklets;          // array of all tracklets, later sorted by HCId
  int fNtrackletsMax;                         // max number of tracklets
  int fMaxThreads;                            // maximum number of supported threads
  int fNTracklets;                            // total number of tracklets in event
  int *fNtrackletsInChamber;                  // number of tracklets in each chamber
  int *fTrackletIndexArray;                   // index of first tracklet for each chamber
  Hypothesis *fHypothesis;                    // array with multiple track hypothesis
  HLTTRDTrack *fCandidates;                   // array of tracks for multiple hypothesis tracking
  AliHLTTRDSpacePointInternal *fSpacePoints;  // array with tracklet coordinates in global tracking frame
  AliHLTTRDGeometry *fGeo;                    // TRD geometry
  bool fDebugOutput;                          // store debug output
  float fMinPt;                               // min pt of TPC tracks for tracking
  float fMaxEta;                              // TPC tracks with higher eta are ignored
  float fMaxChi2;                             // max chi2 for tracklets
  int fMaxMissingLy;                          // max number of missing layers per track
  float fChi2Penalty;                         // chi2 added to the track for no update
  float fZCorrCoefNRC;                        // tracklet z-position depends linearly on track dip angle
  int fNhypothesis;                           // number of track hypothesis per layer
  AliMCEvent* fMCEvent;                       //! externaly supplied optional MC event
  const AliHLTTPCGMMerger *fMerger;           // supplying parameters for AliHLTTPCGMPropagator
  AliHLTTRDTrackerDebug *fDebug;              // debug output

};

#endif
