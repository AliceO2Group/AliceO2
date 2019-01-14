#ifndef ALIGPUTRDTRACKER_H
#define ALIGPUTRDTRACKER_H
/* Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include "AliTPCCommonDef.h"
#include "AliGPUTRDDef.h"
#include "AliGPUTPCDef.h"
#include "AliGPUTRDTrackerDebug.h"

class AliGPUTRDTrackletWord;
class AliGPUTRDGeometry;
class AliExternalTrackParam;
class AliMCEvent;
class AliGPUTPCGMMerger;

//-------------------------------------------------------------------------
class AliGPUTRDTracker {
 public:

#ifndef GPUCA_GPUCODE
  AliGPUTRDTracker();
  AliGPUTRDTracker(const AliGPUTRDTracker &tracker) CON_DELETE;
  AliGPUTRDTracker & operator=(const AliGPUTRDTracker &tracker) CON_DELETE;
  ~AliGPUTRDTracker();
#endif

  enum EGPUTRDTracker {
    kNLayers = 6,
    kNStacks = 5,
    kNSectors = 18,
    kNChambers = 540
  };

  // struct to hold the information on the space points
  struct AliGPUTRDSpacePointInternal {
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

  size_t SetPointersBase(void* base, int maxThreads = 1, bool doConstruct = false);
  size_t SetPointersTracklets(void* base);
  size_t SetPointersTracks(void *base, int nTracks);

  GPUd() bool Init(AliGPUTRDGeometry *geo = nullptr);
  GPUd() void Reset();
  GPUd() void StartLoadTracklets(const int nTrklts);
  GPUd() void LoadTracklet(const AliGPUTRDTrackletWord &tracklet, const int *labels = 0x0);
  void DoTracking(GPUTRDTrack *tracksTPC, int *tracksTPClab, int nTPCtracks, int *tracksTRDnTrklts = 0x0, int *tracksTRDlab = 0x0);
  GPUd() void DoTrackingThread(GPUTRDTrack *tracksTPC, int *tracksTPClab, int nTPCtracks, int iTrk, int threadId, int *tracksTRDnTrklts = 0x0, int *tracksTRDlab = 0x0);
  GPUd() bool CalculateSpacePoints();
  GPUd() bool FollowProlongation(GPUTRDPropagator *prop, GPUTRDTrack *t, int nTPCtracks, int threadId);
  GPUd() int GetDetectorNumber(const float zPos, const float alpha, const int layer) const;
  GPUd() bool AdjustSector(GPUTRDPropagator *prop, GPUTRDTrack *t, const int layer) const;
  GPUd() int GetSector(float alpha) const;
  GPUd() float GetAlphaOfSector(const int sec) const;
  GPUd() float GetRPhiRes(float snp) const { return (0.04f*0.04f+0.33f*0.33f*(snp-0.126f)*(snp-0.126f)); } // parametrization obtained from track-tracklet residuals
  GPUd() void RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3]);
  void CountMatches(const int trackID, std::vector<int> *matches) const;
  GPUd() void CheckTrackRefs(const int trackID, bool *findableMC) const;
  GPUd() void FindChambersInRoad(const GPUTRDTrack *t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const;
  GPUd() bool IsGeoFindable(const GPUTRDTrack *t, const int layer, const float alpha) const;
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
  GPUd() GPUTRDTrack *Tracks()                       const { return fTracks;}
  GPUd() int NTracks()                               const { return fNTracks;}
  GPUd() AliGPUTRDSpacePointInternal *SpacePoints()  const { return fSpacePoints; }
  GPUd() void DumpTracks();

 protected:

  void* fBaseDataPtr;                         // pointer to allocated memory block with base data
  void* fTrackletsDataPtr;                    // pointer to allocated memory block with tracklet data
  void* fTracksDataPtr;                       // pointer to allocated memory block with tracks for I/O
  float *fR;                                  // rough radial position of each TRD layer
  bool fIsInitialized;                        // flag is set upon initialization
  GPUTRDTrack *fTracks;                       // array of trd-updated tracks
  int fNCandidates;                           // max. track hypothesis per layer
  int fNTracks;                               // number of TPC tracks to be matched
  int fNEvents;                               // number of processed events
  AliGPUTRDTrackletWord *fTracklets;          // array of all tracklets, later sorted by HCId
  int fNtrackletsMax;                         // max number of tracklets
  int fMaxThreads;                            // maximum number of supported threads
  int fNTracklets;                            // total number of tracklets in event
  int *fNtrackletsInChamber;                  // number of tracklets in each chamber
  int *fTrackletIndexArray;                   // index of first tracklet for each chamber
  Hypothesis *fHypothesis;                    // array with multiple track hypothesis
  GPUTRDTrack *fCandidates;                   // array of tracks for multiple hypothesis tracking
  AliGPUTRDSpacePointInternal *fSpacePoints;  // array with tracklet coordinates in global tracking frame
  int *fTrackletLabels;                       // array with MC tracklet labels
  AliGPUTRDGeometry *fGeo;                    // TRD geometry
  bool fDebugOutput;                          // store debug output
  float fMinPt;                               // min pt of TPC tracks for tracking
  float fMaxEta;                              // TPC tracks with higher eta are ignored
  float fMaxChi2;                             // max chi2 for tracklets
  int fMaxMissingLy;                          // max number of missing layers per track
  float fChi2Penalty;                         // chi2 added to the track for no update
  float fZCorrCoefNRC;                        // tracklet z-position depends linearly on track dip angle
  int fNhypothesis;                           // number of track hypothesis per layer
  AliMCEvent* fMCEvent;                       //! externaly supplied optional MC event
  const AliGPUTPCGMMerger *fMerger;           // supplying parameters for AliGPUTPCGMPropagator
  AliGPUTRDTrackerDebug *fDebug;              // debug output

};

#endif
