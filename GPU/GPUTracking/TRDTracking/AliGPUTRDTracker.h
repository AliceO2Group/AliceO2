#ifndef ALIGPUTRDTRACKER_H
#define ALIGPUTRDTRACKER_H
/* Copyright(c) 2007-2009, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#include "AliGPUCommonDef.h"
#include "AliGPUProcessor.h"
#include "AliGPUTRDDef.h"
#include "AliGPUTPCDef.h"
#include "AliGPUTRDTrackerDebug.h"
#include "AliGPUTRDTrack.h"

#ifndef __OPENCL__
#include <vector>
#endif

#ifdef GPUCA_ALIROOT_LIB
#define TRD_GEOMETRY_CONST
#else
#define TRD_GEOMETRY_CONST const
#endif

class AliGPUTRDTrackletWord;
class AliGPUTRDGeometry;
class AliExternalTrackParam;
class AliMCEvent;
class AliGPUTPCGMMerger;
class AliGPUChainTracking;

//-------------------------------------------------------------------------
class AliGPUTRDTracker : public AliGPUProcessor {
 public:

#ifndef GPUCA_GPUCODE
  AliGPUTRDTracker();
  AliGPUTRDTracker(const AliGPUTRDTracker &tracker) CON_DELETE;
  AliGPUTRDTracker & operator=(const AliGPUTRDTracker &tracker) CON_DELETE;
  ~AliGPUTRDTracker();
  
  void SetMaxData();
  void RegisterMemoryAllocation();
  void InitializeProcessor();
  void* SetPointersBase(void* base);
  void* SetPointersTracklets(void* base);
  void* SetPointersTracks(void *base);

  bool Init(TRD_GEOMETRY_CONST AliGPUTRDGeometry *geo = nullptr);
  void CountMatches(const int trackID, std::vector<int> *matches) const;
  void DoTracking();
  void SetNCandidates(int n);
  void PrintSettings() const;
  bool IsInitialized() const {return fIsInitialized;}
  void SetTrackingChain(AliGPUChainTracking* c) {fChainTracking = c;}
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

  short MemoryPermanent() const { return fMemoryPermanent; }
  short MemoryTracklets() const { return fMemoryTracklets; }
  short MemoryTracks()    const { return fMemoryTracks; }

  GPUhd() void SetGeometry(TRD_GEOMETRY_CONST AliGPUTRDGeometry* geo) {fGeo = geo;}
  void Reset();
  GPUd() int LoadTracklet(const AliGPUTRDTrackletWord &tracklet, const int *labels = 0x0);
  template<class T> GPUd() int LoadTrack(const T &trk, const int label = -1) {
    if (fNTracks >= fNMaxTracks) {
      printf("Error: Track dropped (no memory available) -> must not happen\n");
      return(1);
    }
#ifdef GPUCA_ALIROOT_LIB
    new (&fTracks[fNTracks++]) GPUTRDTrack(trk);
#else
    fTracks[fNTracks++] = trk;
#endif
    if (label >= 0) {
        fTracks[fNTracks-1].SetLabel(label);
    }
    return(0);
  }
  GPUd() void DoTrackingThread(int iTrk, const AliGPUTPCGMMerger* merger, int threadId = 0);
  GPUd() bool CalculateSpacePoints();
  GPUd() bool FollowProlongation(GPUTRDPropagator *prop, GPUTRDTrack *t, int threadId);
  GPUd() int GetDetectorNumber(const float zPos, const float alpha, const int layer) const;
  GPUd() bool AdjustSector(GPUTRDPropagator *prop, GPUTRDTrack *t, const int layer) const;
  GPUd() int GetSector(float alpha) const;
  GPUd() float GetAlphaOfSector(const int sec) const;
  GPUd() float GetRPhiRes(float snp) const { return (0.04f*0.04f+0.33f*0.33f*(snp-0.126f)*(snp-0.126f)); } // parametrization obtained from track-tracklet residuals
  GPUd() void RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3]);
  GPUd() void CheckTrackRefs(const int trackID, bool *findableMC) const;
  GPUd() void FindChambersInRoad(const GPUTRDTrack *t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const;
  GPUd() bool IsGeoFindable(const GPUTRDTrack *t, const int layer, const float alpha) const;
  GPUd() void  SwapTracklets(const int left, const int right);
  GPUd() int   PartitionTracklets(const int left, const int right);
  GPUd() void  SwapHypothesis(const int left, const int right);
  GPUd() int   PartitionHypothesis(const int left, const int right);
  GPUd() void  Quicksort(const int left, const int right, const int size, const int type = 0);

  // settings
  GPUd() void SetMCEvent(AliMCEvent* mc)       { fMCEvent = mc;}
  GPUd() void EnableDebugOutput()              { fDebugOutput = true; }
  GPUd() void SetPtThreshold(float minPt)      { fMinPt = minPt; }
  GPUd() void SetMaxEta(float maxEta)          { fMaxEta = maxEta; }
  GPUd() void SetChi2Threshold(float chi2)     { fMaxChi2 = chi2; }
  GPUd() void SetChi2Penalty(float chi2)       { fChi2Penalty = chi2; }
  GPUd() void SetMaxMissingLayers(int ly)      { fMaxMissingLy = ly; }

  GPUd() AliMCEvent * GetMCEvent()   const { return fMCEvent; }
  GPUd() bool  GetIsDebugOutputOn()  const { return fDebugOutput; }
  GPUd() float GetPtThreshold()      const { return fMinPt; }
  GPUd() float GetMaxEta()           const { return fMaxEta; }
  GPUd() float GetChi2Threshold()    const { return fMaxChi2; }
  GPUd() float GetChi2Penalty()      const { return fChi2Penalty; }
  GPUd() int   GetMaxMissingLayers() const { return fMaxMissingLy; }
  GPUd() int   GetNCandidates()      const { return fNCandidates; }

  // output
  GPUd() int NTracks()                               const { return fNTracks;}
  GPUd() GPUTRDTrack *Tracks()                       const { return fTracks;}
  GPUd() int NTracklets()                            const { return fNTracklets;}
  GPUd() AliGPUTRDSpacePointInternal *SpacePoints()  const { return fSpacePoints; }
  GPUd() AliGPUTRDTrackletWord *Tracklets()          const { return fTracklets; }
  GPUd() void DumpTracks();

 protected:

  float *fR;                                  // rough radial position of each TRD layer
  bool fIsInitialized;                        // flag is set upon initialization
  short fMemoryPermanent;                     // size of permanent memory for the tracker
  short fMemoryTracklets;                     // size of memory for TRD tracklets
  short fMemoryTracks;                        // size of memory for tracks (used for i/o)
  int fNMaxTracks;                            // max number of tracks the tracker can handle (per event)
  int fNMaxSpacePoints;                       // max number of space points hold by the tracker (per event)
  GPUTRDTrack *fTracks;                       // array of trd-updated tracks
  int fNCandidates;                           // max. track hypothesis per layer
  int fNTracks;                               // number of TPC tracks to be matched
  int fNEvents;                               // number of processed events
  AliGPUTRDTrackletWord *fTracklets;          // array of all tracklets, later sorted by HCId
  int fMaxThreads;                            // maximum number of supported threads
  int fNTracklets;                            // total number of tracklets in event
  int *fNtrackletsInChamber;                  // number of tracklets in each chamber
  int *fTrackletIndexArray;                   // index of first tracklet for each chamber
  Hypothesis *fHypothesis;                    // array with multiple track hypothesis
  GPUTRDTrack *fCandidates;                   // array of tracks for multiple hypothesis tracking
  AliGPUTRDSpacePointInternal *fSpacePoints;  // array with tracklet coordinates in global tracking frame
  int *fTrackletLabels;                       // array with MC tracklet labels
  TRD_GEOMETRY_CONST AliGPUTRDGeometry *fGeo; // TRD geometry
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
  AliGPUChainTracking* fChainTracking;        // Tracking chain with access to input data / parameters
};

#endif
