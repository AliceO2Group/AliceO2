// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTracker.h
/// \brief Online TRD tracker based on extrapolated TPC tracks

/// \author Ole Schmidt

#ifndef GPUTRDTRACKER_H
#define GPUTRDTRACKER_H

#define POSITIVE_B_FIELD_5KG

#include "GPUCommonDef.h"
#include "GPUProcessor.h"
#include "GPUTRDDef.h"
#include "GPUDef.h"
#include "GPUTRDTrackerDebug.h"
#include "GPUTRDTrack.h"
#include "GPULogging.h"

#ifndef __OPENCL__
#include <vector>
#endif

class AliExternalTrackParam;
class AliMCEvent;

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#ifdef GPUCA_ALIROOT_LIB
#define TRD_GEOMETRY_CONST
#else
#define TRD_GEOMETRY_CONST const
#endif

class GPUTRDTrackletWord;
class GPUTRDGeometry;
class GPUChainTracking;

//-------------------------------------------------------------------------
class GPUTRDTracker : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE
  GPUTRDTracker();
  GPUTRDTracker(const GPUTRDTracker& tracker) CON_DELETE;
  GPUTRDTracker& operator=(const GPUTRDTracker& tracker) CON_DELETE;
  ~GPUTRDTracker();

  void SetMaxData(const GPUTrackingInOutPointers& io);
  void RegisterMemoryAllocation();
  void InitializeProcessor();
  void* SetPointersBase(void* base);
  void* SetPointersTracklets(void* base);
  void* SetPointersTracks(void* base);

  bool Init(TRD_GEOMETRY_CONST GPUTRDGeometry* geo = nullptr);
  void CountMatches(const int trackID, std::vector<int>* matches) const;
  void DoTracking();
  void SetNCandidates(int n);
  void PrintSettings() const;
  bool IsInitialized() const { return mIsInitialized; }
  void SetTrackingChain(GPUChainTracking* c) { mChainTracking = c; }
  void StartDebugging() { mDebug->CreateStreamer(); }
#endif

  enum EGPUTRDTracker { kNLayers = 6,
                        kNStacks = 5,
                        kNSectors = 18,
                        kNChambers = 540 };

  // struct to hold the information on the space points
  struct GPUTRDSpacePointInternal {
    float mR;                 // x position (3.5 mm above anode wires) - radial offset due to t0 mis-calibration, measured -1 mm for run 245353
    float mX[2];              // y and z position (sector coordinates)
    My_Float mCov[3];         // sigma_y^2, sigma_yz, sigma_z^2
    float mDy;                // deflection over drift length
    int mId;                  // index
    int mLabel[3];            // MC labels
    unsigned short mVolumeId; // basically derived from TRD chamber number
  };

  struct Hypothesis {
    int mLayers;      // number of layers with TRD space point
    int mCandidateId; // to which track candidate the hypothesis belongs
    int mTrackletId;  // tracklet index to be used for update
    float mChi2;      // predicted chi2 for given space point
    float mChi2YZPhi; // not yet ready (see GetPredictedChi2 method in cxx file)

    GPUd() float GetReducedChi2() { return mLayers > 0 ? mChi2 / mLayers : mChi2; }
    GPUd() Hypothesis() : mLayers(0), mCandidateId(-1), mTrackletId(-1), mChi2(9999.f) {}
    GPUd() Hypothesis(int layers, int candidateId, int trackletId, float chi2, float chi2YZPhi = -1.f) : mLayers(layers), mCandidateId(candidateId), mTrackletId(trackletId), mChi2(chi2), mChi2YZPhi(chi2YZPhi) {}
  };

  short MemoryPermanent() const { return mMemoryPermanent; }
  short MemoryTracklets() const { return mMemoryTracklets; }
  short MemoryTracks() const { return mMemoryTracks; }

  GPUhd() void SetGeometry(TRD_GEOMETRY_CONST GPUTRDGeometry* geo) { mGeo = geo; }
  void Reset(bool fast = false);
  GPUd() int LoadTracklet(const GPUTRDTrackletWord& tracklet, const int* labels = nullptr);
  template <class T>
  GPUd() int LoadTrack(const T& trk, const int label = -1, const int* nTrkltsOffline = nullptr, const int labelOffline = -1)
  {
    if (mNTracks >= mNMaxTracks) {
#ifndef GPUCA_GPUCODE
      GPUError("Error: Track dropped (no memory available) -> must not happen");
#endif
      return (1);
    }
    if (!trk.CheckNumericalQuality()) {
      return (0);
    }
    if (CAMath::Abs(trk.getEta()) > mMaxEta) {
      return (0);
    }
    if (trk.getPt() < mMinPt) {
      return (0);
    }
#ifdef GPUCA_ALIROOT_LIB
    new (&mTracks[mNTracks]) GPUTRDTrack(trk); // We need placement new, since the class is virtual
#else
    mTracks[mNTracks] = trk;
#endif
    mTracks[mNTracks].SetTPCtrackId(mNTracks);
    if (label >= 0) {
      mTracks[mNTracks].SetLabel(label);
    }
    if (nTrkltsOffline) {
      for (int i = 0; i < 4; ++i) {
        mTracks[mNTracks].SetNtrackletsOffline(i, nTrkltsOffline[i]); // see GPUTRDTrack.h for information on the index
      }
    }
    mTracks[mNTracks].SetLabelOffline(labelOffline);
    mNTracks++;
    return (0);
  }
  GPUd() void DoTrackingThread(int iTrk, int threadId = 0);
  GPUd() bool CalculateSpacePoints();
  GPUd() bool FollowProlongation(GPUTRDPropagator* prop, GPUTRDTrack* t, int threadId);
  GPUd() float GetPredictedChi2(const My_Float* pTRD, const My_Float* covTRD, const My_Float* pTrk, const My_Float* covTrk) const;
  GPUd() int GetDetectorNumber(const float zPos, const float alpha, const int layer) const;
  GPUd() bool AdjustSector(GPUTRDPropagator* prop, GPUTRDTrack* t, const int layer) const;
  GPUd() int GetSector(float alpha) const;
  GPUd() float GetAlphaOfSector(const int sec) const;
  // TODO all parametrizations depend on B-field -> need to find correct description.. To be put in CCDB in the future?
#ifdef POSITIVE_B_FIELD_5KG
  // B = +5kG for Run 244340 and 245353
  GPUd() float GetRPhiRes(float snp) const { return (0.04f * 0.04f + 0.31f * 0.31f * (snp - 0.125f) * (snp - 0.125f)); } // parametrization obtained from track-tracklet residuals
  GPUd() float GetAngularResolution(float snp) const { return 0.041f * 0.041f + 0.43f * 0.43f * (snp - 0.15f) * (snp - 0.15f); }
  GPUd() float ConvertAngleToDy(float snp) const { return 0.13f + 2.43f * snp - 0.58f * snp * snp; } // more accurate than sin(phi) = (dy / xDrift) / sqrt(1+(dy/xDrift)^2)
#else
  // B = -5kG for Run 246390
  GPUd() float GetRPhiRes(float snp) const { return (0.04f * 0.04f + 0.34f * 0.34f * (snp + 0.14f) * (snp + 0.14f)); }
  GPUd() float GetAngularResolution(float snp) const { return 0.047f * 0.047f + 0.45f * 0.45f * (snp + 0.15f) * (snp + 0.15f); }
  GPUd() float ConvertAngleToDy(float snp) const { return -0.15f + 2.34f * snp + 0.56f * snp * snp; } // more accurate than sin(phi) = (dy / xDrift) / sqrt(1+(dy/xDrift)^2)
#endif
  GPUd() float GetAngularPull(float dYtracklet, float snp) const;
  GPUd() void RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3]);
  GPUd() void CheckTrackRefs(const int trackID, bool* findableMC) const;
  GPUd() void FindChambersInRoad(const GPUTRDTrack* t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const;
  GPUd() bool IsGeoFindable(const GPUTRDTrack* t, const int layer, const float alpha) const;
  GPUd() void SwapTracklets(const int left, const int right);
  GPUd() int PartitionTracklets(const int left, const int right);
  GPUd() void Quicksort(const int left, const int right, const int size);
  GPUd() void InsertHypothesis(Hypothesis hypo, int& nCurrHypothesis, int idxOffset);

  // settings
  GPUd() void SetMCEvent(AliMCEvent* mc) { mMCEvent = mc; }
  GPUd() void EnableDebugOutput() { mDebugOutput = true; }
  GPUd() void SetPtThreshold(float minPt) { mMinPt = minPt; }
  GPUd() void SetMaxEta(float maxEta) { mMaxEta = maxEta; }
  GPUd() void SetChi2Threshold(float chi2) { mMaxChi2 = chi2; }
  GPUd() void SetChi2Penalty(float chi2) { mChi2Penalty = chi2; }
  GPUd() void SetMaxMissingLayers(int ly) { mMaxMissingLy = ly; }
  GPUd() void SetExtraRoadY(float extraRoadY) { mExtraRoadY = extraRoadY; }
  GPUd() void SetRoadZ(float roadZ) { mRoadZ = roadZ; }

  GPUd() AliMCEvent* GetMCEvent() const { return mMCEvent; }
  GPUd() bool GetIsDebugOutputOn() const { return mDebugOutput; }
  GPUd() float GetPtThreshold() const { return mMinPt; }
  GPUd() float GetMaxEta() const { return mMaxEta; }
  GPUd() float GetChi2Threshold() const { return mMaxChi2; }
  GPUd() float GetChi2Penalty() const { return mChi2Penalty; }
  GPUd() int GetMaxMissingLayers() const { return mMaxMissingLy; }
  GPUd() int GetNCandidates() const { return mNCandidates; }
  GPUd() float GetExtraRoadY() const { return mExtraRoadY; }
  GPUd() float GetRoadZ() const { return mRoadZ; }

  // output
  GPUd() int NTracks() const { return mNTracks; }
  GPUd() GPUTRDTrack* Tracks() const { return mTracks; }
  GPUd() int NTracklets() const { return mNTracklets; }
  GPUd() GPUTRDSpacePointInternal* SpacePoints() const { return mSpacePoints; }
  GPUd() GPUTRDTrackletWord* Tracklets() const { return mTracklets; }
  GPUd() void DumpTracks();

 protected:
  float* mR;                               // radial position of each TRD chamber, alignment taken into account, radial spread within chambers < 7mm
  bool mIsInitialized;                     // flag is set upon initialization
  short mMemoryPermanent;                  // size of permanent memory for the tracker
  short mMemoryTracklets;                  // size of memory for TRD tracklets
  short mMemoryTracks;                     // size of memory for tracks (used for i/o)
  int mNMaxTracks;                         // max number of tracks the tracker can handle (per event)
  int mNMaxSpacePoints;                    // max number of space points hold by the tracker (per event)
  GPUTRDTrack* mTracks;                    // array of trd-updated tracks
  int mNCandidates;                        // max. track hypothesis per layer
  int mNTracks;                            // number of TPC tracks to be matched
  int mNEvents;                            // number of processed events
  GPUTRDTrackletWord* mTracklets;          // array of all tracklets, later sorted by HCId
  int mMaxThreads;                         // maximum number of supported threads
  int mNTracklets;                         // total number of tracklets in event
  int* mNTrackletsInChamber;               // number of tracklets in each chamber
  int* mTrackletIndexArray;                // index of first tracklet for each chamber
  Hypothesis* mHypothesis;                 // array with multiple track hypothesis
  GPUTRDTrack* mCandidates;                // array of tracks for multiple hypothesis tracking
  GPUTRDSpacePointInternal* mSpacePoints;  // array with tracklet coordinates in global tracking frame
  int* mTrackletLabels;                    // array with MC tracklet labels
  TRD_GEOMETRY_CONST GPUTRDGeometry* mGeo; // TRD geometry
  bool mDebugOutput;                       // store debug output
  float mRadialOffset;                     // due to mis-calibration of t0
  float mMinPt;                            // min pt of TPC tracks for tracking
  float mMaxEta;                           // TPC tracks with higher eta are ignored
  float mExtraRoadY;                       // addition to search road in r-phi to account for not exact radial match of tracklets and tracks in first iteration
  float mRoadZ;                            // in z, a constant search road is used
  float mMaxChi2;                          // max chi2 for tracklets
  int mMaxMissingLy;                       // max number of missing layers per track
  float mChi2Penalty;                      // chi2 added to the track for no update
  float mZCorrCoefNRC;                     // tracklet z-position depends linearly on track dip angle
  AliMCEvent* mMCEvent;                    //! externaly supplied optional MC event
  GPUTRDTrackerDebug* mDebug;              // debug output
  GPUChainTracking* mChainTracking;        // Tracking chain with access to input data / parameters
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDTRACKER_H
