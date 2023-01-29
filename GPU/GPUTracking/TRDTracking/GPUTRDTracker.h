// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDTracker.h
/// \brief Online TRD tracker based on extrapolated TPC tracks

/// \author Ole Schmidt

#ifndef GPUTRDTRACKER_H
#define GPUTRDTRACKER_H

#include "GPUCommonDef.h"
#include "GPUProcessor.h"
#include "GPUTRDDef.h"
#include "GPUDef.h"
#include "GPUTRDTrack.h"
#include "GPUTRDSpacePoint.h"
#include "GPULogging.h"
#include "GPUTRDInterfaces.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include <vector>
#endif

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
template <class T>
class GPUTRDTrackerDebug;

//-------------------------------------------------------------------------
template <class TRDTRK, class PROP>
class GPUTRDTracker_t : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE
  GPUTRDTracker_t();
  GPUTRDTracker_t(const GPUTRDTracker_t& tracker) CON_DELETE;
  GPUTRDTracker_t& operator=(const GPUTRDTracker_t& tracker) CON_DELETE;
  ~GPUTRDTracker_t();

  void SetMaxData(const GPUTrackingInOutPointers& io);
  void RegisterMemoryAllocation();
  void InitializeProcessor();
  void* SetPointersBase(void* base);
  void* SetPointersTracklets(void* base);
  void* SetPointersTracks(void* base);

  void PrepareTracking(GPUChainTracking* chainTracking);
  void SetNCandidates(int n);
  void PrintSettings() const;
  bool IsInitialized() const { return mIsInitialized; }
  void StartDebugging();
#endif

  enum EGPUTRDTracker { kNLayers = 6,
                        kNStacks = 5,
                        kNSectors = 18,
                        kNChambers = 540 };

  struct HelperTrackAttributes {
    // additional TRD track attributes which are transient
    float mTime;       // time estimate for seeding track in us
    float mTimeAddMax; // max. time that can be added to this track seed in us
    float mTimeSubMax; // max. time that can be subtracted to this track seed in us
    short mSide;       // -1 : A-side, +1 : C-side (relevant only for TPC-only tracks)
    GPUd() float GetTimeMin() const { return mTime - mTimeSubMax; }
    GPUd() float GetTimeMax() const { return mTime + mTimeAddMax; }
    GPUd() HelperTrackAttributes() : mTime(-1.f), mTimeAddMax(0.f), mTimeSubMax(0.f), mSide(0) {}
  };

  struct Hypothesis {
    int mLayers;      // number of layers with TRD space point
    int mCandidateId; // to which track candidate the hypothesis belongs
    int mTrackletId;  // tracklet index to be used for update (global index within tracklet array)
    float mChi2;      // predicted chi2 for given space point

    GPUd() float GetReducedChi2() { return mLayers > 0 ? mChi2 / mLayers : mChi2; }
    GPUd() Hypothesis() : mLayers(0), mCandidateId(-1), mTrackletId(-1), mChi2(9999.f) {}
    GPUd() Hypothesis(int layers, int candidateId, int trackletId, float chi2) : mLayers(layers), mCandidateId(candidateId), mTrackletId(trackletId), mChi2(chi2) {}
  };

  short MemoryPermanent() const { return mMemoryPermanent; }
  short MemoryTracklets() const { return mMemoryTracklets; }
  short MemoryTracks() const { return mMemoryTracks; }

  GPUhd() void OverrideGPUGeometry(TRD_GEOMETRY_CONST GPUTRDGeometry* geo) { mGeo = geo; }
  void Reset();
  template <class T>
  GPUd() bool PreCheckTrackTRDCandidate(const T& trk) const
  {
    return true;
  }
  GPUd() bool PreCheckTrackTRDCandidate(const GPUTPCGMMergedTrack& trk) const { return trk.OK() && !trk.Looper(); }
  GPUd() bool CheckTrackTRDCandidate(const TRDTRK& trk) const;
  GPUd() int LoadTrack(const TRDTRK& trk, unsigned int tpcTrackId, bool checkTrack = true, HelperTrackAttributes* attribs = nullptr);

  GPUd() int GetCollisionIDs(int iTrk, int* collisionIds) const;
  GPUd() void DoTrackingThread(int iTrk, int threadId = 0);
  static GPUd() bool ConvertTrkltToSpacePoint(const GPUTRDGeometry& geo, GPUTRDTrackletWord& trklt, GPUTRDSpacePoint& sp);
  GPUd() bool CalculateSpacePoints(int iCollision = 0);
  GPUd() bool FollowProlongation(PROP* prop, TRDTRK* t, int iTrk, int threadId, int collisionId);
  GPUd() int GetDetectorNumber(const float zPos, const float alpha, const int layer) const;
  GPUd() bool AdjustSector(PROP* prop, TRDTRK* t) const;
  GPUd() int GetSector(float alpha) const;
  GPUd() float GetAlphaOfSector(const int sec) const;
  GPUd() float GetRPhiRes(float snp) const { return (mRPhiA2 + mRPhiC2 * (snp - mRPhiB) * (snp - mRPhiB)); }           // parametrization obtained from track-tracklet residuals:
  GPUd() float GetAngularResolution(float snp) const { return mDyA2 + mDyC2 * (snp - mDyB) * (snp - mDyB); }           // a^2 + c^2 * (snp - b)^2
  GPUd() float ConvertAngleToDy(float snp) const { return mAngleToDyA + mAngleToDyB * snp + mAngleToDyC * snp * snp; } // a + b*snp + c*snp^2 is more accurate than sin(phi) = (dy / xDrift) / sqrt(1+(dy/xDrift)^2)
  GPUd() float GetAngularPull(float dYtracklet, float snp) const;
  GPUd() void RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3]);
  GPUd() void FindChambersInRoad(const TRDTRK* t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha, const float zShiftTrk) const;
  GPUd() bool IsGeoFindable(const TRDTRK* t, const int layer, const float alpha, const float zShiftTrk) const;
  GPUd() void InsertHypothesis(Hypothesis hypo, int& nCurrHypothesis, int idxOffset);


  // settings
  GPUd() void SetGenerateSpacePoints(bool flag) { mGenerateSpacePoints = flag; }
  GPUd() bool GenerateSpacepoints() const { return mGenerateSpacePoints; }
  GPUd() void SetProcessPerTimeFrame(bool flag) { mProcessPerTimeFrame = flag; }
  GPUd() void EnableDebugOutput() { mDebugOutput = true; }
  GPUd() void SetMaxEta(float maxEta) { mMaxEta = maxEta; }
  GPUd() void SetRoadZ(float roadZ) { mRoadZ = roadZ; }
  GPUd() void SetTPCVdrift(float vDrift) { mTPCVdrift = vDrift; }
  GPUd() void SetTPCTDriftOffset(float t) { mTPCTDriftOffset = t; }

  GPUd() bool GetIsDebugOutputOn() const { return mDebugOutput; }
  GPUd() float GetMaxEta() const { return mMaxEta; }
  GPUd() int GetNCandidates() const { return mNCandidates; }
  GPUd() float GetRoadZ() const { return mRoadZ; }

  // output
  GPUd() int NTracks() const { return mNTracks; }
  GPUd() GPUTRDSpacePoint* SpacePoints() const { return mSpacePoints; }
  GPUd() TRDTRK* Tracks() const { return mTracks; }
  GPUd() void DumpTracks();

  // utility
  GPUd() const typename PROP::propagatorParam* getPropagatorParam(bool externalDefaultO2Propagator);

 protected:
  float* mR;                               // radial position of each TRD chamber, alignment taken into account, radial spread within chambers < 7mm
  bool mIsInitialized;                     // flag is set upon initialization
  bool mGenerateSpacePoints;               // if true, only tracklets are provided as input and they will be converted into space points by the tracker
  bool mProcessPerTimeFrame;               // if true, tracking is done per time frame instead of on a single events basis
  short mNAngleHistogramBins;              // number of bins per chamber for the angular difference histograms
  float mAngleHistogramRange;              // range of impact angles covered by each histogram
  short mMemoryPermanent;                  // size of permanent memory for the tracker
  short mMemoryTracklets;                  // size of memory for TRD tracklets
  short mMemoryTracks;                     // size of memory for tracks (used for i/o)
  int mNMaxCollisions;                     // max number of collisions to process (per time frame)
  int mNMaxTracks;                         // max number of tracks the tracker can handle (per event)
  int mNMaxSpacePoints;                    // max number of space points hold by the tracker (per event)
  TRDTRK* mTracks;                         // array of trd-updated tracks
  HelperTrackAttributes* mTrackAttribs;    // array with additional (transient) track attributes
  int mNCandidates;                        // max. track hypothesis per layer
  int mNTracks;                            // number of TPC tracks to be matched
  int mNEvents;                            // number of processed events
  int mMaxThreads;                         // maximum number of supported threads
  // index of first tracklet for each chamber within tracklets array, last entry is total number of tracklets for given collision
  // the array has (kNChambers + 1) * numberOfCollisions entries
  // note, that for collision iColl one has to add an offset corresponding to the index of the first tracklet of iColl to the index stored in mTrackletIndexArray
  int* mTrackletIndexArray;
  Hypothesis* mHypothesis;                 // array with multiple track hypothesis
  TRDTRK* mCandidates;                     // array of tracks for multiple hypothesis tracking
  GPUTRDSpacePoint* mSpacePoints;          // array with tracklet coordinates in global tracking frame
  TRD_GEOMETRY_CONST GPUTRDGeometry* mGeo; // TRD geometry
  /// ---- error parametrization depending on magnetic field ----
  float mRPhiA2;     // parameterization for tracklet position resolution
  float mRPhiB;      // parameterization for tracklet position resolution
  float mRPhiC2;     // parameterization for tracklet position resolution
  float mDyA2;       // parameterization for tracklet angular resolution
  float mDyB;        // parameterization for tracklet angular resolution
  float mDyC2;       // parameterization for tracklet angular resolution
  float mAngleToDyA; // parameterization for conversion track angle -> tracklet deflection
  float mAngleToDyB; // parameterization for conversion track angle -> tracklet deflection
  float mAngleToDyC; // parameterization for conversion track angle -> tracklet deflection
  /// ---- end error parametrization ----
  bool mDebugOutput;                  // store debug output
  static CONSTEXPR float sRadialOffset GPUCA_CPP11_INIT(= -0.1f); // due to (possible) mis-calibration of t0 -> will become obsolete when tracklet conversion is done outside of the tracker
  float mMaxEta;                                                  // TPC tracks with higher eta are ignored
  float mRoadZ;                       // in z, a constant search road is used
  float mZCorrCoefNRC;                // tracklet z-position depends linearly on track dip angle
  float mTPCVdrift;                   // TPC drift velocity used for shifting TPC tracks along Z
  float mTPCTDriftOffset;             // TPC drift time additive offset
  GPUTRDTrackerDebug<TRDTRK>* mDebug; // debug output
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDTRACKER_H
