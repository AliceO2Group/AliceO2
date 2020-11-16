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


#include "GPUCommonDef.h"
#include "GPUProcessor.h"
#include "GPUTRDDef.h"
#include "GPUDef.h"
#include "GPUTRDTrack.h"
#include "GPULogging.h"

#ifndef GPUCA_GPUCODE_DEVICE
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

  void CountMatches(const int trackID, std::vector<int>* matches) const;
  void DoTracking(GPUChainTracking* chainTracking);
  void SetNCandidates(int n);
  void PrintSettings() const;
  bool IsInitialized() const { return mIsInitialized; }
  void StartDebugging();
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

    GPUd() float GetReducedChi2() { return mLayers > 0 ? mChi2 / mLayers : mChi2; }
    GPUd() Hypothesis() : mLayers(0), mCandidateId(-1), mTrackletId(-1), mChi2(9999.f) {}
    GPUd() Hypothesis(int layers, int candidateId, int trackletId, float chi2, float chi2YZPhi = -1.f) : mLayers(layers), mCandidateId(candidateId), mTrackletId(trackletId), mChi2(chi2) {}
  };

  short MemoryPermanent() const { return mMemoryPermanent; }
  short MemoryTracklets() const { return mMemoryTracklets; }
  short MemoryTracks() const { return mMemoryTracks; }

  GPUhd() void OverrideGPUGeometry(TRD_GEOMETRY_CONST GPUTRDGeometry* geo) { mGeo = geo; }
  void Reset();
  GPUd() int LoadTracklet(const GPUTRDTrackletWord& tracklet, const int* labels = nullptr);
  template <class T>
  GPUd() bool PreCheckTrackTRDCandidate(const T& trk) const
  {
    return true;
  }
  GPUd() bool PreCheckTrackTRDCandidate(const GPUTPCGMMergedTrack& trk) const { return trk.OK() && !trk.Looper(); }
  GPUd() bool CheckTrackTRDCandidate(const TRDTRK& trk) const;
  GPUd() int LoadTrack(const TRDTRK& trk, const int label = -1, const int* nTrkltsOffline = nullptr, const int labelOffline = -1, int tpcTrackId = -1, bool checkTrack = true);

  GPUd() int GetCollisionID(float trkTime) const;
  GPUd() void DoTrackingThread(int iTrk, int threadId = 0);
  GPUd() bool CalculateSpacePoints(int iCollision = 0);
  GPUd() bool FollowProlongation(PROP* prop, TRDTRK* t, int threadId, int collisionId);
  GPUd() int GetDetectorNumber(const float zPos, const float alpha, const int layer) const;
  GPUd() bool AdjustSector(PROP* prop, TRDTRK* t) const;
  GPUd() int GetSector(float alpha) const;
  GPUd() float GetAlphaOfSector(const int sec) const;
  GPUd() float GetRPhiRes(float snp) const { return (mRPhiA2 + mRPhiC2 * (snp - mRPhiB) * (snp - mRPhiB)); }           // parametrization obtained from track-tracklet residuals:
  GPUd() float GetAngularResolution(float snp) const { return mDyA2 + mDyC2 * (snp - mDyB) * (snp - mDyB); }           // a^2 + c^2 * (snp - b)^2
  GPUd() float ConvertAngleToDy(float snp) const { return mAngleToDyA + mAngleToDyB * snp + mAngleToDyC * snp * snp; } // a + b*snp + c*snp^2 is more accurate than sin(phi) = (dy / xDrift) / sqrt(1+(dy/xDrift)^2)
  GPUd() float GetAngularPull(float dYtracklet, float snp) const;
  GPUd() void RecalcTrkltCov(const float tilt, const float snp, const float rowSize, My_Float (&cov)[3]);
  GPUd() void CheckTrackRefs(const int trackID, bool* findableMC) const;
  GPUd() void FindChambersInRoad(const TRDTRK* t, const float roadY, const float roadZ, const int iLayer, int* det, const float zMax, const float alpha) const;
  GPUd() bool IsGeoFindable(const TRDTRK* t, const int layer, const float alpha) const;
  GPUd() void SwapTracklets(const int left, const int right);
  GPUd() int PartitionTracklets(const int left, const int right);
  GPUd() void Quicksort(const int left, const int right, const int size);
  GPUd() void InsertHypothesis(Hypothesis hypo, int& nCurrHypothesis, int idxOffset);

  // input from TRD trigger record
  GPUd() void SetNMaxCollisions(int nColl) { mNMaxCollisions = nColl; } // can this be fixed to a sufficiently large value?
  GPUd() void SetNCollisions(int nColl);
  GPUd() void SetTriggerRecordIndices(int* indices) { mTriggerRecordIndices = indices; }
  GPUd() void SetTriggerRecordTimes(float* times) { mTriggerRecordTimes = times; }

  // settings
  GPUd() void SetProcessPerTimeFrame() { mProcessPerTimeFrame = true; }
  GPUd() void SetMCEvent(AliMCEvent* mc) { mMCEvent = mc; }
  GPUd() void EnableDebugOutput() { mDebugOutput = true; }
  GPUd() void SetMaxEta(float maxEta) { mMaxEta = maxEta; }
  GPUd() void SetExtraRoadY(float extraRoadY) { mExtraRoadY = extraRoadY; }
  GPUd() void SetRoadZ(float roadZ) { mRoadZ = roadZ; }

  GPUd() AliMCEvent* GetMCEvent() const { return mMCEvent; }
  GPUd() bool GetIsDebugOutputOn() const { return mDebugOutput; }
  GPUd() float GetMaxEta() const { return mMaxEta; }
  GPUd() int GetNCandidates() const { return mNCandidates; }
  GPUd() float GetExtraRoadY() const { return mExtraRoadY; }
  GPUd() float GetRoadZ() const { return mRoadZ; }

  // output
  GPUd() int NTracks() const { return mNTracks; }
  GPUd() TRDTRK* Tracks() const { return mTracks; }
  GPUd() int NTracklets() const { return mNTracklets; }
  GPUd() GPUTRDSpacePointInternal* SpacePoints() const { return mSpacePoints; }
  GPUd() GPUTRDTrackletWord* Tracklets() const { return mTracklets; }
  GPUd() void DumpTracks();

 protected:
  float* mR;                               // radial position of each TRD chamber, alignment taken into account, radial spread within chambers < 7mm
  bool mIsInitialized;                     // flag is set upon initialization
  bool mProcessPerTimeFrame;               // if true, tracking is done per time frame instead of on a single events basis //FIXME is this needed??
  short mMemoryPermanent;                  // size of permanent memory for the tracker
  short mMemoryTracklets;                  // size of memory for TRD tracklets
  short mMemoryTracks;                     // size of memory for tracks (used for i/o)
  int mNMaxCollisions;                     // max number of collisions to process (per time frame)
  int mNMaxTracks;                         // max number of tracks the tracker can handle (per event)
  int mNMaxSpacePoints;                    // max number of space points hold by the tracker (per event)
  TRDTRK* mTracks;                         // array of trd-updated tracks
  int mNCandidates;                        // max. track hypothesis per layer
  int mNCollisions;                        // number of collisions with TRD tracklet data
  int mNTracks;                            // number of TPC tracks to be matched
  int mNEvents;                            // number of processed events
  int* mTriggerRecordIndices;              // index of first tracklet for each collision
  float* mTriggerRecordTimes;              // time in us for each collision
  GPUTRDTrackletWord* mTracklets;          // array of all tracklets, later sorted by HCId
  int mMaxThreads;                         // maximum number of supported threads
  int mNTracklets;                         // total number of tracklets in event
  int* mTrackletIndexArray;                // index of first tracklet for each chamber, last entry is the total amount of tracklets
  Hypothesis* mHypothesis;                 // array with multiple track hypothesis
  TRDTRK* mCandidates;                     // array of tracks for multiple hypothesis tracking
  GPUTRDSpacePointInternal* mSpacePoints;  // array with tracklet coordinates in global tracking frame
  int* mTrackletLabels;                    // array with MC tracklet labels
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
  bool mDebugOutput;                       // store debug output
  float mTimeWindow;                       // max. deviation of the ITS-TPC track time w.r.t. TRD trigger record time stamp (in us, default is 100 ns)
  float mRadialOffset;                     // due to mis-calibration of t0
  float mMaxEta;                           // TPC tracks with higher eta are ignored
  float mExtraRoadY;                       // addition to search road in r-phi to account for not exact radial match of tracklets and tracks in first iteration
  float mRoadZ;                            // in z, a constant search road is used
  float mZCorrCoefNRC;                     // tracklet z-position depends linearly on track dip angle
  AliMCEvent* mMCEvent;                    //! externaly supplied optional MC event
  GPUTRDTrackerDebug<TRDTRK>* mDebug;      // debug output
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTRDTRACKER_H
