#ifndef GPUTRDTRACK_H
#define GPUTRDTRACK_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
* See cxx source for full Copyright notice                               */

#include "GPUTRDDef.h"
#include "GPUTPCDef.h"

struct GPUTRDTrackDataRecord;
class AliHLTExternalTrackParam;

//_____________________________________________________________________________
#if defined(__CINT__) || defined(__ROOTCINT__)
namespace o2
{
namespace gpu
{
template <typename T>
class GPUTRDTrack_t;
}
} // namespace o2
#else
#include "GPUTRDInterfaces.h"

namespace o2
{
namespace gpu
{

template <typename T>
class GPUTRDTrack_t : public T
{
 public:
  enum EGPUTRDTrack { kNLayers = 6 };

  GPUd() GPUTRDTrack_t();
  GPUTRDTrack_t(const typename T::baseClass& t) = delete;
  GPUd() GPUTRDTrack_t(const GPUTRDTrack_t& t);
  GPUd() GPUTRDTrack_t(const AliHLTExternalTrackParam& t);
  GPUd() GPUTRDTrack_t(const T& t);
  GPUd() GPUTRDTrack_t& operator=(const GPUTRDTrack_t& t);

  GPUd() int GetNlayers() const;
  GPUd() int GetTracklet(int iLayer) const;
  GPUd() int GetTPCtrackId() const { return mTPCTrackId; }
  GPUd() int GetNtracklets() const { return mNTracklets; }
  GPUd() int GetNtrackletsOffline(int type) const { return mNTrackletsOffline[type]; }
  GPUd() int GetLabelOffline() const { return mLabelOffline; }
  GPUd() int GetLabel() const { return mLabel; }
  GPUd() float GetChi2() const { return mChi2; }
  GPUd() float GetReducedChi2() const { return GetNlayers() == 0 ? mChi2 : mChi2 / GetNlayers(); }
  GPUd() float GetMass() const { return mMass; }
  GPUd() bool GetIsStopped() const { return mIsStopped; }
  GPUd() bool GetIsFindable(int iLayer) const { return mIsFindable[iLayer]; }
  GPUd() int GetTrackletIndex(int iLayer) const { return GetTracklet(iLayer); }
  GPUd() int GetNmissingConsecLayers(int iLayer) const;

  GPUd() void AddTracklet(int iLayer, int idx)
  {
    mAttachedTracklets[iLayer] = idx;
    mNTracklets++;
  }
  GPUd() void SetTPCtrackId(int v) { mTPCTrackId = v; }
  GPUd() void SetNtracklets(int nTrklts) { mNTracklets = nTrklts; }
  GPUd() void SetIsFindable(int iLayer) { mIsFindable[iLayer] = true; }
  GPUd() void SetNtrackletsOffline(int type, int nTrklts) { mNTrackletsOffline[type] = nTrklts; }
  GPUd() void SetLabelOffline(int lab) { mLabelOffline = lab; }
  GPUd() void SetIsStopped() { mIsStopped = true; }

  GPUd() void SetChi2(float chi2) { mChi2 = chi2; }
  GPUd() void SetMass(float mass) { mMass = mass; }
  GPUd() void SetLabel(int label) { mLabel = label; }

  // conversion to / from HLT track structure

  GPUd() void ConvertTo(GPUTRDTrackDataRecord& t) const;
  GPUd() void ConvertFrom(const GPUTRDTrackDataRecord& t);

 protected:
  float mChi2;                      // total chi2
  float mMass;                      // mass hypothesis
  int mLabel;                       // MC label
  int mTPCTrackId;                  // corresponding TPC track
  int mNTracklets;                  // number of attached TRD tracklets
  int mNMissingConsecLayers;        // number of missing consecutive layers
  int mNTrackletsOffline[4];        // for debugging: attached offline TRD tracklets (0: total, 1: match, 2: related, 3: fake)
  int mLabelOffline;                // offline TRD MC label of this track
  int mAttachedTracklets[kNLayers]; // IDs for attached tracklets sorted by layer
  bool mIsFindable[kNLayers];       // number of layers where tracklet should exist
  bool mIsStopped;                  // track ends in TRD
};
} // namespace gpu
} // namespace o2

#endif
#endif
