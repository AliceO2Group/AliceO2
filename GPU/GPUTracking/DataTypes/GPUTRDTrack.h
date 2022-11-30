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

/// \file GPUTRDTrack.h
/// \author Ole Schmidt

#ifndef GPUTRDTRACK_H
#define GPUTRDTRACK_H

#include "GPUTRDDef.h"
#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"

struct GPUTRDTrackDataRecord;
class AliHLTExternalTrackParam;

namespace o2
{
namespace tpc
{
class TrackTPC;
} // namespace tpc
namespace dataformats
{
class TrackTPCITS;
class GlobalTrackID;
} // namespace dataformats
} // namespace o2

//_____________________________________________________________________________
#if (defined(__CINT__) || defined(__ROOTCINT__)) && !defined(__CLING__)
namespace GPUCA_NAMESPACE
{
namespace gpu
{
template <typename T>
class GPUTRDTrack_t;
} // namespace gpu
} // namespace GPUCA_NAMESPACE
#else
#if (!defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)) || defined(GPUCA_HAVE_O2HEADERS)
#include "GPUTRDInterfaceO2Track.h"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <typename T>
class GPUTRDTrack_t : public T
{
 public:
  enum EGPUTRDTrack {
    kNLayers = 6,
    kAmbiguousFlag = 6,
    kStopFlag = 7
  };

  GPUd() GPUTRDTrack_t();
  GPUTRDTrack_t(const typename T::baseClass& t) = delete;
  GPUd() GPUTRDTrack_t(const GPUTRDTrack_t& t);
  GPUd() GPUTRDTrack_t(const AliHLTExternalTrackParam& t);
  GPUd() GPUTRDTrack_t(const o2::dataformats::TrackTPCITS& t);
  GPUd() GPUTRDTrack_t(const o2::tpc::TrackTPC& t);
  GPUd() GPUTRDTrack_t(const T& t);
  GPUd() GPUTRDTrack_t& operator=(const GPUTRDTrack_t& t);

  // attach a tracklet to this track; this overwrites the mFlags flag to true for this layer
  GPUd() void addTracklet(int iLayer, int idx) { mAttachedTracklets[iLayer] = idx; }

  // getters
  GPUd() int getNlayersFindable() const;
  GPUd() int getTrackletIndex(int iLayer) const { return mAttachedTracklets[iLayer]; }
  GPUd() unsigned int getRefGlobalTrackIdRaw() const { return mRefGlobalTrackId; }
  // This method is only defined in TrackTRD.h and is intended to be used only with that TRD track type
  GPUd() o2::dataformats::GlobalTrackID getRefGlobalTrackId() const;
  GPUd() short getCollisionId() const { return mCollisionId; }
  GPUd() int getNtracklets() const
  {
    // returns number of tracklets attached to this track
    int retVal = 0;
    for (int iLy = 0; iLy < kNLayers; ++iLy) {
      if (mAttachedTracklets[iLy] >= 0) {
        ++retVal;
      }
    }
    return retVal;
  }

  GPUd() float getChi2() const { return mChi2; }
  GPUd() unsigned char getIsCrossingNeighbor() const { return mIsCrossingNeighbor; }
  GPUd() bool getIsCrossingNeighbor(int iLayer) const { return mIsCrossingNeighbor & (1 << iLayer); }
  GPUd() bool getHasNeighbor() const { return mIsCrossingNeighbor & (1 << 6); }
  GPUd() bool getHasPadrowCrossing() const { return mIsCrossingNeighbor & (1 << 7); }
  GPUd() float getReducedChi2() const { return getNlayersFindable() == 0 ? mChi2 : mChi2 / getNlayersFindable(); }
  GPUd() bool getIsStopped() const { return (mFlags >> kStopFlag) & 0x1; }
  GPUd() bool getIsAmbiguous() const { return (mFlags >> kAmbiguousFlag) & 0x1; }
  GPUd() bool getIsFindable(int iLayer) const { return (mFlags >> iLayer) & 0x1; }
  GPUd() int getNmissingConsecLayers(int iLayer) const;
  GPUd() int getIsPenaltyAdded(int iLayer) const { return getIsFindable(iLayer) && getTrackletIndex(iLayer) < 0; }
  // for AliRoot compatibility. To be removed once HLT/global/AliHLTGlobalEsdConverterComponent.cxx does not require them anymore
  GPUd() int GetTPCtrackId() const { return mRefGlobalTrackId; }
  GPUd() bool GetIsStopped() const { return getIsStopped(); }
  GPUd() int GetNtracklets() const { return getNtracklets(); }

  // setters
  GPUd() void setRefGlobalTrackIdRaw(unsigned int id) { mRefGlobalTrackId = id; }
  // This method is only defined in TrackTRD.h and is intended to be used only with that TRD track type
  GPUd() void setRefGlobalTrackId(o2::dataformats::GlobalTrackID id);
  GPUd() void setCollisionId(short id) { mCollisionId = id; }
  GPUd() void setIsFindable(int iLayer) { mFlags |= (1U << iLayer); }
  GPUd() void setIsStopped() { mFlags |= (1U << kStopFlag); }
  GPUd() void setIsAmbiguous() { mFlags |= (1U << kAmbiguousFlag); }
  GPUd() void setChi2(float chi2) { mChi2 = chi2; }
  GPUd() void setIsCrossingNeighbor(int iLayer) { mIsCrossingNeighbor |= (1U << iLayer); }
  GPUd() void setHasNeighbor() { mIsCrossingNeighbor |= (1U << 6); }
  GPUd() void setHasPadrowCrossing() { mIsCrossingNeighbor |= (1U << 7); }

  // conversion to / from HLT track structure (only for AliRoot)
  GPUd() void ConvertTo(GPUTRDTrackDataRecord& t) const;
  GPUd() void ConvertFrom(const GPUTRDTrackDataRecord& t);

 protected:
  float mChi2;                       // total chi2.
  unsigned int mRefGlobalTrackId;    // raw GlobalTrackID of the seeding track (either ITS-TPC or TPC)
  int mAttachedTracklets[kNLayers];  // indices of the tracklets attached to this track; -1 means no tracklet in that layer
  short mCollisionId;                // the collision ID of the tracklets attached to this track; is used to retrieve the BC information for this track after the tracking is done
  unsigned char mFlags;              // bits 0 to 5 indicate whether track is findable in layer 0 to 5, bit 6 indicates an ambiguous track and bit 7 flags if the track is stopped in the TRD
  unsigned char mIsCrossingNeighbor; // bits 0 to 5 indicate if a tracklet was either a neighboring tracklet (e.g. a potential split tracklet) or crossed a padrow, bit 6 indicates that a neighbor in any layer has been found and bit 7 if a padrow was crossed

 private:
  GPUd() void initialize();
#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)
  ClassDefNV(GPUTRDTrack_t, 3);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // !((defined(__CINT__) || defined(__ROOTCINT__)) && !defined(__CLING__))

#endif // GPUTRDTRACK_H
