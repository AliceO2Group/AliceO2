// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTRDInterfaceO2Track.h
/// \author Ole Schmidt

#ifndef GPUTRDINTERFACEO2TRACK_H
#define GPUTRDINTERFACEO2TRACK_H

// This is the interface for the GPUTRDTrack based on the O2 track type
#include "GPUCommonDef.h"
namespace GPUCA_NAMESPACE
{
namespace gpu
{
template <typename T>
class trackInterface;
class GPUTPCGMMergedTrack;
namespace gputpcgmmergertypes
{
struct GPUTPCOuterParam;
} // namespace gputpcgmmergertypes
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "GPUTRDO2BaseTrack.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <>
class trackInterface<GPUTRDO2BaseTrack> : public GPUTRDO2BaseTrack
{
 public:
  GPUdDefault() trackInterface<GPUTRDO2BaseTrack>() = default;
  trackInterface<GPUTRDO2BaseTrack>(const GPUTRDO2BaseTrack& param) = delete;
  GPUd() trackInterface<GPUTRDO2BaseTrack>(const o2::dataformats::TrackTPCITS& trkItsTpc, float vDrift) : GPUTRDO2BaseTrack(trkItsTpc.getParamOut())
  {
    mTime = trkItsTpc.getTimeMUS().getTimeStamp();
    mTimeAddMax = trkItsTpc.getTimeMUS().getTimeStampError();
    mTimeSubMax = trkItsTpc.getTimeMUS().getTimeStampError();
    mRefITS = trkItsTpc.getRefITS();
    mRefTPC = trkItsTpc.getRefTPC();
    float tmp = trkItsTpc.getTimeMUS().getTimeStampError() * vDrift;
    updateCov(tmp * tmp, o2::track::CovLabels::kSigZ2); // account for time uncertainty by increasing sigmaZ2
  }
  GPUd() trackInterface<GPUTRDO2BaseTrack>(const o2::tpc::TrackTPC& trkTpc, float tbWidth, float vDrift, unsigned int iTrk) : GPUTRDO2BaseTrack(trkTpc.getParamOut())
  {
    mRefTPC = {iTrk, o2::dataformats::GlobalTrackID::TPC};
    mTime = trkTpc.getTime0() * tbWidth;
    mTimeAddMax = trkTpc.getDeltaTFwd() * tbWidth;
    mTimeSubMax = trkTpc.getDeltaTBwd() * tbWidth;
    if (trkTpc.hasASideClustersOnly()) {
      mSide = -1;
    } else if (trkTpc.hasCSideClustersOnly()) {
      mSide = 1;
    } else {
      // CE-crossing tracks are not shifted along z, but the time uncertainty is taken into account by increasing sigmaZ2
      float timeWindow = (mTimeAddMax + mTimeSubMax) * .5f;
      float tmp = timeWindow * vDrift;
      updateCov(tmp * tmp, o2::track::CovLabels::kSigZ2);
    }
  }
  GPUd() void set(float x, float alpha, const float* param, const float* cov)
  {
    setX(x);
    setAlpha(alpha);
    for (int i = 0; i < 5; i++) {
      setParam(param[i], i);
    }
    for (int i = 0; i < 15; i++) {
      setCov(cov[i], i);
    }
  }
  GPUd() trackInterface<GPUTRDO2BaseTrack>(const GPUTPCGMMergedTrack& trk);
  GPUd() trackInterface<GPUTRDO2BaseTrack>(const gputpcgmmergertypes::GPUTPCOuterParam& param);

  GPUdi() const float* getPar() const { return getParams(); }
  GPUdi() float getTime() const { return mTime; }
  GPUdi() void setTime(float t) { mTime = t; }
  GPUdi() float getTimeMax() const { return mTime + mTimeAddMax; }
  GPUdi() float getTimeMin() const { return mTime - mTimeSubMax; }
  GPUdi() short getSide() const { return mSide; }
  GPUdi() float getZShift() const { return mZShift; }
  GPUdi() void setZShift(float z) { mZShift = z; }

  GPUdi() bool CheckNumericalQuality() const { return true; }

  typedef GPUTRDO2BaseTrack baseClass;

 private:
  o2::dataformats::GlobalTrackID mRefTPC; // reference on TPC track entry in its original container
  o2::dataformats::GlobalTrackID mRefITS; // reference on ITS track entry in its original container
  float mTime{-1.f};                      // time estimate for this track in us
  float mTimeAddMax{0.f};                 // max. time that can be added to this track in us
  float mTimeSubMax{0.f};                 // max. time that can be subtracted to this track in us
  short mSide{0};                         // -1 : A-side, +1 : C-side (relevant only for TPC-only tracks)
  float mZShift{0.f};                     // calculated new for each TRD trigger candidate for this track

  ClassDefNV(trackInterface, 1);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
