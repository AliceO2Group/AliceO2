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
#include "ReconstructionDataFormats/TrackLTIntegral.h"
#include "CommonConstants/LHCConstants.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

template <>
class trackInterface<o2::track::TrackParCov> : public o2::track::TrackParCov
{
 public:
  GPUdDefault() trackInterface<o2::track::TrackParCov>() = default;
  trackInterface<o2::track::TrackParCov>(const o2::track::TrackParCov& param) = delete;
  GPUd() trackInterface<o2::track::TrackParCov>(const o2::dataformats::TrackTPCITS& trkItsTpc) : o2::track::TrackParCov(trkItsTpc.getParamOut()) {}
  GPUd() trackInterface<o2::track::TrackParCov>(const o2::tpc::TrackTPC& trkTpc) : o2::track::TrackParCov(trkTpc.getParamOut()) {}

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
  GPUd() trackInterface<o2::track::TrackParCov>(const GPUTPCGMMergedTrack& trk);
  GPUd() trackInterface<o2::track::TrackParCov>(const gputpcgmmergertypes::GPUTPCOuterParam& param);
  GPUd() void updateCovZ2(float addZerror) { updateCov(addZerror, o2::track::CovLabels::kSigZ2); }
  GPUd() o2::track::TrackLTIntegral& getLTIntegralOut() { return mLTOut; }
  GPUd() const o2::track::TrackLTIntegral& getLTIntegralOut() const { return mLTOut; }
  GPUd() o2::track::TrackParCov& getOuterParam() { return mParamOut; }
  GPUd() const o2::track::TrackParCov& getOuterParam() const { return mParamOut; }

  GPUdi() const float* getPar() const { return getParams(); }

  GPUdi() bool CheckNumericalQuality() const { return true; }

  GPUdi() void setPileUpDistance(unsigned char bwd, unsigned char fwd) { setUserField((((unsigned short)bwd) << 8) | fwd); }
  GPUdi() bool hasPileUpInfo() const { return getUserField() != 0; }
  GPUdi() bool hasPileUpInfoBothSides() const { return getPileUpDistanceBwd() > 0 && getPileUpDistanceFwd() > 0; }
  GPUdi() unsigned char getPileUpDistanceBwd() const { return getUserField() >> 8; }
  GPUdi() unsigned char getPileUpDistanceFwd() const { return getUserField() & 255; }
  GPUdi() unsigned short getPileUpSpan() const { return ((unsigned short)getPileUpDistanceBwd()) + getPileUpDistanceFwd(); }
  GPUdi() float getPileUpMean() const { return hasPileUpInfoBothSides() ? 0.5 * (getPileUpDistanceFwd() + getPileUpDistanceBwd()) : getPileUpDistanceFwd() + getPileUpDistanceBwd(); }
  GPUdi() float getPileUpTimeShiftMUS() const { return getPileUpMean() * o2::constants::lhc::LHCBunchSpacingMUS; }
  GPUdi() float getPileUpTimeErrorMUS() const { return getPileUpSpan() * o2::constants::lhc::LHCBunchSpacingMUS / 3.4641016; }

  typedef o2::track::TrackParCov baseClass;

 private:
  o2::track::TrackLTIntegral mLTOut;
  o2::track::TrackParCov mParamOut;

  ClassDefNV(trackInterface, 1);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
