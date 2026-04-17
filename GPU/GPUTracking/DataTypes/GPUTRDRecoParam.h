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

/// \file RecoParam.h
/// \brief Error parameterizations and helper functions for TRD reconstruction
/// \author Ole Schmidt

#ifndef O2_GPU_TRD_RECOPARAM_H
#define O2_GPU_TRD_RECOPARAM_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonArray.h"
#include "GPUCommonMath.h"

namespace o2
{
namespace gpu
{
struct GPUSettingsRec;

class GPUTRDRecoParam
{
 public:
  GPUTRDRecoParam() = default;
  GPUTRDRecoParam(const GPUTRDRecoParam&) = default;
  ~GPUTRDRecoParam() = default;

  /// Load parameterization for given magnetic field
  void init(float bz, const GPUSettingsRec* rec = nullptr);

#if !defined(GPUCA_GPUCODE_DEVICE)
  /// Recalculate tracklet covariance based on phi angle of related track
  GPUd() void recalcTrkltCov(const float tilt, const float snp, const float rowSize, std::array<float, 3>& cov) const
  {
    recalcTrkltCov(tilt, snp, rowSize, cov.data());
  }
#endif
  GPUd() void recalcTrkltCov(const float tilt, const float snp, const float rowSize, float* cov) const;

  /// Get tracklet r-phi resolution for given phi angle
  /// Resolution depends on the track angle sin(phi) = snp and is approximated by the formula
  /// sigma_y(snp) = sqrt(a^2 + c^2 * (snp - b)^2)
  /// more details are given in http://cds.cern.ch/record/2724259 in section 5.3.3
  /// \param phi angle of related track
  /// \return sigma_y^2 of tracklet
  GPUd() float getRPhiRes(float snp) const { return (mRPhiA2 + mRPhiC2 * (snp - mLorentzAngle) * (snp - mLorentzAngle)); }
  GPUd() float getDyRes(float snp) const { return mDyA2 + mDyC2 * (snp - mLorentzAngle) * (snp - mLorentzAngle); }           // a^2 + c^2 * (snp - b)^2
  GPUd() float convertAngleToDy(float snp) const { return 3.f * snp / CAMath::Sqrt(1 - snp * snp); }                         // when calibrated, sin(phi) = (dy / xDrift) / sqrt(1+(dy/xDrift)^2) works well
  GPUd() float getCorrYDy(float snp) const { return mCorrYDyA + mCorrYDyC * (snp - mLorentzAngle) * (snp - mLorentzAngle); } // a + c * (snp - b)^2
  GPUd() float getPileUpProbTracklet(int nBC, int Q0, int Q1) const;
  GPUd() float getPileUpProbTrack(int nBC, std::array<int, 6> Q0, std::array<int, 6> Q1) const;

  /// Get tracklet z correction coefficient for track-eta based corraction
  GPUd() float getZCorrCoeffNRC() const { return mZCorrCoefNRC; }

  /// Get BC intervals for pile-up
  GPUd() int getPileUpRangeBefore() const { return mPileUpRangeBefore; }
  GPUd() int getPileUpRangeAfter() const { return mPileUpRangeAfter; }

 private:
  // tracklet error parameterization depends on the magnetic field
  float mLorentzAngle{0.f};
  // rphi
  float mRPhiA2{1.f}; ///< parameterization for tracklet position resolution
  float mRPhiC2{0.f}; ///< parameterization for tracklet position resolution
  // angle
  float mDyA2{1.225e-3f}; ///< parameterization for tracklet angular resolution
  float mDyC2{0.f};       ///< parameterization for tracklet angular resolution
  // correlation coefficient between y residual and dy residual
  float mCorrYDyA{0.f};
  float mCorrYDyC{0.f};

  float mZCorrCoefNRC{1.4f}; ///< tracklet z-position depends linearly on track dip angle

  // pile-up prob parametrization, depending on charges
  // default parametrization, all tracklets
  int mPileUpRangeBefore{-130}; ///< maximal number of BC for which pile-up from previous collision has an influence
  int mPileUpMaxProb{0};        ///< number of BC with respect to triggered BC for the event with maximal probability
  int mPileUpRangeAfter{70};    ///< maximal number of BC for which pile-up from next collision has an influence
  // tracklets with Q0!=0 and Q1!=0
  int mPileUpRangeBefore11{-130}; ///< maximal number of BC for which pile-up from previous collision has an influence
  int mPileUpMaxProb11{0};        ///< number of BC with respect to triggered BC for the event with maximal probability
  int mPileUpRangeAfter11{30};    ///< maximal number of BC for which pile-up from next collision has an influence
  // tracklets with Q0=0 and Q1!=0
  int mPileUpRangeBefore01{-80}; ///< maximal number of BC for which pile-up from previous collision has an influence
  int mPileUpMaxProb01{30};      ///< number of BC with respect to triggered BC for the event with maximal probability
  int mPileUpRangeAfter01{70};   ///< maximal number of BC for which pile-up from next collision has an influence
  // tracklets with Q0!=0 and Q1=0
  int mPileUpRangeBefore10{-130}; ///< maximal number of BC for which pile-up from previous collision has an influence
  int mPileUpMaxProb10{-60};      ///< number of BC with respect to triggered BC for the event with maximal probability
  int mPileUpRangeAfter10{30};    ///< maximal number of BC for which pile-up from next collision has an influence
  // tracklets with Q0=0 and Q1=0
  int mPileUpRangeBefore00{-10}; ///< maximal number of BC for which pile-up from previous collision has an influence
  int mPileUpMaxProb00{22};      ///< number of BC with respect to triggered BC for the event with maximal probability
  int mPileUpRangeAfter00{40};   ///< maximal number of BC for which pile-up from next collision has an influence

  ClassDefNV(GPUTRDRecoParam, 4);
};

} // namespace gpu
} // namespace o2

#endif // O2_GPU_TRD_RECOPARAM_H
