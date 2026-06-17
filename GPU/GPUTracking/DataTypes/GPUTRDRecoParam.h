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
  GPUd() void recalcTrkltCov(const float tilt, const float snp, const float rowSize, std::array<float, 3>& cov, const float pull = 0., const int occupancy = 0) const
  {
    recalcTrkltCov(tilt, snp, rowSize, cov.data(), pull, occupancy);
  }
#endif
  GPUd() void recalcTrkltCov(const float tilt, const float snp, const float rowSize, float* cov, const float pull = 0., const int occupancy = 0) const;

  GPUd() float getRPhiRes(float snp, float pull = 0.f, int occupancy = 0) const;
  GPUd() float getDyRes(float snp, int occupancy = 0) const { return mDyA2 + mDyC2 * (snp - mLorentzAngle) * (snp - mLorentzAngle) + mOccDyA * occupancy; } // a^2 + c^2 * (snp - b)^2
  GPUd() float convertAngleToDy(float snp) const { return 3.f * snp / CAMath::Sqrt(1 - snp * snp); }                                                        // when calibrated, sin(phi) = (dy / xDrift) / sqrt(1+(dy/xDrift)^2) works well
  GPUd() float getCorrYDy() const { return mCorrYDy; }
  GPUd() float getPileUpProbTracklet(int nBC, bool withChargeInfo, bool Q0 = true, bool Q1 = true) const;
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
  float mRPhiA{1.f};    ///< parameterization for tracklet position resolution
  float mRPhiATgp{1.f}; ///< parameterization for tracklet position resolution
  float mRPhiC2{0.f};   ///< parameterization for tracklet position resolution
  // angle
  float mDyA2{1.225e-3f}; ///< parameterization for tracklet angular resolution
  float mDyC2{0.f};       ///< parameterization for tracklet angular resolution
  // variation in y when dy variates by one sigma (= cov / sigma_dy = corr * sigma_y) (valid within 2sigma of dy)
  float mCorrYDy{0.13f};
  // error parametrization vs angular pull (pol2)
  float mPullA{6.8e-3f};
  float mPullB{0.049f};
  // error parametrization of y position vs occupancy defined as ntracklets within chamber (prop to sqrt(occupancy))
  float mOccA{3.3e-4f};
  // error parametrization for dy vs occupancy defined as ntracklets within chamber (prop to sqrt(occupancy))
  float mOccDyA{2.5e-4f};

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

/// Get tracklet r-phi resolution for given phi angle
/// Resolution depends on the track angle sin(phi) = snp and is approximated by the formula
/// sigma_y(snp) = sqrt(a^2 + c^2 * (snp - b)^2)
/// more details are given in http://cds.cern.ch/record/2724259 in section 5.3.3
/// \param phi angle of related track
/// \return sigma_y^2 of tracklet
/// also depend on absolute pull and on chamber occupancy
GPUdi() float GPUTRDRecoParam::getRPhiRes(float snp, float pull, int occupancy) const
{
  // flat uncertainty + radial-alignment uncertainty depending on tan(phi)
  float tgp = (CAMath::Abs(snp) < 0.99999f) ? CAMath::Abs(snp) / CAMath::Sqrt(1 - snp * snp) : 1e6;
  float resIdeal = mRPhiA + mRPhiATgp * tgp;
  if (pull > 10) {
    // parametrization does not really work well for such large pull values
    pull = 10.f;
  }
  float resPull = mPullA * pull * pull + mPullB * pull; // parametrization as pol2 summed in quadrature
  float resOccupancy = mOccA * occupancy;               // parametrization as sqrt() summed in quadrature
  return (resIdeal * resIdeal + mRPhiC2 * (snp - mLorentzAngle) * (snp - mLorentzAngle) + resPull * resPull + resOccupancy);
}

GPUdi() float GPUTRDRecoParam::getPileUpProbTracklet(int nBC, bool withChargeInfo, bool Q0, bool Q1) const
{
  // get the probability that the tracklet with charges Q0 and Q1 belongs to a given BC, with a (signed) distance nBC from the TRD-triggered BC
  // parametrization depends on whether charges are 0 (bool is false) or not (bool is true)

  float prob = 0.;

  int maxBC = mPileUpRangeAfter;
  int minBC = mPileUpRangeBefore;
  int maxProbBC = mPileUpMaxProb;
  if (nBC <= mPileUpRangeBefore || nBC >= mPileUpRangeAfter) {
    return prob;
  }

  if (withChargeInfo) {
    if (Q0 && Q1) {
      maxBC = mPileUpRangeAfter11;
      minBC = mPileUpRangeBefore11;
      maxProbBC = mPileUpMaxProb11;
    }
    if (!Q0 && Q1) {
      maxBC = mPileUpRangeAfter01;
      minBC = mPileUpRangeBefore01;
      maxProbBC = mPileUpMaxProb01;
    }
    if (Q0 && !Q1) {
      maxBC = mPileUpRangeAfter10;
      minBC = mPileUpRangeBefore10;
      maxProbBC = mPileUpMaxProb10;

      // if Q1 = 0, there is a second maximum at nBC=0, probably due to tracklets with low energy loss in the drift/TR regions
      // so we enlarge the probability around there
      if (nBC > maxProbBC && nBC <= 0) {
        prob += 2. / (maxBC - minBC) / (0 - maxProbBC) * (nBC - maxProbBC);
      }
      if (nBC > 0 && nBC < maxBC) {
        prob += 2. / (maxBC - minBC) / (0 - maxBC) * (nBC - maxBC);
      }
    }
    if (!Q0 && !Q1) {
      maxBC = mPileUpRangeAfter00;
      minBC = mPileUpRangeBefore00;
      maxProbBC = mPileUpMaxProb00;
    }
  }

  // prob is 0 if the BC is too far, maximal for a given nBC, and with two linear functions in between. The maximum is chosen so that the integral is 1.
  if (nBC <= minBC || nBC >= maxBC) {
    return 0.;
  }
  float maxProb = 2. / (maxBC - minBC);
  if (nBC > minBC && nBC <= maxProbBC) {
    prob += maxProb / (maxProbBC - minBC) * (nBC - minBC);
  } else {
    prob += maxProb / (maxProbBC - maxBC) * (nBC - maxBC);
  }
  return prob;
}

GPUdi() float GPUTRDRecoParam::getPileUpProbTrack(int nBC, std::array<int, 6> Q0, std::array<int, 6> Q1) const
{
  // get the probability that the track belongs to a given BC, with a (signed) distance nBC from the TRD-triggered BC
  // it depends on the individual probabilities for every of its tracklets.
  //
  // If P(BC|L0,L1,...) is the probability that the track belongs to a given BC, given the information on the tracklet charges in L0,L1, ...
  // P(BC|L0,L1,...) proportional to P(BC)*P(L0,L1,...|BC), prop to P(BC)*P(L0|BC)*P(L1|BC)*... since for a given track and BC, charge in different layers are independent
  // prop to P(BC) * P(BC|L0)/P(BC) * P(BC|L1)/P(BC) * ...
  //
  // P(BC) is the probability with no charge information: we start from this probability, and each tracklet adds new information on pileup probability

  // basic probability, if we had no info on the charges
  float probNoInfo = GPUTRDRecoParam::getPileUpProbTracklet(nBC, false);

  float probTrack = probNoInfo;
  if (probNoInfo < 1e-6f)
    return 0.;

  // For each tracklet, we add the info on its charge
  for (int i = 0; i < 6; i++) {
    // negative charge values if the tracklet is not present
    if (Q0[i] < 0 || Q1[i] < 0)
      continue;
    float probTracklet = GPUTRDRecoParam::getPileUpProbTracklet(nBC, true, (Q0[i] != 0), (Q1[i] != 0));
    probTrack *= probTracklet / probNoInfo;
  }

  return probTrack;
}

} // namespace gpu
} // namespace o2

#endif // O2_GPU_TRD_RECOPARAM_H
