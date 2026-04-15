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

/// \file GPUTRDRecoParam.cxx
/// \brief Error parameterizations and helper functions for TRD reconstruction
/// \author Ole Schmidt

#include "GPUO2InterfaceConfiguration.inc"
#include "GPUSettings.h"
#include "GPUTRDRecoParam.h"
#include "GPUCommonLogger.h"
#include "GPUCommonMath.h"

using namespace o2::gpu;

// error parameterizations taken from http://cds.cern.ch/record/2724259 Appendix A
void GPUTRDRecoParam::init(float bz, const GPUSettingsRec* rec)
{
  float resRPhiIdeal2 = 1.6e-3f;
  if (rec) {
    resRPhiIdeal2 = rec->trd.trkltResRPhiIdeal * rec->trd.trkltResRPhiIdeal;
  }
#ifndef GPUCA_STANDALONE
  else {
    const auto& rtrd = GPU_GET_CONFIG(GPUSettingsRecTRD);
    resRPhiIdeal2 = rtrd.trkltResRPhiIdeal * rtrd.trkltResRPhiIdeal;
  }
#endif

  if (CAMath::Abs(CAMath::Abs(bz) - 2) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.2 T
      mRPhiC2 = 4.55e-2f;
    } else {
      // magnetic field -0.2 T
      mRPhiC2 = 4.55e-2f;
    }
  } else if (CAMath::Abs(CAMath::Abs(bz) - 5) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.5 T
      mRPhiC2 = 0.0961f;
    } else {
      // magnetic field -0.5 T
      mRPhiC2 = 0.1156f;
    }
  } else {
    LOGP(warning, "No error parameterization available for Bz= {}. Keeping default value (sigma_y = const. = 1cm)", bz);
  }

  mRPhiA2 = resRPhiIdeal2;
  mLorentzAngle = -0.02f + 0.13f * bz / 5.f;

  mDyA2 = 6e-3f;
  mDyC2 = 0.3f;
  mCorrYDyA = 0.27f;
  mCorrYDyC = -0.44f;

  LOGP(info, "Loaded parameterizations for Bz={}: PhiRes:[{},{},{}] DyRes:[{},{},{}] CorrYDy:[{},{},{}]",
       bz, mRPhiA2, mLorentzAngle, mRPhiC2, mDyA2, mLorentzAngle, mDyC2, mCorrYDyA, mLorentzAngle, mCorrYDyC);
}

void GPUTRDRecoParam::recalcTrkltCov(const float tilt, const float snp, const float rowSize, float* cov) const
{
  float t2 = tilt * tilt;      // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = getRPhiRes(snp);
  float sz2 = rowSize * rowSize / 12.f;
  cov[0] = c2 * (sy2 + t2 * sz2);
  cov[1] = c2 * tilt * (sz2 - sy2);
  cov[2] = c2 * (t2 * sy2 + sz2);
}

float GPUTRDRecoParam::getPileUpProbTracklet(int nBC, int Q0, int Q1) const
{
  // get the probability that the tracklet with charges Q0 and Q1 belongs to a given BC, with a (signed) distance nBC from the TRD-triggered BC
  // parametrization depends on whether charges are 0 or not
  int maxBC = mPileUpRangeAfter;
  int minBC = mPileUpRangeBefore;
  int maxProbBC = mPileUpMaxProb;
  if (Q0 > 0 && Q1 > 0) {
    maxBC = mPileUpRangeAfter11;
    minBC = mPileUpRangeBefore11;
    maxProbBC = mPileUpMaxProb11;
  }
  if (Q0 == 0 && Q1 > 0) {
    maxBC = mPileUpRangeAfter01;
    minBC = mPileUpRangeBefore01;
    maxProbBC = mPileUpMaxProb01;
  }
  if (Q0 > 0 && Q1 == 0) {
    maxBC = mPileUpRangeAfter10;
    minBC = mPileUpRangeBefore10;
    maxProbBC = mPileUpMaxProb10;
  }
  if (Q0 == 0 && Q1 == 0) {
    maxBC = mPileUpRangeAfter00;
    minBC = mPileUpRangeBefore00;
    maxProbBC = mPileUpMaxProb00;
  }

  // prob is 0 if the BC is too far, maximal for a given nBC, and with two linear functions in between. The maximum is chosen so that the integral is 1.
  if (nBC <= minBC || nBC >= maxBC)
    return 0.;
  float maxProb = 2. / (maxBC - minBC);
  if (nBC > minBC && nBC <= maxProbBC)
    return maxProb / (maxProbBC - minBC) * (nBC - minBC);
  else
    return maxProb / (maxProbBC - maxBC) * (nBC - maxBC);
}

float GPUTRDRecoParam::getPileUpProbTrack(int nBC, std::array<int, 6> Q0, std::array<int, 6> Q1) const
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
  float probNoInfo = GPUTRDRecoParam::getPileUpProbTracklet(nBC, -1, -1);

  float probTrack = probNoInfo;
  if (probNoInfo < 1e-6f)
    return 0.;

  // For each tracklet, we add the info on its charge
  for (int i = 0; i < 6; i++) {
    // negative charge values if the tracklet is not present
    if (Q0[i] < 0 || Q1[i] < 0)
      continue;
    float probTracklet = GPUTRDRecoParam::getPileUpProbTracklet(nBC, Q0[i], Q1[i]);
    probTrack *= probTracklet / probNoInfo;
  }

  return probTrack;
}
