// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file V0Hypothesis.h
/// \brief V0 hypothesis checker
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_V0_HYPOTHESIS_H
#define ALICEO2_V0_HYPOTHESIS_H

#include "ReconstructionDataFormats/PID.h"
#include "DetectorsVertexing/SVertexerParams.h"

namespace o2
{
namespace vertexing
{

class V0Hypothesis
{

 public:
  using PID = o2::track::PID;

  void set(PID v0, PID ppos, PID pneg, float sig, float nSig, float margin, float cpt, float bz = 0.f);
  void set(PID v0, PID ppos, PID pneg, const float pars[SVertexerParams::NPIDParams], float bz = 0.f);

  float getMassV0Hyp() const { return PID::getMass(mPIDV0); }
  float getMassPosProng() const { return PID::getMass(mPIDPosProng); }
  float getMassNegProng() const { return PID::getMass(mPIDNegProng); }

  float calcMass2(float p2Pos, float p2Neg, float p2V0) const
  {
    // calculate v0 mass from squared momentum of its prongs and total momentum
    float ePos = std::sqrt(p2Pos + getMass2PosProng()), eNeg = std::sqrt(p2Neg + getMass2NegProng()), eV0 = ePos + eNeg;
    return eV0 * eV0 - p2V0;
  }

  float calcMass(float p2Pos, float p2Neg, float p2V0) const { return std::sqrt(calcMass2(p2Pos, p2Neg, p2V0)); }

  bool check(float p2Pos, float p2Neg, float p2V0, float ptV0) const
  { // check if given mass and pt is matching to hypothesis
    return check(calcMass(p2Pos, p2Neg, p2V0), ptV0);
  }

  bool check(float mass, float pt) const
  { // check if given mass and pt is matching to hypothesis
    return std::abs(mass - getMassV0Hyp()) < getMargin(pt);
  }

  float getSigma(float pt) const { return 1.f + mCPt * pt; }
  float getMargin(float pt) const { return mNSigma * getSigma(pt) + mMargin; }

 private:
  float getMass2PosProng() const { return PID::getMass2(mPIDPosProng); }
  float getMass2NegProng() const { return PID::getMass2(mPIDNegProng); }

  PID mPIDV0 = PID::K0;
  PID mPIDPosProng = PID::Pion;
  PID mPIDNegProng = PID::Pion;

  float mNSigma = 0.; // number of sigmas of mass res
  float mSigma = 0.;  // sigma of mass res at 0 pt
  float mMargin = 0.; // additive safety margin in mass cut
  float mCPt = 0.;    // pT dependence of mass resolution parameterized as mSigma*(1+mC1*pt);

  ClassDefNV(V0Hypothesis, 1);
};

} // namespace vertexing
} // namespace o2

#endif
