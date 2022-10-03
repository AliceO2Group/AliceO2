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

/// \file SVertexHypothesis.h
/// \brief V0 or Cascade and 3-body decay hypothesis checker
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_SVERTEX_HYPOTHESIS_H
#define ALICEO2_SVERTEX_HYPOTHESIS_H

#include "ReconstructionDataFormats/PID.h"
#include <cmath>
#include <array>

namespace o2
{
namespace vertexing
{

class SVertexHypothesis
{

 public:
  using PID = o2::track::PID;
  enum PIDParams { SigmaM,  // sigma of mass res at 0 pt
                   NSigmaM, // number of sigmas of mass res
                   MarginM, // additive safety margin in mass cut
                   CPt };   // pT dependence of mass resolution parameterized as mSigma*(1+mC1*pt);

  static constexpr int NPIDParams = 4;

  void set(PID v0, PID ppos, PID pneg, float sig, float nSig, float margin, float cpt, float bz = 0.f);
  void set(PID v0, PID ppos, PID pneg, const float pars[NPIDParams], float bz = 0.f);

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

  float getSigma(float pt) const { return mPars[SigmaM] * (1.f + mPars[CPt] * pt); }
  float getMargin(float pt) const { return mPars[NSigmaM] * getSigma(pt) + mPars[MarginM]; }

 private:
  float getMass2PosProng() const { return PID::getMass2(mPIDPosProng); }
  float getMass2NegProng() const { return PID::getMass2(mPIDNegProng); }

  PID mPIDV0{PID::K0};
  PID mPIDPosProng{PID::Pion};
  PID mPIDNegProng{PID::Pion};

 public: // to be deleted
  std::array<float, NPIDParams> mPars{};

  ClassDefNV(SVertexHypothesis, 1);
};

class SVertex3Hypothesis
{

 public:
  using PID = o2::track::PID;
  enum PIDParams { SigmaM,  // sigma of mass res at 0 pt
                   NSigmaM, // number of sigmas of mass res
                   MarginM, // additive safety margin in mass cut
                   CPt };   // pT dependence of mass resolution parameterized as mSigma*(1+mC1*pt);

  static constexpr int NPIDParams = 4;

  void set(PID v0, PID ppos, PID pneg, PID pbach, float sig, float nSig, float margin, float cpt, float bz = 0.f);
  void set(PID v0, PID ppos, PID pneg, PID pbach, const float pars[NPIDParams], float bz = 0.f);

  float getMassV0Hyp() const { return PID::getMass(mPIDV0); }
  float getMassPosProng() const { return PID::getMass(mPIDPosProng); }
  float getMassNegProng() const { return PID::getMass(mPIDNegProng); }
  float getMassBachProng() const { return PID::getMass(mPIDBachProng); }

  float calcMass2(float p2Pos, float p2Neg, float p2Bach, float p2Tot) const
  {
    // calculate v0 mass from squared momentum of its prongs and total momentum
    float ePos = std::sqrt(p2Pos + getMass2PosProng()), eNeg = std::sqrt(p2Neg + getMass2NegProng()), eBach = std::sqrt(p2Bach + getMass2BachProng()), eVtx = ePos + eNeg + eBach;
    return eVtx * eVtx - p2Tot;
  }

  float calcMass(float p2Pos, float p2Neg, float p2Bach, float p2Tot) const { return std::sqrt(calcMass2(p2Pos, p2Neg, p2Bach, p2Tot)); }

  bool check(float p2Pos, float p2Neg, float p2Bach, float p2Tot, float ptV0) const
  { // check if given mass and pt is matching to hypothesis
    return check(calcMass(p2Pos, p2Neg, p2Bach, p2Tot), ptV0);
  }

  bool check(float mass, float pt) const
  { // check if given mass and pt is matching to hypothesis
    return std::abs(mass - getMassV0Hyp()) < getMargin(pt);
  }

  float getSigma(float pt) const { return mPars[SigmaM] * (1.f + mPars[CPt] * pt); }
  float getMargin(float pt) const { return mPars[NSigmaM] * getSigma(pt) + mPars[MarginM]; }

 private:
  float getMass2PosProng() const { return PID::getMass2(mPIDPosProng); }
  float getMass2NegProng() const { return PID::getMass2(mPIDNegProng); }
  float getMass2BachProng() const { return PID::getMass2(mPIDBachProng); }

  PID mPIDV0{PID::HyperTriton};
  PID mPIDPosProng{PID::Proton};
  PID mPIDNegProng{PID::Pion};
  PID mPIDBachProng{PID::Deuteron};

 public: // to be deleted
  std::array<float, NPIDParams> mPars{};

  ClassDefNV(SVertex3Hypothesis, 1);
};

} // namespace vertexing
} // namespace o2

#endif
