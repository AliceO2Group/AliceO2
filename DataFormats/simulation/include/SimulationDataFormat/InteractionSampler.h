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

/// @brief Simulated interaction record sampler

#ifndef ALICEO2_INTERACTIONSAMPLER_H
#define ALICEO2_INTERACTIONSAMPLER_H

#include <Rtypes.h>
#include <TMath.h>
#include <TRandom.h>
#include <vector>
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonConstants/LHCConstants.h"
#include <TH1F.h>

namespace o2
{
namespace steer
{
class InteractionSampler
{
 public:
  static constexpr float Sec2NanoSec = 1.e9; // s->ns conversion
  const o2::InteractionTimeRecord& generateCollisionTime();
  void generateCollisionTimes(std::vector<o2::InteractionTimeRecord>& dest);

  void init();

  void setInteractionRate(float rateHz)
  {
    mIntRate = rateHz;
    mMuBC = -1.; // invalidate
  }
  float getInteractionRate() const { return mIntRate; }
  void setFirstIR(const o2::InteractionRecord& ir)
  {
    mFirstIR.InteractionRecord::operator=(ir);
    if (mFirstIR.orbit == 0 && mFirstIR.bc < 4) {
      mFirstIR.bc = 4;
    }
  }
  const o2::InteractionRecord& getFirstIR() const { return mFirstIR; }

  void setMuPerBC(float mu)
  {
    mMuBC = mu;
    mIntRate = -1.; // invalidate
  }
  float getMuPerBC() const { return mMuBC; }
  void setBCTimeRMS(float tNS = 0.2) { mBCTimeRMS = tNS; }
  float getBCTimeRMS() const { return mBCTimeRMS; }
  const BunchFilling& getBunchFilling() const { return mBCFilling; }
  BunchFilling& getBunchFilling() { return mBCFilling; }
  void setBunchFilling(const BunchFilling& bc) { mBCFilling = bc; }
  void setBunchFilling(const std::string& bcFillingFile);

  void print() const;

  /// Poisson number >= 1 for mean mu, drawn by inverting its cumulative distribution
  static int drawZeroTruncatedPoisson(double mu);

 protected:
  virtual int simulateInteractingBC();
  void nextCollidingBC(int n);
  int drawBCJump() const;         ///< random jump in colliding BCs to the next interacting one
  int drawNCollInBC() const;      ///< random number of collisions in an interacting BC (>= 1)
  float drawCollTimeInBC() const; ///< random collision time within a BC

  o2::InteractionTimeRecord mIR{{0, 0}, 0.};
  o2::InteractionTimeRecord mFirstIR{{4, 0}, 0.};
  int mIntBCCache = 0; ///< N interactions left for current BC

  float mIntRate = -1.;   ///< total interaction rate in Hz
  float mBCTimeRMS = 0.2; ///< BC time spread in NANOSECONDS
  double mMuBC = -1.;     ///< interaction probability per BC

  o2::BunchFilling mBCFilling;           ///< patter of active BCs
  std::vector<float> mTimeInBC;          ///< interaction times within single BC
  std::vector<uint16_t> mInteractingBCs; // vector of interacting BCs
  int mCurrBCIdx = 0;                    ///< counter for current interacting bunch

  static constexpr float DefIntRate = 50e3; ///< default interaction rate

  ClassDef(InteractionSampler, 2);
};

//_________________________________________________
inline void InteractionSampler::generateCollisionTimes(std::vector<o2::InteractionTimeRecord>& dest)
{
  // fill vector with interaction records
  dest.clear();
  for (int i = dest.capacity(); i--;) {
    dest.push_back(generateCollisionTime());
  }
}

//_________________________________________________
inline void InteractionSampler::nextCollidingBC(int n)
{
  /// get colliding BC as n-th after current one
  if ((mCurrBCIdx += n) >= (int)mInteractingBCs.size()) {
    mIR.orbit += mCurrBCIdx / mInteractingBCs.size();
    mCurrBCIdx %= mInteractingBCs.size();
  }
  mIR.bc = mInteractingBCs[mCurrBCIdx];
}

// Special case of InteractionSampler without actual sampling.
// Engineers interaction sequence by putting one in each N-th BC with multiplicity mult.
class FixedSkipBC_InteractionSampler : public InteractionSampler
{

 public:
  FixedSkipBC_InteractionSampler(int every_n, int mult) : mEveryN{every_n}, mMultiplicity{mult}, InteractionSampler() {}

 protected:
  int simulateInteractingBC() override;

 private:
  int mEveryN;       // the skip number ---> fills every N-th BC in the bunch filling scheme
  int mMultiplicity; // how many events to put if bc is filled
  ClassDefOverride(FixedSkipBC_InteractionSampler, 1);
};

// A version of the interaction sampler which can sample according to non-uniform mu(bc) as
// observed during data taking.
class NonUniformMuInteractionSampler : public InteractionSampler
{
 public:
  NonUniformMuInteractionSampler() : InteractionSampler() { mBCIntensityScales.resize(o2::constants::lhc::LHCMaxBunches, 1); }
  bool setBCIntensityScales(const std::vector<float>& scales_from_vector);
  bool setBCIntensityScales(const TH1F& scales_from_histo); // initialize scales

  // helper function to determine the scales from a histogram (count from event selection analysis)
  std::vector<float> determineBCIntensityScalesFromHistogram(const TH1F& scales_from_histo);

  const std::vector<float>& getBCIntensityScales() const { return mBCIntensityScales; }

 protected:
  int simulateInteractingBC() override;
  int getBCJump() const;

 private:
  // non-uniformity
  std::vector<float> mBCIntensityScales;
  ClassDefOverride(NonUniformMuInteractionSampler, 1);
};

} // namespace steer
} // namespace o2

#endif
