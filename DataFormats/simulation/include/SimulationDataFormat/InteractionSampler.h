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
#include "MathUtils/RandomRing.h"

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

 protected:
  int simulateInteractingBC();
  void nextCollidingBC(int n);

  o2::math_utils::RandomRing<10000> mBCJumpGenerator;  // generator of random jumps in BC
  o2::math_utils::RandomRing<1000> mNCollBCGenerator;  // generator of number of interactions in BC
  o2::math_utils::RandomRing<1000> mCollTimeGenerator; // generator of number of interactions in BC

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

  ClassDefNV(InteractionSampler, 1);
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

} // namespace steer
} // namespace o2

#endif
