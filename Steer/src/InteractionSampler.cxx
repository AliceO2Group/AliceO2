// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Steer/InteractionSampler.h"
#include "MathUtils/RandomRing.h"
#include <FairLogger.h>

using namespace o2::steer;

namespace o2::steer{
struct InteractionSamplerContext {
  o2::math_utils::RandomRing<10000> mBCJumpGenerator;  // generator of random jumps in BC
  o2::math_utils::RandomRing<1000> mNCollBCGenerator;  // generator of number of interactions in BC
  o2::math_utils::RandomRing<1000> mCollTimeGenerator; // generator of number of interactions in BC
};
}

InteractionSampler::~InteractionSampler() {
   delete mContext;
}
//_________________________________________________
void InteractionSampler::init()
{
  mContext = new InteractionSamplerContext{};
  // (re-)initialize sample and check parameters consistency

  int nBCSet = mBCFilling.getNBunches();
  if (!nBCSet) {
    LOG(WARNING) << "No bunch filling provided, impose default one";
    mBCFilling.setDefault();
    nBCSet = mBCFilling.getNBunches();
  }

  if (mMuBC < 0. && mIntRate < 0.) {
    LOG(WARNING) << "No IR or muBC is provided, setting default IR";
    mIntRate = DefIntRate;
  }

  if (mMuBC > 0.) {
    mIntRate = mMuBC * nBCSet * o2::constants::lhc::LHCRevFreq;
    LOG(INFO) << "Deducing IR=" << mIntRate << "Hz from " << nBCSet << " BCs at mu=" << mMuBC;
  } else {
    mMuBC = mIntRate / (nBCSet * o2::constants::lhc::LHCRevFreq);
    LOG(INFO) << "Deducing mu=" << mMuBC << " per BC from IR=" << mIntRate << " with " << nBCSet << " BCs";
  }

  mInteractingBCs.clear();
  mInteractingBCs.reserve(nBCSet);
  for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
    if (mBCFilling.testBC(i)) {
      mInteractingBCs.push_back(i);
    }
  }

  auto mu = mMuBC;
  // prob. of not having interaction in N consecutive BCs is P(N) = mu*exp(-(N-1)*mu), hence its cumulative distribution
  // is T(N) = integral_1^N {P(N)} = 1. - exp(-(N-1)*mu)
  // We generate random N using its inverse, N = 1 - log(1 - Rndm)/mu
  mContext->mBCJumpGenerator.initialize([mu]() { return (1. - std::log(1. - gRandom->Rndm()) / mu); });

  // Poisson distribution of number of collisions in the bunch excluding 0
  mContext->mNCollBCGenerator.initialize([mu]() {
    int n = 0;
    while ((n = gRandom->Poisson(mu)) == 0) {
      ;
    }
    return n;
  });

  auto trms = mBCTimeRMS;
  mContext->mCollTimeGenerator.initialize([trms]() {
    float t; // make sure it does not go outside half bunch
    while (std::abs(t = gRandom->Gaus(0, trms)) > o2::constants::lhc::LHCBunchSpacingNS / 2.1) {
      ;
    }
    return t;
  });

  mIntBCCache = 0;
  mCurrBCIdx = 0;
  mIR = mFirstIR;
  while (mCurrBCIdx < mInteractingBCs.size() && mInteractingBCs[mCurrBCIdx] < mIR.bc) {
    mCurrBCIdx++;
  }
  // set the "current BC" right in front of the 1st BC to generate. There will be a jump by at least 1 during generation
  mCurrBCIdx--;
}

//_________________________________________________
void InteractionSampler::print() const
{
  if (mIntRate < 0) {
    LOG(ERROR) << "not yet initialized";
    return;
  }
  LOG(INFO) << "InteractionSampler with " << mInteractingBCs.size() << " colliding BCs, mu(BC)= "
            << getMuPerBC() << " -> total IR= " << getInteractionRate();
  LOG(INFO) << "Current " << mIR << '(' << mIntBCCache << " coll left)";
}

//_________________________________________________
const o2::InteractionTimeRecord& InteractionSampler::generateCollisionTime()
{
  // generate single interaction record
  if (mIntRate < 0) {
    init();
  }

  if (mIntBCCache < 1) {                   // do we still have interaction in current BC?
    mIntBCCache = simulateInteractingBC(); // decide which BC interacts and N collisions
  }
  mIR.timeInBCNS = mTimeInBC.back();
  mTimeInBC.pop_back();
  mIntBCCache--;

  return mIR;
}

//_________________________________________________
int InteractionSampler::simulateInteractingBC()
{
  // Returns number of collisions assigned to selected BC

  nextCollidingBC(mContext->mBCJumpGenerator.getNextValue());
  // once BC is decided, enforce at least one interaction
  int ncoll = mContext->mNCollBCGenerator.getNextValue();

  // assign random time withing a bunch
  for (int i = ncoll; i--;) {
    mTimeInBC.push_back(mContext->mCollTimeGenerator.getNextValue());
  }
  if (ncoll > 1) { // sort in DECREASING time order (we are reading vector from the end)
    std::sort(mTimeInBC.begin(), mTimeInBC.end(), [](const float a, const float b) { return a > b; });
  }
  return ncoll;
}

//_________________________________________________
void InteractionSampler::setBunchFilling(const std::string& bcFillingFile)
{
  // load bunch filling from the file
  auto* bc = o2::BunchFilling::loadFrom(bcFillingFile);
  if (!bc) {
    LOG(FATAL) << "Failed to load bunch filling from " << bcFillingFile;
  }
  mBCFilling = *bc;
  delete bc;
}
