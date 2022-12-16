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

#include "SimulationDataFormat/InteractionSampler.h"
#include <fairlogger/Logger.h>

using namespace o2::steer;

//_________________________________________________
void InteractionSampler::init()
{
  // (re-)initialize sample and check parameters consistency

  int nBCSet = mBCFilling.getNBunches();
  if (!nBCSet) {
    LOG(warning) << "No bunch filling provided, impose default one";
    mBCFilling.setDefault();
    nBCSet = mBCFilling.getNBunches();
  }

  if (mMuBC < 0. && mIntRate < 0.) {
    LOG(warning) << "No IR or muBC is provided, setting default IR";
    mIntRate = DefIntRate;
  }

  if (mMuBC > 0.) {
    mIntRate = mMuBC * nBCSet * o2::constants::lhc::LHCRevFreq;
    LOG(info) << "Deducing IR=" << mIntRate << "Hz from " << nBCSet << " BCs at mu=" << mMuBC;
  } else {
    mMuBC = mIntRate / (nBCSet * o2::constants::lhc::LHCRevFreq);
    LOG(info) << "Deducing mu=" << mMuBC << " per BC from IR=" << mIntRate << " with " << nBCSet << " BCs";
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
  mBCJumpGenerator.initialize([mu]() { return (1. - std::log(1. - gRandom->Rndm()) / mu); });

  // Poisson distribution of number of collisions in the bunch excluding 0
  mNCollBCGenerator.initialize([mu]() {
    int n = 0;
    while ((n = gRandom->Poisson(mu)) == 0) {
      ;
    }
    return n;
  });

  auto trms = mBCTimeRMS;
  mCollTimeGenerator.initialize([trms]() {
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
    LOG(error) << "not yet initialized";
    return;
  }
  LOG(info) << "InteractionSampler with " << mInteractingBCs.size() << " colliding BCs, mu(BC)= "
            << getMuPerBC() << " -> total IR= " << getInteractionRate();
  LOG(info) << "Current " << mIR << '(' << mIntBCCache << " coll left)";
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

  nextCollidingBC(mBCJumpGenerator.getNextValue());
  // once BC is decided, enforce at least one interaction
  int ncoll = mNCollBCGenerator.getNextValue();

  // assign random time withing a bunch
  for (int i = ncoll; i--;) {
    mTimeInBC.push_back(mCollTimeGenerator.getNextValue());
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
  auto* bc = o2::BunchFilling::loadFrom(bcFillingFile, "ccdb_object");
  if (!bc) {
    bc = o2::BunchFilling::loadFrom(bcFillingFile); // retry with default naming in case of failure
  }
  if (!bc) {
    LOG(fatal) << "Failed to load bunch filling from " << bcFillingFile;
  }
  mBCFilling = *bc;
  delete bc;
}
