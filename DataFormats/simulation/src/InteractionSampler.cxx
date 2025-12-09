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
int FixedSkipBC_InteractionSampler::simulateInteractingBC()
{
  // Returns number of collisions assigned to selected BC

  nextCollidingBC(mEveryN);  // we jump regular intervals
  int ncoll = mMultiplicity; // well defined pileup

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

// ________________________________________________
bool NonUniformMuInteractionSampler::setBCIntensityScales(const std::vector<float>& scales_from_vector)
{
  // Sets the intensity scales per bunch crossing index
  // The length of this vector needs to be compatible with the bunch filling chosen
  mBCIntensityScales = scales_from_vector;

  if (scales_from_vector.size() != mInteractingBCs.size()) {
    LOG(error) << "Scaling factors and bunch filling scheme are not compatible. Not doing anything";
    return false;
  }

  float sum = 0.;
  for (auto v : mBCIntensityScales) {
    sum += std::abs(v);
  }
  if (sum == 0) {
    LOGP(warn, "total intensity is 0, assuming uniform");
    for (auto& v : mBCIntensityScales) {
      v = 1.f;
    }
  } else { // normalize
    float norm = mBCIntensityScales.size() / sum;
    for (auto& v : mBCIntensityScales) {
      v = std::abs(v) * norm;
    }
  }
  return false;
}

// ________________________________________________

bool NonUniformMuInteractionSampler::setBCIntensityScales(const TH1F& hist)
{
  return setBCIntensityScales(determineBCIntensityScalesFromHistogram(hist));
}

std::vector<float> NonUniformMuInteractionSampler::determineBCIntensityScalesFromHistogram(const TH1F& hist)
{
  if (mInteractingBCs.size() == 0) {
    LOG(error) << " Initialize bunch crossing scheme before assigning scales";
  }
  std::vector<float> scales;
  // we go through the BCs and query the count from histogram
  for (auto bc : mInteractingBCs) {
    scales.push_back(hist.GetBinContent(bc + 1));
  }
  return scales;
}

int NonUniformMuInteractionSampler::getBCJump() const
{
  auto muFunc = [this](int bc_position) {
    return mBCIntensityScales[bc_position % mInteractingBCs.size()] * mMuBC;
  };

  double U = gRandom->Rndm();    // uniform (0,1)
  double T = -std::log(1.0 - U); // threshold
  double sumMu = 0.0;
  int offset = 0;
  auto bcStart = mCurrBCIdx; // the current bc

  while (sumMu < T) {
    auto mu_here = muFunc(bcStart + offset); // mu at next BC
    sumMu += mu_here;
    if (sumMu >= T) {
      break; // found BC with at least one collision
    }
    ++offset;
  }
  return offset;
}

int NonUniformMuInteractionSampler::simulateInteractingBC()
{
  nextCollidingBC(getBCJump());

  auto muFunc = [this](int bc_position) {
    return mBCIntensityScales[bc_position % mInteractingBCs.size()] * mMuBC;
  };

  // now sample number of collisions in chosenBC, conditioned >=1:
  double mu_chosen = muFunc(mCurrBCIdx); // or does it need to be mCurrBCIdx
  int ncoll = 0;
  do {
    ncoll = gRandom->Poisson(mu_chosen);
  } while (ncoll == 0);

  // assign random time withing a bunch
  for (int i = ncoll; i--;) {
    mTimeInBC.push_back(mCollTimeGenerator.getNextValue());
  }
  if (ncoll > 1) { // sort in DECREASING time order (we are reading vector from the end)
    std::sort(mTimeInBC.begin(), mTimeInBC.end(), [](const float a, const float b) { return a > b; });
  }
  return ncoll;
}