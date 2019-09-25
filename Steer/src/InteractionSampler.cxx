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
#include <FairLogger.h>

using namespace o2::steer;

//_________________________________________________
void InteractionSampler::init()
{
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

  mBCMin = 0;
  mBCMax = -1;
  for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
    if (mBCFilling.testBC(i)) {
      if (mBCMin > i) {
        mBCMin = i;
      }
      if (mBCMax < i) {
        mBCMax = i;
      }
    }
  }
  double muexp = TMath::Exp(-mMuBC);
  mProbInteraction = 1. - muexp;
  mMuBCZTRed = mMuBC * muexp / mProbInteraction;
  mBCCurrent = mBCMin + gRandom->Integer(mBCMax - mBCMin + 1);
  mIntBCCache = 0;
  mOrbit = 0;
}

//_________________________________________________
void InteractionSampler::print() const
{
  if (mIntRate < 0) {
    LOG(ERROR) << "not yet initialized";
    return;
  }
  LOG(INFO) << "InteractionSampler with " << mBCFilling.getNBunches() << " colliding BCs in [" << mBCMin
            << '-' << mBCMax << "], mu(BC)= " << getMuPerBC() << " -> total IR= " << getInteractionRate();
  LOG(INFO) << "Current BC= " << mBCCurrent << '(' << mIntBCCache << " coll left) at orbit " << mOrbit;
}

//_________________________________________________
o2::InteractionTimeRecord InteractionSampler::generateCollisionTime()
{
  // generate single interaction record
  if (mIntRate < 0) {
    init();
  }

  if (mIntBCCache < 1) {                   // do we still have interaction in current BC?
    mIntBCCache = simulateInteractingBC(); // decide which BC interacts and N collisions
  }
  double timeInt = mTimeInBC.back() + o2::InteractionTimeRecord::bc2ns(mBCCurrent, mOrbit);
  mTimeInBC.pop_back();
  mIntBCCache--;

  o2::InteractionTimeRecord tmp(timeInt);

  return o2::InteractionTimeRecord(timeInt);
}

//_________________________________________________
int InteractionSampler::simulateInteractingBC()
{
  // Returns number of collisions assigned to selected BC

  do {
    nextCollidingBC();                          // pick next interacting bunch
  } while (gRandom->Rndm() > mProbInteraction); // skip BCs w/o collisions

  // once BC is decided, enforce at least one interaction
  int ncoll = genPoissonZT();
  // assign random time withing a bunch
  for (int i = ncoll; i--;) {
    double tInBC = 0; // tInBC should be in the vicinity of the BC
    do {
      tInBC = gRandom->Gaus(0., mBCTimeRMS);
    } while (std::abs(tInBC) > o2::constants::lhc::LHCBunchSpacingNS / 2.1);
    mTimeInBC.push_back(tInBC);
  }
  if (ncoll > 1) { // sort in DECREASING time order (we are reading vector from the end)
    std::sort(mTimeInBC.begin(), mTimeInBC.end(), [](const float a, const float b) { return a > b; });
  }
  return ncoll;
}

//_________________________________________________
void InteractionSampler::warnOrbitWrapped() const
{
  /// in run3 the orbit is 32 bits and should never wrap
  LOG(WARN) << "Orbit wraps, current state of InteractionSampler:";
  print();
}
