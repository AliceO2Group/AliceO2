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

  int nBCSet = mBCFilling.count();
  if (!nBCSet) {
    LOG(WARNING) << "No bunch filling provided, impose default one" << FairLogger::endl;
    ;
    setDefaultBunchFilling();
    nBCSet = mBCFilling.count();
  }

  if (mMuBC < 0. && mIntRate < 0.) {
    LOG(WARNING) << "No IR or muBC is provided, setting default IR" << FairLogger::endl;
    ;
    mIntRate = DefIntRate;
  }

  if (mMuBC > 0.) {
    mIntRate = mMuBC * nBCSet * LHCRevFreq;
    LOG(INFO) << "Deducing IR=" << mIntRate << "Hz from " << nBCSet << " BCs at mu=" << mMuBC << FairLogger::endl;
  } else {
    mMuBC = mIntRate / (nBCSet * LHCRevFreq);
    LOG(INFO) << "Deducing mu=" << mMuBC << " per BC from IR=" << mIntRate << " with " << nBCSet << " BCs"
              << FairLogger::endl;
  }

  mBCMin = 0;
  mBCMax = -1;
  for (int i = 0; i < LHCBCSlots; i++) {
    if (!mBCFilling[i])
      continue;
    if (mBCMin > i)
      mBCMin = i;
    if (mBCMax < i)
      mBCMax = i;
  }
  double muexp = TMath::Exp(-mMuBC);
  mProbNoInteraction = 1. - muexp;
  mMuBCZTRed = mMuBC * muexp / mProbNoInteraction;
  mBCCurrent = mBCMin + gRandom->Integer(mBCMax - mBCMin + 1);
  mIntBCCache = 0;
  mOrbit = mPeriod = 0;
}

//_________________________________________________
void InteractionSampler::print() const
{
  if (mIntRate < 0) {
    LOG(ERROR) << "not yet initialized";
    return;
  }
  LOG(INFO) << "InteractionSampler with " << getNCollidingBC() << " colliding BCs in [" << mBCMin << '-' << mBCMax
            << "], mu(BC)= " << getMuPerBC() << " with " << getNCollidingBC()
            << " -> total IR= " << getInteractionRate() << FairLogger::endl;
  LOG(INFO) << "Current BC= " << mBCCurrent << '(' << mIntBCCache << " coll left) at orbit " << mOrbit << " period "
            << mPeriod << FairLogger::endl;
}

//_________________________________________________
void InteractionSampler::printBunchFilling(int bcPerLine) const
{
  bool endlOK = false;
  for (int i = 0; i < LHCBCSlots; i++) {
    printf("%c", mBCFilling[i] ? '+' : '-');
    if (((i + 1) % bcPerLine) == 0) {
      printf("\n");
      endlOK = true;
    } else {
      endlOK = false;
    }
  }
  if (!endlOK)
    printf("\n");
}
//_________________________________________________
void InteractionSampler::setBC(int bcID, bool active)
{
  // add interacting BC slot
  if (bcID >= LHCBCSlots) {
    LOG(FATAL) << "BCid is limited to " << mBCMin << '-' << mBCMax << FairLogger::endl;
  }
  mBCFilling.set(bcID, active);
}

//_________________________________________________
void InteractionSampler::setBCTrain(int nBC, int bcSpacing, int firstBC)
{
  // add interacting BC train with given spacing starting at given place, i.e.
  // train with 25ns spacing should have bcSpacing = 1
  for (int i = 0; i < nBC; i++) {
    setBC(firstBC);
    firstBC += bcSpacing;
  }
}

//_________________________________________________
void InteractionSampler::setBCTrains(int nTrains, int trainSpacingInBC, int nBC, int bcSpacing, int firstBC)
{
  // add nTrains trains of interacting BCs with bcSpacing within the train and trainSpacingInBC empty slots
  // between the trains
  for (int it = 0; it < nTrains; it++) {
    setBCTrain(nBC, bcSpacing, firstBC);
    firstBC += nBC * bcSpacing + trainSpacingInBC;
  }
}

//_________________________________________________
void InteractionSampler::setDefaultBunchFilling()
{
  // set BC filling a la TPC TDR, 12 50ns trains of 48 BCs
  // but instead of uniform train spacing we add 96empty BCs after each train
  setBCTrains(12, 96, 48, 2, 0);
}
//_________________________________________________
o2::MCInteractionRecord InteractionSampler::generateCollisionTime()
{
  // generate single interaction record
  if (mIntRate < 0)
    init();

  if (mIntBCCache < 1) {                   // do we still have interaction in current BC?
    mIntBCCache = simulateInteractingBC(); // decide which BC interacts and N collisions
  }
  double timeInt = mTimeInBC.back();
  timeInt += mBCCurrent * BCSpacingLHC + mOrbit * OrbitDuration;
  if (mPeriod)
    timeInt += mPeriod * PeriodDuration;
  mTimeInBC.pop_back();
  mIntBCCache--;
  return o2::MCInteractionRecord(timeInt, mBCCurrent, mOrbit, mPeriod);
}

//_________________________________________________
int InteractionSampler::simulateInteractingBC()
{
  // in order not to test random numbers for too many BC's with low mu,
  // we estimate from very beginning how many BC's happen w/o interaction
  // Returns number of collisions assigned to selected BC
  double prob = gRandom->Rndm();
  while (prob > 0.) {  // skip BCs w/o interaction
    nextCollidingBC(); // pick next interacting bunch
    prob -= mProbNoInteraction;
  }
  // once BC is decided, enforce at least one interaction
  int ncoll = genPoissonZT();
  // assign random time withing a bunch
  for (int i = ncoll; i--;) {
    mTimeInBC.push_back(gRandom->Gaus(mBCTimeRMS));
  }
  if (ncoll > 1) { // sort in DECREASING time order (we are reading vector from the end)
    std::sort(mTimeInBC.begin(), mTimeInBC.end(), [](const double a, const double b) { return a > b; });
  }
  return ncoll;
}
