// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <bitset>
#include <vector>
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonConstants/LHCConstants.h"

namespace o2
{
namespace steer
{
class InteractionSampler
{
 public:
  static constexpr double Sec2NanoSec = 1.e9;   // s->ns conversion

  using BunchFilling = std::bitset<o2::constants::lhc::LHCMaxBunches>;

  o2::InteractionRecord generateCollisionTime();
  void generateCollisionTimes(std::vector<o2::InteractionRecord>& dest);

  void init();

  void setInteractionRate(double rateHz) { mIntRate = rateHz; }
  double getInteractionRate() const { return mIntRate; }
  void setMuPerBC(double mu) { mMuBC = mu; }
  double getMuPerBC() const { return mMuBC; }
  void setBCTimeRMS(double tNS = 0.2) { mBCTimeRMS = tNS; }
  double getBCTimeRMS() const { return mBCTimeRMS; }
  const BunchFilling& getBunchFilling() const { return mBCFilling; }
  BunchFilling& getBunchFilling() { return mBCFilling; }
  int getNCollidingBC() const { return mBCFilling.count(); }
  int getBCMin() const { return mBCMin; }
  int getBCMax() const { return mBCMax; }
  bool getBC(int bcID) const { return mBCFilling.test(bcID); }
  void setBC(int bcID, bool active = true);
  void setBCTrain(int nBC, int bcSpacing, int firstBC);
  void setBCTrains(int nTrains, int trainSpacingInBC, int nBC, int bcSpacing, int firstBC);
  void setDefaultBunchFilling();

  void print() const;
  void printBunchFilling(int bcPerLine = 100) const;

 protected:
  int simulateInteractingBC();
  int genPoissonZT();
  void nextCollidingBC();

  int mIntBCCache = 0;            ///< N interactions left for current BC
  int mBCCurrent = 0;             ///< current BC
  int mOrbit = 0;                 ///< current orbit
  int mPeriod = 0;                ///< current period
  int mBCMin = 0;                 ///< 1st filled BCID
  int mBCMax = -1;                ///< last filled BCID
  double mIntRate = -1.;          ///< total interaction rate in Hz
  double mBCTimeRMS = 0.2;        ///< BC time spread in NANOSECONDS
  double mMuBC = -1.;             ///< interaction probability per BC
  double mProbNoInteraction = 1.; ///< probability of BC w/o interaction
  double mMuBCZTRed = 0;          ///< reduced mu for fast zero-truncated Poisson derivation

  BunchFilling mBCFilling;       ///< patter of active BCs
  std::vector<double> mTimeInBC; ///< interaction times within single BC

  static constexpr double DefIntRate = 50e3; ///< default interaction rate

  ClassDefNV(InteractionSampler, 1);
};

//_________________________________________________
inline void InteractionSampler::generateCollisionTimes(std::vector<o2::InteractionRecord>& dest)
{
  // fill vector with interaction records
  dest.clear();
  for (int i = dest.capacity(); i--;) {
    dest.push_back(generateCollisionTime());
  }
}

//_________________________________________________
inline void InteractionSampler::nextCollidingBC()
{
  // increment bunch ID till next colliding bunch
  do {
    if (++mBCCurrent > mBCMax) { // did we exhaust full orbit?
      mBCCurrent = mBCMin;
      if (++mOrbit >= o2::constants::lhc::MaxNOrbits) { // did we exhaust full period?
        mOrbit = 0;
        mPeriod++;
      }
    }
  } while (!mBCFilling[mBCCurrent]);
}

//_________________________________________________
inline int InteractionSampler::genPoissonZT()
{
  // generate 0-truncated poisson number
  // https://en.wikipedia.org/wiki/Zero-truncated_Poisson_distribution
  int k = 1;
  double t = mMuBCZTRed, u = gRandom->Rndm(), s = t;
  while (s < u) {
    s += t *= mMuBC / (++k);
  }
  return k;
}
}
}

#endif
