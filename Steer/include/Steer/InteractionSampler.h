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
#include <vector>
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/BunchFilling.h"
#include "CommonConstants/LHCConstants.h"

namespace o2
{
namespace steer
{
class InteractionSampler
{
 public:
  static constexpr float Sec2NanoSec = 1.e9; // s->ns conversion
  static constexpr int FirstOrbit = 1;       // start from orbit > 0 to avoid problems with negative BCs
  const o2::InteractionTimeRecord& generateCollisionTime();
  void generateCollisionTimes(std::vector<o2::InteractionTimeRecord>& dest);

  void init();

  void setInteractionRate(float rateHz) { mIntRate = rateHz; }
  float getInteractionRate() const { return mIntRate; }
  void setMuPerBC(float mu) { mMuBC = mu; }
  float getMuPerBC() const { return mMuBC; }
  void setBCTimeRMS(float tNS = 0.2) { mBCTimeRMS = tNS; }
  float getBCTimeRMS() const { return mBCTimeRMS; }
  const BunchFilling& getBunchFilling() const { return mBCFilling; }
  BunchFilling& getBunchFilling() { return mBCFilling; }
  void setBunchFilling(const BunchFilling& bc) { mBCFilling = bc; }
  void setBunchFilling(const std::string& bcFillingFile);
  int getBCMin() const { return mBCMin; }
  int getBCMax() const { return mBCMax; }

  void print() const;

 protected:
  int simulateInteractingBC();
  int genPoissonZT();
  void nextCollidingBC();
  void warnOrbitWrapped() const;

  o2::InteractionTimeRecord mIR{{0, FirstOrbit}, 0.};
  int mIntBCCache = 0;         ///< N interactions left for current BC
  int mBCMin = 0;              ///< 1st filled BCID
  int mBCMax = -1;             ///< last filled BCID
  float mIntRate = -1.;        ///< total interaction rate in Hz
  float mBCTimeRMS = 0.2;      ///< BC time spread in NANOSECONDS
  float mMuBC = -1.;           ///< interaction probability per BC
  float mProbInteraction = 1.; ///< probability of non-0 interactions at per BC
  float mMuBCZTRed = 0;        ///< reduced mu for fast zero-truncated Poisson derivation

  o2::BunchFilling mBCFilling;  ///< patter of active BCs
  std::vector<float> mTimeInBC; ///< interaction times within single BC

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
inline void InteractionSampler::nextCollidingBC()
{
  // increment bunch ID till next colliding bunch
  do {
    if (mIR.bc >= mBCMax) { // did we exhaust full orbit?
      mIR.bc = mBCMin;
      if (mIR.orbit == o2::constants::lhc::MaxNOrbits) { // wrap orbit (should not happen in run3)
        warnOrbitWrapped();
      }
      mIR.orbit++;
    } else {
      mIR.bc++;
    }
  } while (!mBCFilling.testBC(mIR.bc));
}

//_________________________________________________
inline int InteractionSampler::genPoissonZT()
{
  // generate 0-truncated poisson number
  // https://en.wikipedia.org/wiki/Zero-truncated_Poisson_distribution
  int k = 1;
  if (mMuBCZTRed > 0) {
    double t = mMuBCZTRed, u = gRandom->Rndm(), s = t;
    while (s < u) {
      s += t *= mMuBC / (++k);
    }
  } else { // need to generate explicitly since the prob of
    while (!(k = gRandom->Poisson(mMuBC))) {
    };
  }
  return k;
}
} // namespace steer
} // namespace o2

#endif
