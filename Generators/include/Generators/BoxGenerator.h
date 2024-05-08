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

/// \author Sandro Wenzel - April 2024

#ifndef ALICEO2_EVENTGEN_BOX
#define ALICEO2_EVENTGEN_BOX

#include "Generators/Generator.h"
#include "TParticle.h"
#include <vector>

namespace o2::eventgen
{

/*
 * A simple mono-pdg "BoxGenerator". Or particle gun.
 * Re-implements FairBoxGenerator for more convenient O2-processing.
 */
class BoxGenerator : public Generator
{
 public:
  BoxGenerator() = default;
  BoxGenerator(int pdgid, int mult = 1);

  BoxGenerator(int pdgid,
               int mult,
               double etamin,
               double etamax,
               double pmin,
               double pmax,
               double phimin,
               double phimax) : mPDG{pdgid}, mMult{mult}
  {
    SetEtaRange(etamin, etamax);
    SetPRange(pmin, pmax);
    SetPhiRange(phimin, phimax);
  }

  void SetPRange(Double32_t pmin = 0, Double32_t pmax = 10)
  {
    mPMin = pmin;
    mPMax = pmax;
    mPRangeIsSet = true;
  }

  void SetPhiRange(double phimin = 0, double phimax = 360)
  {
    mPhiMin = phimin;
    mPhiMax = phimax;
  }

  void SetEtaRange(double etamin = -5, double etamax = 5)
  {
    mEtaMin = etamin;
    mEtaMax = etamax;
    mEtaRangeIsSet = true;
  }

  /// generates a single particle conforming to particle gun parameters
  TParticle sampleParticle() const;

  /// implements the main O2 generator interfaces
  bool generateEvent() override
  {
    mEvent.clear();
    for (int i = 0; i < mMult; ++i) {
      mEvent.push_back(sampleParticle());
    }
    return true;
  }
  bool importParticles() override
  {
    mParticles.clear();
    std::copy(mEvent.begin(), mEvent.end(), std::back_insert_iterator(mParticles));
    return true;
  }

 private:
  double mPtMin{0.}, mPtMax{0.};       // Transverse momentum range [GeV]
  double mPhiMin{0.}, mPhiMax{360.};   // Azimuth angle range [degree]
  double mEtaMin{0.}, mEtaMax{0.};     // Pseudorapidity range in lab system
  double mYMin{0.}, mYMax{0.};         // Rapidity range in lab system
  double mPMin{0.}, mPMax{0.};         // Momentum range in lab system
  double mThetaMin{0.}, mThetaMax{0.}; // Polar angle range in lab system [degree]
  double mEkinMin{0.}, mEkinMax{0.};   // Kinetic Energy range in lab system [GeV]

  int mPDG{0};
  int mMult{1};

  bool mEtaRangeIsSet{false};   // True if eta range is set
  bool mYRangeIsSet{false};     // True if rapidity range is set
  bool mThetaRangeIsSet{false}; // True if theta range is set
  bool mCosThetaIsSet{false};   // True if uniform distribution in
  // cos(theta) is set (default -> not set)
  bool mPtRangeIsSet{false};   // True if transverse momentum range is set
  bool mPRangeIsSet{false};    // True if abs.momentum range is set
  bool mEkinRangeIsSet{false}; // True if kinetic energy range is set

  std::vector<TParticle> mEvent; // internal event container
};

} // namespace o2::eventgen

#endif
