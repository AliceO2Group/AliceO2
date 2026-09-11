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

/// \file GeneratorKrDecay.h
/// \brief Generator for 83mKr decays, for TPC gain-map calibration simulation
/// \author Ankur Yadav <ankur.yadav@cern.ch>

#ifndef ALICEO2_TPC_GeneratorKrDecay_H_
#define ALICEO2_TPC_GeneratorKrDecay_H_

#include "Generators/Generator.h"
#include <array>
#include <memory>
#include <vector>

namespace o2
{
namespace tpc
{

// O2 status encoding from MCGenProperties.h
// bits 0-8: hepmc(9), bits 9-18: gen(10), bits 19-28: reserved(10), bits 29-31: sentinel=5
inline int krO2EncodedStatus(int hepmc, int gen = 0)
{
  return (5 << 29) | ((gen & 0x3FF) << 9) | (hepmc & 0x1FF);
}

/// Table of 83mKr internal-conversion/gamma decay channels and their
/// branching fractions, derived at runtime from Geant4's level/gamma data
/// (falls back to hardcoded PhotonEvaporation5.7/z36.a83 values if
/// unavailable). Used by GeneratorKrDecay to sample one decay channel per
/// generated vertex.
class KrDecayTable
{
 public:
  struct Product {
    int pdg;
    double eKin;
  };
  struct Channel {
    double fraction;
    int nProducts;
    Product products[6]; // max 5 used; 6 for safety
  };
  // Eight physically motivated channels (T1 mode x T2 mode):
  //   T1: ICC_total=2035 -> 99.951% IC (75.163% outer-shell, 24.788% K-shell), 0.049% gamma
  //       K-shell: 65.2% K-fluorescence (Kalpha), 34.8% K-Auger
  //   T2: ICC_total=17.09 -> 94.472% IC, 5.528% gamma
  //   Source: G4 PhotonEvaporation5.7/z36.a83, RadioactiveDecay5.6/z36.a83
  static const int kNChannels = 8;
  Channel channels[kNChannels];
  double cumulative[kNChannels];
  KrDecayTable();
  const Channel& sample() const;
};

} // namespace tpc
} // namespace o2

namespace o2
{
namespace eventgen
{

/// FairGenerator producing 83mKr decay vertices uniformly distributed in the
/// TPC drift volume, for gain-map/energy-resolution calibration simulation.
/// Each vertex emits the conversion electrons/photons of one randomly
/// sampled o2::tpc::KrDecayTable::Channel.
class GeneratorKrDecay : public Generator
{
 public:
  GeneratorKrDecay();
  ~GeneratorKrDecay() override;
  Bool_t Init() override;
  Bool_t generateEvent() override;
  Bool_t importParticles() override;

 private:
  static constexpr double kRInner = 83.5;
  static constexpr double kROuter = 246.5; // TPC outermost pad row outer edge ~247 cm; stay inside
  static constexpr double kHalfZ = 249.7;

  int mNPerEvent = 1000; // Kr decays per event; overridable at runtime via KR_N_PER_EVENT env var
  std::unique_ptr<o2::tpc::KrDecayTable> mTable;
  std::vector<std::array<double, 3>> mVertices;
  // No ClassDefOverride: the base Generator class's dictionary is sufficient
  // for a runtime-only generator that is never streamed via ROOT I/O
  // (matches o2::eventgen::GeneratorGeantinos and other Generator subclasses).
};

} // namespace eventgen
} // namespace o2

#endif // ALICEO2_TPC_GeneratorKrDecay_H_
