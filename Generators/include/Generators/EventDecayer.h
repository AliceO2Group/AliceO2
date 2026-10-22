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

#ifndef ALICEO2_EVENTGEN_EVENTDECAYER_H_
#define ALICEO2_EVENTGEN_EVENTDECAYER_H_

#include "Pythia8/Pythia.h"
#include <TParticle.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace o2
{
namespace eventgen
{

/// Counts of what one or more EventDecayer::decay calls did, per PDG code
struct EventDecayerReport {
  std::map<int, long> decayed;                               // particles decayed by Pythia8
  std::map<int, long> failed;                                // particles Pythia8 could not decay
  std::map<int, long> unknownInO2;                           // particles to transport unknown to O2DatabasePDG
  std::map<int, long> unknownInG4;                           // particles to transport unknown to Geant4
  std::map<int, std::pair<double, double>> lifetimeMismatch; // pdg -> {pythia ctau mm, g4 ctau mm}
  void merge(EventDecayerReport const& other);
  void print() const;
};

/// One entry of the Geant4 particle table
struct G4ParticleInfo {
  std::string name;
  double ctauMM;
  bool shortLived;
  bool hasDecayTable;
};

/// Reads the Geant4 particle table written by o2-sim-g4-particle-table
std::unordered_map<int, G4ParticleInfo> loadG4ParticleTable(std::string const& path);

/// Decays unstable generator particles with Pythia8 before transport, keeping the full chain
class EventDecayer
{
 public:
  bool init();
  /// Sets the caller's units in cm, s, GeV/c and GeV, as the unit setters of Generator (default cm, s, GeV/c, GeV)
  void setUnits(double positionUnit, double timeUnit, double momentumUnit = 1., double energyUnit = 1.)
  {
    mPositionUnit = positionUnit;
    mTimeUnit = timeUnit;
    mMomentumUnit = momentumUnit;
    mEnergyUnit = energyUnit;
  }
  EventDecayerReport decay(std::vector<TParticle>& particles);
  std::string configSummary() const;
  bool vertexLimitsActive() const;
  int seed() const { return mSeed; }

 private:
  bool decayable(int pdg) const;
  bool decayOne(std::vector<TParticle>& particles, int index);
  void checkParticles(std::vector<TParticle> const& particles, EventDecayerReport& report) const;
  void checkLifetimes(EventDecayerReport& report) const;

  mutable ::Pythia8::Pythia mPythia;                    // mutable since Pythia8 settings lookups are not const
  std::vector<std::string> mConfigs;                    // expanded config paths
  std::unordered_map<int, G4ParticleInfo> mG4Particles; // Geant4 particle table
  bool mHasG4Table = false;
  bool mVerbose = false;
  double mPositionUnit = 1.; // caller's position unit in cm
  double mTimeUnit = 1.;     // caller's time unit in s
  double mMomentumUnit = 1.; // caller's momentum unit in GeV/c
  double mEnergyUnit = 1.;   // caller's energy unit in GeV
  double mTau0Max = 0.;      // tau0 limit in mm/c in use after init, 0 when off
  int mSeed = 0;             // Pythia8 seed in use after init
};

} // namespace eventgen
} // namespace o2

#endif // ALICEO2_EVENTGEN_EVENTDECAYER_H_
