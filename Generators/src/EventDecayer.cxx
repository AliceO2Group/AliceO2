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

#include "Generators/EventDecayer.h"
#include "Generators/EventDecayerParam.h"
#include "Generators/DecayerPythia8Param.h"
#include "SimulationDataFormat/MCGenProperties.h"
#include "SimulationDataFormat/MCUtils.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include <fairlogger/Logger.h>
#include <TMCProcess.h>
#include <TRandom.h>
#include <TString.h>
#include <TSystem.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace o2
{
namespace eventgen
{

namespace
{
// speed of light in mm/s, to convert Pythia times (mm/c) to seconds
constexpr double kSpeedOfLightMMPerS = 2.99792458e11;

// largest seed Pythia8 accepts for Random:seed
constexpr long kMaxSeed = 900000000;

// mixed into a seed taken from gRandom, so that it differs from the one of GeneratorPythia8
constexpr unsigned int kGRandomSeedXor = 0x5eed0d3c;

// Pythia settings that cut decays on the decay vertex, not supported here
const char* const kVertexLimits[] = {"ParticleDecays:limitTau", "ParticleDecays:limitRadius", "ParticleDecays:limitCylinder"};

std::string expand(std::string const& path)
{
  TString expanded = path;
  gSystem->ExpandPathName(expanded);
  return expanded.Data();
}

// sets the Pythia8 seed as GeneratorPythia8::seedGenerator, a seed of 0 derives it from gRandom
bool setSeed(::Pythia8::Pythia& pythia, long seed)
{
  if (seed < 0) {
    LOG(error) << "Failed to init \'EventDecayer\': negative seed " << seed;
    return false;
  }
  if (seed == 0) {
    seed = gRandom->TRandom::GetSeed() ^ kGRandomSeedXor; // from the seed set by o2-sim --seed
    LOG(info) << "EventDecayer: using random seed from gRandom ^ " << kGRandomSeedXor << " % " << kMaxSeed + 1;
  }
  seed = seed % (kMaxSeed + 1);
  if (!pythia.readString("Random:setSeed on") || !pythia.readString("Random:seed " + std::to_string(seed))) {
    LOG(error) << "Failed to init \'EventDecayer\': cannot set the Pythia8 seed " << seed;
    return false;
  }
  return true;
}

// parses one skipPdg entry into |pdg|, false if it is not an integer
bool parsePdg(std::string const& code, int& pdg)
{
  try {
    size_t used = 0;
    pdg = std::abs(std::stoi(code, &used));
    return used == code.size();
  } catch (std::exception const&) {
    return false;
  }
}

// switches off the decays of the species listed in skipPdg
bool skipDecays(::Pythia8::ParticleData& particleData, std::string const& skipPdg)
{
  std::istringstream skip(skipPdg);
  std::string code;
  while (skip >> code) {
    int pdg = 0;
    if (!parsePdg(code, pdg)) {
      LOG(error) << "Failed to init \'EventDecayer\': skipPdg entry " << code << " is not a PDG code";
      return false;
    }
    if (particleData.isParticle(pdg)) {
      particleData.mayDecay(pdg, false);
    } else {
      LOG(info) << "EventDecayer: skipPdg entry " << pdg << " is unknown to Pythia8, so never decayed anyway";
    }
  }
  return true;
}
} // namespace

/*****************************************************************/

void EventDecayerReport::merge(EventDecayerReport const& other)
{
  auto add = [](std::map<int, long>& to, std::map<int, long> const& from) {
    for (auto const& [pdg, count] : from) {
      to[pdg] += count;
    }
  };
  add(decayed, other.decayed);
  add(failed, other.failed);
  add(unknownInO2, other.unknownInO2);
  add(unknownInG4, other.unknownInG4);
  for (auto const& [pdg, lifetimes] : other.lifetimeMismatch) {
    lifetimeMismatch[pdg] = lifetimes;
  }
}

void EventDecayerReport::print() const
{
  for (auto const& [pdg, count] : decayed) {
    LOG(info) << "EventDecayer: decayed pdg " << pdg << ": " << count;
  }
  for (auto const& [pdg, count] : failed) {
    LOG(warn) << "EventDecayer: Pythia8 failed to decay pdg " << pdg << ": " << count;
  }
  for (auto const& [pdg, count] : unknownInO2) {
    LOG(warn) << "EventDecayer: pdg " << pdg << " left for transport is unknown to O2DatabasePDG: " << count;
  }
  for (auto const& [pdg, count] : unknownInG4) {
    LOG(warn) << "EventDecayer: pdg " << pdg << " left for transport is not in the Geant4 table (o2-sim may define it at runtime): " << count;
  }
  for (auto const& [pdg, lifetimes] : lifetimeMismatch) {
    LOG(warn) << "EventDecayer: lifetime of pdg " << pdg << " differs between Pythia8 (ctau " << lifetimes.first
              << " mm) and Geant4 (ctau " << lifetimes.second << " mm)";
  }
}

/*****************************************************************/

std::unordered_map<int, G4ParticleInfo> loadG4ParticleTable(std::string const& path)
{
  std::unordered_map<int, G4ParticleInfo> table;
  std::ifstream in(path);
  if (!in) {
    LOG(error) << "EventDecayer: cannot read Geant4 particle table " << path;
    return table;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream fields(line);
    int pdg = 0;
    G4ParticleInfo info;
    if (!(fields >> pdg >> info.name >> info.ctauMM >> info.shortLived >> info.hasDecayTable)) {
      LOG(warn) << "EventDecayer: skipping malformed line in " << path << ": " << line;
      continue;
    }
    table[pdg] = info;
  }
  return table;
}

/*****************************************************************/

bool EventDecayer::init()
{
  mPythia.readString("ProcessLevel:all off");

  auto& param = EventDecayerParam::Instance();
  LOG(info) << "Init \'EventDecayer\' with following parameters";
  LOG(info) << param;

  // before the config files, so that they can override it
  if (!setSeed(mPythia, param.seed)) {
    return false;
  }

  mConfigs.clear();
  auto& fallback = DecayerPythia8Param::Instance();
  for (int i = 0; i < 8; ++i) {
    auto const& entry = param.config[i].empty() ? fallback.config[i] : param.config[i];
    if (entry.empty()) {
      continue;
    }
    auto config = expand(entry);
    LOG(info) << "Reading configuration from file: " << config;
    if (!mPythia.readFile(config, true)) {
      LOG(error) << "Failed to init \'EventDecayer\': problems with configuration file " << config;
      return false;
    }
    mConfigs.push_back(config);
  }

  for (auto name : kVertexLimits) {
    if (mPythia.settings.flag(name)) {
      LOG(error) << "EventDecayer does not support " << name << ", switching it off";
      mPythia.settings.flag(name, false);
    }
  }
  if (param.tau0Max > 0.) {
    mPythia.settings.flag("ParticleDecays:limitTau0", true);
    mPythia.settings.parm("ParticleDecays:tau0Max", param.tau0Max);
  }

  if (!skipDecays(mPythia.particleData, param.skipPdg)) {
    return false;
  }

  mVerbose = param.verbose;
  mPythia.readString(param.showChanged ? "Init:showChangedParticleData on" : "Init:showChangedParticleData off");

  if (!mPythia.init()) {
    LOG(error) << "Failed to init \'EventDecayer\': init returned with error";
    return false;
  }
  mSeed = mPythia.settings.mode("Random:seed");
  LOG(info) << "EventDecayer: Pythia8 seed " << mSeed;
  mTau0Max = mPythia.settings.flag("ParticleDecays:limitTau0") ? mPythia.settings.parm("ParticleDecays:tau0Max") : 0.;
  if (mTau0Max > 0.) {
    LOG(info) << "EventDecayer: species with tau0 above " << mTau0Max << " mm/c stay undecayed";
  }

  mG4Particles.clear();
  mHasG4Table = !param.g4ParticleTable.empty();
  if (mHasG4Table) {
    auto path = expand(param.g4ParticleTable);
    mG4Particles = loadG4ParticleTable(path);
    if (mG4Particles.empty()) {
      LOG(error) << "Failed to init \'EventDecayer\': no particles in Geant4 particle table " << path;
      return false;
    }
    LOG(info) << "EventDecayer: read " << mG4Particles.size() << " particles from Geant4 particle table " << path;
  } else {
    LOG(info) << "EventDecayer: no Geant4 particle table given, skipping the Geant4 PDG and lifetime check";
  }
  return true;
}

/*****************************************************************/

std::string EventDecayer::configSummary() const
{
  std::string summary;
  for (auto const& config : mConfigs) {
    summary += (summary.empty() ? "" : ";") + config;
  }
  return summary;
}

bool EventDecayer::vertexLimitsActive() const
{
  for (auto name : kVertexLimits) {
    if (mPythia.settings.flag(name)) {
      return true;
    }
  }
  return false;
}

/*****************************************************************/

bool EventDecayer::decayable(int pdg) const
{
  auto& particleData = mPythia.particleData;
  return particleData.mayDecay(pdg) && particleData.canDecay(pdg) && !(mTau0Max > 0. && particleData.tau0(pdg) > mTau0Max);
}

EventDecayerReport EventDecayer::decay(std::vector<TParticle>& particles)
{
  EventDecayerReport report;
  const int nOriginal = particles.size();
  for (int i = 0; i < nOriginal; ++i) {
    auto const& particle = particles[i];
    auto pdg = particle.GetPdgCode();
    if (!particle.TestBit(ParticleStatus::kToBeDone) || particle.GetUniqueID() != kPPrimary || !decayable(pdg)) {
      continue;
    }
    if (decayOne(particles, i)) {
      report.decayed[pdg]++;
    } else {
      report.failed[pdg]++;
    }
  }

  checkParticles(particles, report);
  checkLifetimes(report);
  return report;
}

/*****************************************************************/

bool EventDecayer::decayOne(std::vector<TParticle>& particles, int index)
{
  // a copy, since appending to particles may reallocate
  const TParticle mother = particles[index];
  auto& event = mPythia.event;
  event.clear();
  // the caller's units to GeV/c and GeV
  const double px = mother.Px() * mMomentumUnit;
  const double py = mother.Py() * mMomentumUnit;
  const double pz = mother.Pz() * mMomentumUnit;
  const double e = mother.Energy() * mEnergyUnit;
  const double m2 = e * e - px * px - py * py - pz * pz;
  event.append(mother.GetPdgCode(), 1, 0, 0, px, py, pz, e, m2 > 0. ? std::sqrt(m2) : -std::sqrt(-m2));
  event[0].vProd(0., 0., 0., 0.);
  event[0].tau(event[0].tau0() * mPythia.rndm.exp());
  if (!mPythia.moreDecays() || event[0].isFinal()) {
    return false;
  }
  if (mVerbose) {
    event.list();
  }
  // partons (meMode 91-100) or a product that Pythia failed to decay make the whole decay fail
  for (int k = 1; k < event.size(); ++k) {
    auto const& entry = event[k];
    if (entry.isFinal() && (entry.isParton() || decayable(entry.id()))) {
      return false;
    }
  }

  // Pythia entry 0 is the original particle, entry k > 0 goes to offset + k - 1
  const int offset = particles.size();
  auto toIndex = [index, offset](int k) { return k == 0 ? index : offset + k - 1; };
  // Pythia mm and mm/c to the caller's units
  const double toPosition = 0.1 / mPositionUnit;
  const double toTime = 1. / (kSpeedOfLightMMPerS * mTimeUnit);

  for (int k = 1; k < event.size(); ++k) {
    auto const& entry = event[k];
    int mother1 = toIndex(entry.mother1());
    int mother2 = (entry.mother2() > 0 && entry.mother2() != entry.mother1()) ? toIndex(entry.mother2()) : -1;
    int daughter1 = entry.daughter1() > 0 ? toIndex(entry.daughter1()) : -1;
    int daughter2 = entry.daughter2() > 0 ? toIndex(entry.daughter2()) : daughter1;
    TParticle product(entry.id(), 0, mother1, mother2, daughter1, daughter2,
                      entry.px() / mMomentumUnit, entry.py() / mMomentumUnit, entry.pz() / mMomentumUnit, entry.e() / mEnergyUnit,
                      mother.Vx() + entry.xProd() * toPosition, mother.Vy() + entry.yProd() * toPosition,
                      mother.Vz() + entry.zProd() * toPosition, mother.T() + entry.tProd() * toTime);
    bool isFinal = entry.isFinal();
    o2::mcutils::MCGenHelper::encodeParticleStatusAndTracking(product, isFinal ? 1 : 2, std::abs(entry.status()), isFinal);
    product.SetWeight(mother.GetWeight());
    product.SetUniqueID(kPPrimary);
    particles.push_back(product);
  }

  auto& decayed = particles[index];
  // the mother keeps its generator status code, only its HepMC status becomes decayed
  o2::mcutils::MCGenHelper::encodeParticleStatusAndTracking(decayed, 2, o2::mcgenstatus::getGenStatusCode(decayed.GetStatusCode()), false);
  decayed.SetFirstDaughter(toIndex(event[0].daughter1()));
  decayed.SetLastDaughter(toIndex(event[0].daughter2() > 0 ? event[0].daughter2() : event[0].daughter1()));
  return true;
}

/*****************************************************************/

void EventDecayer::checkParticles(std::vector<TParticle> const& particles, EventDecayerReport& report) const
{
  auto pdgDB = o2::O2DatabasePDG::Instance();
  for (auto const& particle : particles) {
    if (!particle.TestBit(ParticleStatus::kToBeDone)) {
      continue;
    }
    auto pdg = particle.GetPdgCode();
    if (!pdgDB->GetParticle(pdg)) {
      report.unknownInO2[pdg]++;
    }
    // ions (codes above 10^9) are created by Geant4 on demand, so they are not in the table
    if (mHasG4Table && std::abs(pdg) < 1000000000 && mG4Particles.find(pdg) == mG4Particles.end()) {
      report.unknownInG4[pdg]++;
    }
  }
}

void EventDecayer::checkLifetimes(EventDecayerReport& report) const
{
  if (!mHasG4Table) {
    return;
  }
  for (auto const& [pdg, count] : report.decayed) {
    auto g4 = mG4Particles.find(pdg);
    if (g4 == mG4Particles.end() || g4->second.ctauMM == 0.) {
      continue;
    }
    double pythiaCtau = mPythia.particleData.tau0(pdg);
    double g4Ctau = g4->second.ctauMM;
    if (std::abs(pythiaCtau - g4Ctau) / g4Ctau > 0.01) {
      report.lifetimeMismatch[pdg] = {pythiaCtau, g4Ctau};
    }
  }
}

} // namespace eventgen
} // namespace o2
