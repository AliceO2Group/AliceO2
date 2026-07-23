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

/// \file GeneratorKrDecay.cxx
/// \brief Generator for 83mKr decays, for TPC gain-map calibration simulation
/// \author Ankur Yadav <ankur.yadav@cern.ch>

#include "TPCSimulation/GeneratorKrDecay.h"
#include "Framework/Logger.h"
#include "TDatabasePDG.h"
#include "TParticle.h"
#include "TParticlePDG.h"
#include "TRandom.h"
#include "TMath.h"
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace o2
{
namespace tpc
{

// ── 83mKr decay physics ──────────────────────────────────────────────────
//
// Energies and ICC values are read at runtime from $G4LEVELGAMMADATA/z36.a83
// (set automatically by Geant4 in any O2 alienv session). The hardcoded
// fallback values below are taken from PhotonEvaporation5.7/z36.a83 and are
// used only if the file cannot be opened or parsed.
//
// Atomic constants (NIST) — stable across G4 data releases, always hardcoded:
//   Kr K-binding  = 14.3256 keV
//   Kr L1-binding =  1.9210 keV
//   Kr Kα X-ray   = 12.6000 keV
//   Kr K-shell fluorescence yield ω_K = 0.652  (Bambynek et al.)
//
// From the G4 file we read for two levels:
//   Level 2 (41.5569 keV, T1): E_gamma, ICC_total, K_shell_fraction
//   Level 1 ( 9.4053 keV, T2): E_gamma, ICC_total
// Everything else is derived from these five numbers + the atomic constants.
// ─────────────────────────────────────────────────────────────────────────

// Parse $G4LEVELGAMMADATA/z36.a83.
// Returns true and fills five values (energies in keV) on success.
static bool parseG4PhotonEvap(const char* path,
                              double& E_T1,     // T1 gamma energy [keV]
                              double& ICC_T1,   // T1 ICC_total
                              double& Kfrac_T1, // T1 K-shell fraction of ICC
                              double& E_T2,     // T2 gamma energy [keV]
                              double& ICC_T2)   // T2 ICC_total
{
  std::ifstream f(path);
  if (!f.is_open()) {
    return false;
  }

  bool gotT1 = false, gotT2 = false;
  bool wantT1 = false, wantT2 = false;
  std::string line;

  while (std::getline(f, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream ss(line);
    int idx;
    std::string tok;
    double eLevel;

    // Header line: "   N  -  E_level  halflife  ..."
    if ((ss >> idx >> tok >> eLevel) && tok == "-") {
      wantT1 = (idx == 2); // 41.5569 keV metastable state -> T1 transition
      wantT2 = (idx == 1); //  9.4053 keV metastable state -> T2 transition
      continue;
    }

    if (!wantT1 && !wantT2) {
      continue;
    }

    // Transition line: "  daughter  E_gamma  intensity  multipolarity  delta  ICC_total  K_frac  ..."
    ss.clear();
    ss.str(line);
    int daughter, multi;
    double Eg, inten, delta, icc, kfrac;
    if (!(ss >> daughter >> Eg >> inten >> multi >> delta >> icc >> kfrac)) {
      continue;
    }

    if (wantT1) {
      E_T1 = Eg;
      ICC_T1 = icc;
      Kfrac_T1 = kfrac;
      gotT1 = true;
      wantT1 = false;
    }
    if (wantT2) {
      E_T2 = Eg;
      ICC_T2 = icc;
      gotT2 = true;
      wantT2 = false;
    }

    if (gotT1 && gotT2) {
      break;
    }
  }

  if (!gotT1 || !gotT2) {
    return false;
  }

  // Sanity check — values far outside these ranges indicate a corrupt or wrong file
  if (E_T1 < 25. || E_T1 > 40.) {
    return false;
  }
  if (E_T2 < 5. || E_T2 > 15.) {
    return false;
  }
  if (ICC_T1 < 100.) {
    return false;
  }
  if (ICC_T2 < 5.) {
    return false;
  }
  if (Kfrac_T1 < 0.1 || Kfrac_T1 > 0.5) {
    return false;
  }

  return true;
}

KrDecayTable::KrDecayTable()
{
  // ── Atomic constants (NIST, keV, converted to GeV for ROOT) ──────────
  static constexpr double kKbind = 14.3256e-6;  // Kr K-shell binding
  static constexpr double kL1bind = 1.9210e-6;  // Kr L1-shell binding
  static constexpr double kKalpha = 12.6000e-6; // Kr Kα X-ray
  static constexpr double kKfluY = 0.652;       // Kr K-shell fluorescence yield

  // ── Fallback values from PhotonEvaporation5.7/z36.a83 ────────────────
  double E_T1 = 32.1516e-6; // [GeV] T1 gamma energy
  double ICC_T1 = 2035.0;
  double Kfrac_T1 = 0.248; // fraction of ICC_T1 going through K-shell
  double E_T2 = 9.4053e-6; // [GeV] T2 gamma energy
  double ICC_T2 = 17.09;

  // ── Try to load from installed G4 data (keV in file → convert to GeV) ─
  const char* g4dir = std::getenv("G4LEVELGAMMADATA");
  if (g4dir) {
    std::string path = std::string(g4dir) + "/z36.a83";
    double fE1, fICC1, fKf1, fE2, fICC2;
    if (parseG4PhotonEvap(path.c_str(), fE1, fICC1, fKf1, fE2, fICC2)) {
      E_T1 = fE1 * 1e-6;
      ICC_T1 = fICC1;
      Kfrac_T1 = fKf1;
      E_T2 = fE2 * 1e-6;
      ICC_T2 = fICC2;
      LOG(info) << "[KrDecayTable] Loaded from " << path << " -- "
                << "T1: E=" << fE1 << " keV ICC=" << ICC_T1 << " K_frac=" << Kfrac_T1 << ", "
                << "T2: E=" << fE2 << " keV ICC=" << ICC_T2;
    } else {
      LOG(warning) << "[KrDecayTable] Could not parse " << path << " -- using hardcoded fallback values";
    }
  } else {
    LOG(warning) << "[KrDecayTable] G4LEVELGAMMADATA not set -- using hardcoded fallback values";
  }

  // ── Derived probabilities ─────────────────────────────────────────────
  const double P_T1_g = 1.0 / (1.0 + ICC_T1);                  // T1 gamma
  const double P_T1_K_IC = Kfrac_T1 * ICC_T1 / (1.0 + ICC_T1); // T1 K-shell IC
  const double P_T1_out = ICC_T1 / (1.0 + ICC_T1) - P_T1_K_IC; // T1 outer-shell IC
  const double P_T2_g = 1.0 / (1.0 + ICC_T2);                  // T2 gamma
  const double P_T2_IC = ICC_T2 / (1.0 + ICC_T2);              // T2 IC

  const double P_T1_Kf = P_T1_K_IC * kKfluY;         // T1 K-IC → K-fluorescence
  const double P_T1_Ka = P_T1_K_IC * (1.0 - kKfluY); // T1 K-IC → K-Auger

  // ── Particle kinetic energies ─────────────────────────────────────────
  const double E_L_CE_T1 = E_T1 - kL1bind;     // T1 L1-shell CE
  const double E_K_CE = E_T1 - kKbind;         // T1 K-shell CE
  const double E_KLL = kKbind - 2.0 * kL1bind; // KLL Auger
  const double E_res_aug = kKbind - E_KLL;     // residual Auger (K-Auger path)
  const double E_Laug_Kf = kKbind - kKalpha;   // L-Auger after Kα emission
  const double E_L_CE_T2 = E_T2 - kL1bind;     // T2 L1-shell CE

  // ── Eight channels (T1-mode × T2-mode) ───────────────────────────────
  int i = 0;

  // Ch 0: T1 outer-IC + T2 IC → 41.6 keV local
  channels[i] = {P_T1_out * P_T2_IC, 4, {{11, E_L_CE_T1}, {11, kL1bind}, {11, E_L_CE_T2}, {11, kL1bind}}};
  i++;

  // Ch 1: T1 outer-IC + T2 γ → 32.2 keV local + γ(T2) separate
  channels[i] = {P_T1_out * P_T2_g, 3, {{11, E_L_CE_T1}, {11, kL1bind}, {22, E_T2}}};
  i++;

  // Ch 2: T1 K-IC + K-Auger + T2 IC → 41.6 keV local
  channels[i] = {P_T1_Ka * P_T2_IC, 5, {{11, E_K_CE}, {11, E_KLL}, {11, E_res_aug}, {11, E_L_CE_T2}, {11, kL1bind}}};
  i++;

  // Ch 3: T1 K-IC + K-fluor + T2 IC → 29.1 keV local + Kα separate
  channels[i] = {P_T1_Kf * P_T2_IC, 5, {{11, E_K_CE}, {11, E_Laug_Kf}, {22, kKalpha}, {11, E_L_CE_T2}, {11, kL1bind}}};
  i++;

  // Ch 4: T1 K-IC + K-fluor + T2 γ → 19.6 keV local + Kα + γ(T2) separate
  channels[i] = {P_T1_Kf * P_T2_g, 4, {{11, E_K_CE}, {11, E_Laug_Kf}, {22, kKalpha}, {22, E_T2}}};
  i++;

  // Ch 5: T1 K-IC + K-Auger + T2 γ → 32.2 keV local + γ(T2) separate
  channels[i] = {P_T1_Ka * P_T2_g, 4, {{11, E_K_CE}, {11, E_KLL}, {11, E_res_aug}, {22, E_T2}}};
  i++;

  // Ch 6: T1 γ + T2 IC → 9.4 keV local + γ(T1) separate
  channels[i] = {P_T1_g * P_T2_IC, 3, {{22, E_T1}, {11, E_L_CE_T2}, {11, kL1bind}}};
  i++;

  // Ch 7: T1 γ + T2 γ → both photons escape
  channels[i] = {P_T1_g * P_T2_g, 2, {{22, E_T1}, {22, E_T2}}};
  i++;

  double sum = 0.;
  for (int j = 0; j < kNChannels; j++) {
    sum += channels[j].fraction;
  }
  double cum = 0.;
  for (int j = 0; j < kNChannels; j++) {
    cum += channels[j].fraction / sum;
    cumulative[j] = cum;
  }
}

const KrDecayTable::Channel& KrDecayTable::sample() const
{
  double r = gRandom->Uniform();
  for (int i = 0; i < kNChannels; i++) {
    if (r <= cumulative[i]) {
      return channels[i];
    }
  }
  return channels[kNChannels - 1];
}

} // namespace tpc
} // namespace o2

// ── GeneratorKrDecay ─────────────────────────────────────────────────────

namespace o2
{
namespace eventgen
{

GeneratorKrDecay::GeneratorKrDecay() : Generator("KrDecay", "83mKr TPC calibration source")
{
}

GeneratorKrDecay::~GeneratorKrDecay() = default;

Bool_t GeneratorKrDecay::Init()
{
  if (const char* env = std::getenv("KR_N_PER_EVENT")) {
    int n = std::atoi(env);
    if (n > 0) {
      mNPerEvent = n;
    }
  }
  LOG(info) << "[GeneratorKrDecay] Init: rInner=" << kRInner << " rOuter=" << kROuter
            << " halfZ=" << kHalfZ << " nPerEvent=" << mNPerEvent;

  mTable = std::make_unique<o2::tpc::KrDecayTable>();
  setPositionUnit(1.0); // coords in cm
  return Generator::Init();
}

Bool_t GeneratorKrDecay::generateEvent()
{
  mVertices.clear();
  // 1 cm safety margin inside field cage boundaries — avoids placing
  // electrons exactly on sector boundaries which can cause hit coordinate
  // transformation crashes in the merger when ROOT fills the TTree.
  const double rInner = kRInner + 1.0;
  const double rOuter = kROuter - 1.0;
  const double halfZ = kHalfZ - 1.0;
  const double r2Min = rInner * rInner;
  const double r2Max = rOuter * rOuter;
  for (int i = 0; i < mNPerEvent; ++i) {
    double r = std::sqrt(gRandom->Uniform(r2Min, r2Max));
    double phi = gRandom->Uniform(0., TMath::TwoPi());
    double z = gRandom->Uniform(-halfZ, halfZ);
    mVertices.push_back({{r * std::cos(phi), r * std::sin(phi), z}});
  }
  return kTRUE;
}

Bool_t GeneratorKrDecay::importParticles()
{
  mParticles.clear();
  // Reserve before any push_back to prevent std::vector reallocation.
  // TParticle inherits from TObject (ROOT memory pool) and is not safe
  // to move-construct via std::vector reallocation on macOS arm64 —
  // ROOT's TStorage bookkeeping gets corrupted, causing malloc failures
  // ~50-100 events later. Reserving eliminates all reallocations.
  mParticles.reserve(mNPerEvent * o2::tpc::KrDecayTable::kNChannels);

  const int status = o2::tpc::krO2EncodedStatus(1, 0);

  for (size_t iv = 0; iv < mVertices.size(); ++iv) {
    double vx = mVertices[iv][0];
    double vy = mVertices[iv][1];
    double vz = mVertices[iv][2];

    const o2::tpc::KrDecayTable::Channel& ch = mTable->sample();
    for (int ip = 0; ip < ch.nProducts; ++ip) {
      int pdg = ch.products[ip].pdg;
      double eKin = ch.products[ip].eKin;
      if (eKin < 0.1e-6) {
        continue;
      }

      double mass = (pdg == 11) ? 0.000511 : 0.0;
      double E = eKin + mass;
      double pmag = std::sqrt(std::max(0., E * E - mass * mass));
      double cosT = gRandom->Uniform(-1., 1.);
      double sinT = std::sqrt(1. - cosT * cosT);
      double phi = gRandom->Uniform(0., TMath::TwoPi());

      TParticle part(pdg, status, -1, -1, -1, -1,
                     pmag * sinT * std::cos(phi),
                     pmag * sinT * std::sin(phi),
                     pmag * cosT,
                     E, vx, vy, vz, 0.);
      mParticles.push_back(part);
      // kToBeDone=BIT(16), kPrimary=BIT(17)
      // Must be set AFTER push_back — copy constructor resets fBits
      mParticles.back().SetBit(BIT(16));
      mParticles.back().SetBit(BIT(17));
    }
  }
  return kTRUE;
}

} // namespace eventgen
} // namespace o2
