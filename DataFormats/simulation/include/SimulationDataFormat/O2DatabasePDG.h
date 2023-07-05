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

//
// Created by Sandro Wenzel on 11.08.22.
//

#ifndef O2_O2DATABASEPDG_H
#define O2_O2DATABASEPDG_H
#include <string>
#include "TDatabasePDG.h"
#include "TParticlePDG.h"

namespace o2
{

// An ALICE specific extension of ROOT's TDatabasePDG
//
// By using O2DatabasePDG::Instance() in our code instead of TDatabasePDG::Instance(), correct initialization
// is guaranteed. Alternatively, a static function is exposed with which particles can be added to TDatabasePDG objects
// directly.
class O2DatabasePDG
{
  //
 public:
  static TDatabasePDG* Instance()
  {
    static bool initialized = false; // initialize this --> adds particles to TDatabasePDG;
    auto db = TDatabasePDG::Instance();
    if (!initialized) {
      addALICEParticles(db);
      if (const char* o2Root = std::getenv("O2_ROOT")) {
        auto inputExtraPDGs = std::string(o2Root) + "/share/Detectors/gconfig/data/extra_ions_pdg_table.dat";
        db->ReadPDGTable(inputExtraPDGs.c_str());
      }
      initialized = true;
    }
    return db;
  }

  // adds ALICE particles to a given TDatabasePDG instance
  static void addALICEParticles(TDatabasePDG* db = TDatabasePDG::Instance());

  // get particle's (if any) mass
  static Double_t MassImpl(TParticlePDG* particle, bool& success)
  {
    success = false;
    if (!particle) {
      return -1.;
    }
    success = true;
    return particle->Mass();
  }

  // determine particle to get mass for based on PDG
  static Double_t Mass(int pdg, bool& success, TDatabasePDG* db = O2DatabasePDG::Instance())
  {
    if (pdg < IONBASELOW || pdg > IONBASEHIGH) {
      // not an ion, return immediately
      return MassImpl(db->GetParticle(pdg), success);
    }
    if (auto particle = db->GetParticle(pdg)) {
      // see if this ion can be found
      return MassImpl(particle, success);
    }
    // if we are here, try one last time to look for ground state of potential isomere state
    pdg = pdg / 10 * 10;
    return MassImpl(db->GetParticle(pdg), success);
  }

  // remove default constructor
  O2DatabasePDG() = delete;

 private:
  static constexpr int IONBASELOW{1000000000};
  static constexpr int IONBASEHIGH{1099999999};
};

// by keeping this inline, we can use it in other parts of the code, for instance Framework or Analysis,
// without needing to link against this library
inline void O2DatabasePDG::addALICEParticles(TDatabasePDG* db)
{
  //
  // Add ALICE particles to the ROOT PDG data base
  // Code has been taken from AliRoot
  const Int_t kspe = 50000000;

  // PDG nuclear states are 10-digit numbers
  // 10LZZZAAAI e.g. deuteron is
  // 1000010020
  const Int_t kion = 1000000000;

  /*
  const Double_t kAu2Gev=0.9314943228;
*/

  const Double_t khSlash = 1.0545726663e-27;
  const Double_t kErg2Gev = 1 / 1.6021773349e-3;
  const Double_t khShGev = khSlash * kErg2Gev;
  const Double_t kYear2Sec = 3600 * 24 * 365.25;

  // Heavy-flavour particles

  // χc1(3872) aka X(3872), taken from PDG 2022 (https://pdg.lbl.gov/2022/listings/rpp2022-list-chi-c1-3872.pdf)
  db->AddParticle("Chi_c1(3872)", "Chi_c1(3872)", 3.87165, kFALSE, 0, 0, "CCBarMeson", 9920443);

  //
  // Bottom mesons
  // mass and life-time from PDG
  //
  db->AddParticle("Upsilon(3S)", "Upsilon(3S)", 10.3552, kTRUE,
                  0, 0, "Bottonium", 200553);

  // QCD diffractive states
  db->AddParticle("rho_diff0", "rho_diff0", 0, kTRUE,
                  0, 0, "QCD diffr. state", 9900110);
  db->AddParticle("pi_diffr+", "pi_diffr+", 0, kTRUE,
                  0, 3, "QCD diffr. state", 9900210);
  db->AddParticle("omega_di", "omega_di", 0, kTRUE,
                  0, 0, "QCD diffr. state", 9900220);
  db->AddParticle("phi_diff", "phi_diff", 0, kTRUE,
                  0, 0, "QCD diffr. state", 9900330);
  db->AddParticle("J/psi_di", "J/psi_di", 0, kTRUE,
                  0, 0, "QCD diffr. state", 9900440);
  db->AddParticle("n_diffr0", "n_diffr0", 0, kTRUE,
                  0, 0, "QCD diffr. state", 9902110);
  db->AddParticle("p_diffr+", "p_diffr+", 0, kTRUE,
                  0, 3, "QCD diffr. state", 9902210);

  // From Herwig
  db->AddParticle("PSID    ", " ", 3.7699, kFALSE, 0.0, 0, "meson", 30443);

  db->AddParticle("A_00    ", " ", 0.9960, kFALSE, 0.0, 0, "meson", 9000111);
  db->AddParticle("A_0+    ", " ", 0.9960, kFALSE, 0.0, +3, "meson", 9000211);
  db->AddParticle("A_0-    ", " ", 0.9960, kFALSE, 0.0, -3, "meson", -9000211);

  //db->AddParticle("F0P0    ", " ", 0.9960, kFALSE, 0.0, 0, "meson",  9010221);

  db->AddParticle("KDL_2+  ", " ", 1.773, kFALSE, 0.0, +3, "meson", 10325);
  db->AddParticle("KDL_2-  ", " ", 1.773, kFALSE, 0.0, -3, "meson", -10325);

  db->AddParticle("KDL_20  ", " ", 1.773, kFALSE, 0.0, 0, "meson", 10315);
  db->AddParticle("KDL_2BR0", " ", 1.773, kFALSE, 0.0, 0, "meson", -10315);

  db->AddParticle("PI_2+   ", " ", 1.670, kFALSE, 0.0, +3, "meson", 10215);
  db->AddParticle("PI_2-   ", " ", 1.670, kFALSE, 0.0, -3, "meson", -10215);
  db->AddParticle("PI_20   ", " ", 1.670, kFALSE, 0.0, 0, "meson", 10115);

  db->AddParticle("KD*+    ", " ", 1.717, kFALSE, 0.0, +3, "meson", 30323);
  db->AddParticle("KD*-    ", " ", 1.717, kFALSE, 0.0, -3, "meson", -30323);

  db->AddParticle("KD*0    ", " ", 1.717, kFALSE, 0.0, 0, "meson", 30313);
  db->AddParticle("KDBR*0  ", " ", 1.717, kFALSE, 0.0, 0, "meson", -30313);

  db->AddParticle("RHOD+   ", " ", 1.700, kFALSE, 0.0, +3, "meson", 30213);
  db->AddParticle("RHOD-   ", " ", 1.700, kFALSE, 0.0, -3, "meson", -30213);
  db->AddParticle("RHOD0   ", " ", 1.700, kFALSE, 0.0, 0, "meson", 30113);

  db->AddParticle("ETA_2(L)", " ", 1.632, kFALSE, 0.0, 0, "meson", 10225);
  db->AddParticle("ETA_2(H)", " ", 1.854, kFALSE, 0.0, 0, "meson", 10335);
  db->AddParticle("OMEGA(H)", " ", 1.649, kFALSE, 0.0, 0, "meson", 30223);

  db->AddParticle("KDH_2+  ", " ", 1.816, kFALSE, 0.0, +3, "meson", 20325);
  db->AddParticle("KDH_2-  ", " ", 1.816, kFALSE, 0.0, -3, "meson", -20325);

  db->AddParticle("KDH_20  ", " ", 1.816, kFALSE, 0.0, 0, "meson", 20315);
  db->AddParticle("KDH_2BR0", " ", 1.816, kFALSE, 0.0, 0, "meson", -20315);

  db->AddParticle("KD_3+   ", " ", 1.773, kFALSE, 0.0, +3, "meson", 327);
  db->AddParticle("KD_3-   ", " ", 1.773, kFALSE, 0.0, -3, "meson", -327);

  db->AddParticle("KD_30   ", " ", 1.773, kFALSE, 0.0, 0, "meson", 317);
  db->AddParticle("KD_3BR0 ", " ", 1.773, kFALSE, 0.0, 0, "meson", -317);

  db->AddParticle("RHO_3+  ", " ", 1.691, kFALSE, 0.0, +3, "meson", 217);
  db->AddParticle("RHO_3-  ", " ", 1.691, kFALSE, 0.0, -3, "meson", -217);
  db->AddParticle("RHO_30  ", " ", 1.691, kFALSE, 0.0, 0, "meson", 117);
  db->AddParticle("OMEGA_3 ", " ", 1.667, kFALSE, 0.0, 0, "meson", 227);
  db->AddParticle("PHI_3   ", " ", 1.854, kFALSE, 0.0, 0, "meson", 337);

  db->AddParticle("CHI2P_B0", " ", 10.232, kFALSE, 0.0, 0, "meson", 110551);
  db->AddParticle("CHI2P_B1", " ", 10.255, kFALSE, 0.0, 0, "meson", 120553);
  db->AddParticle("CHI2P_B2", " ", 10.269, kFALSE, 0.0, 0, "meson", 100555);
  db->AddParticle("UPSLON4S", " ", 10.580, kFALSE, 0.0, 0, "meson", 300553);

  // IONS
  //
  // Done by default now from Pythia6 table
  // Needed for other generators
  // So check if already defined

  /// UPDATED VALUES FROM CODATA 2018
  Int_t ionCode = kion + 10020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Deuteron", "Deuteron", 1.87561294257, kTRUE,
                    0, 3, "Ion", ionCode);
  }
  db->AddAntiParticle("AntiDeuteron", -ionCode);

  ionCode = kion + 10030;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Triton", "Triton", 2.80892113298, kFALSE,
                    khShGev / (12.33 * kYear2Sec), 3, "Ion", ionCode);
  }
  db->AddAntiParticle("AntiTriton", -ionCode);

  ionCode = kion + 20030;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("HE3", "HE3", 2.80839160743, kFALSE,
                    0, 6, "Ion", ionCode);
  }
  db->AddAntiParticle("AntiHE3", -ionCode);

  ionCode = kion + 20040;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Alpha", "Alpha", 3.7273794066, kTRUE,
                    khShGev / (12.33 * kYear2Sec), 6, "Ion", ionCode);
  }
  db->AddAntiParticle("AntiAlpha", -ionCode);
  /// PLEASE UPDATE REGULARLY

  // Special particles
  //
  db->AddParticle("Cherenkov", "Cherenkov", 0, kFALSE,
                  0, 0, "Special", kspe + 50);
  db->AddParticle("FeedbackPhoton", "FeedbackPhoton", 0, kFALSE,
                  0, 0, "Special", kspe + 51);

  //Hyper nuclei and exotica
  ionCode = 1010010030;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("HyperTriton", "HyperTriton", 2.99131, kFALSE,
                    2.5e-15, 3, "Ion", ionCode);
  }

  ionCode = -1010010030;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiHyperTriton", "AntiHyperTriton", 2.99131, kFALSE,
                    2.5e-15, 3, "Ion", ionCode);
  }

  //hyper hydrogen 4 ground state
  ionCode = 1010010040;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Hyperhydrog4", "Hyperhydrog4", 3.9226, kFALSE,
                    2.5e-15, 3, "Ion", ionCode);
  }
  //anti hyper hydrogen 4 ground state
  ionCode = -1010010040;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiHyperhydrog4", "AntiHyperhydrog4", 3.9226, kFALSE,
                    2.5e-15, 3, "Ion", ionCode);
  }
  //hyper hydrogen 4 excited state
  ionCode = 1010010041;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Hyperhydrog4*", "Hyperhydrog4*", 3.9237, kFALSE,
                    2.5e-15, 3, "Ion", ionCode);
  }
  //anti hyper hydrogen 4 excited state
  ionCode = -1010010041;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiHyperhydrog4*", "AntiHyperhydrog4*", 3.9237, kFALSE,
                    2.5e-15, 3, "Ion", ionCode);
  }
  //hyper helium 4 ground state
  ionCode = 1010020040;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Hyperhelium4", "Hyperhelium4", 3.9217, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }
  //anti hyper helium 4 ground state
  ionCode = -1010020040;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiHyperhelium4", "AntiHyperhelium4", 3.9217, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }
  //hyper helium 4 excited state
  ionCode = 1010020041;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Hyperhelium4*", "Hyperhelium4*", 3.9231, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }
  //anti hyper helium 4 excited state
  ionCode = -1010020041;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiHyperhelium4*", "AntiHyperhelium4*", 3.9231, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }

  ionCode = 1010020050;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Hyperhelium5", "Hyperhelium5", 4.841, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }

  ionCode = -1010020050;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiHyperhelium5", "AntiHyperhelium5", 4.841, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }

  ionCode = 1020010040;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("DoubleHyperhydrogen4", "DoubleHyperhydrogen4", 4.106, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }

  ionCode = -1020010040;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("DoubleAntiHyperhydrogen4", "DoubleAntiHyperhydrogen4", 4.106, kFALSE,
                    2.5e-15, 6, "Ion", ionCode);
  }

  ionCode = 1010000020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("LambdaNeutron", "LambdaNeutron", 2.054, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1010000020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiLambdaNeutron", "AntiLambdaNeutron", 2.054, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1020000020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Hdibaryon", "Hdibaryon", 2.23, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1020000020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiHdibaryon", "AntiHdibaryon", 2.23, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1010000030;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("LambdaNeutronNeutron", "LambdaNeutronNeutron", 2.99, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1010000030;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiLambdaNeutronNeutron", "AntiLambdaNeutronNeutron", 2.99, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1020010020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Xi0Proton", "Xi0Proton", 2.248, kFALSE,
                    5e-15, 3, "Ion", ionCode);
  }

  ionCode = -1020010020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiXi0Proton", "AntiXi0Proton", 2.248, kFALSE,
                    5e-15, 3, "Ion", ionCode);
  }

  ionCode = 1030000020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("OmegaProton", "OmegaProton", 2.592, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = -1030000020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiOmegaProton", "AntiOmegaProton", 2.592, kFALSE,
                    2.5e-15, 0, "Special", ionCode);
  }

  ionCode = 1030010020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("OmegaNeutron", "OmegaNeutron", 2.472, kFALSE,
                    0.003, 3, "Special", ionCode);
  }

  ionCode = -1030010020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiOmegaNeutron", "AntiOmegaNeutron", 2.472, kFALSE,
                    0.003, 3, "Special", ionCode);
  }

  ionCode = 1060020020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("OmegaOmega", "OmegaOmega", 3.229, kFALSE,
                    2.5e-15, 6, "Special", ionCode);
  }

  ionCode = -1060020020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiOmegaOmega", "AntiOmegaOmega", 3.229, kFALSE,
                    2.5e-15, 6, "Special", ionCode);
  }

  ionCode = 1010010021;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Lambda1405Proton", "Lambda1405Proton", 2.295, kFALSE,
                    0.05, 3, "Special", ionCode);
  }

  ionCode = -1010010021;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiLambda1405Proton", "AntiLambda1405Proton", 2.295, kFALSE,
                    0.05, 3, "Special", ionCode);
  }

  ionCode = 1020000021;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Lambda1405Lambda1405", "Lambda1405Lambda1405", 2.693, kFALSE,
                    0.05, 0, "Special", ionCode);
  }

  ionCode = -1020000021;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("AntiLambda1405Lambda1405", "AntiLambda1405Lambda1405", 2.693, kFALSE,
                    0.05, 0, "Special", ionCode);
  }

  ionCode = 2010010030;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("CTriton", "CTriton", 4.162, kFALSE,
                    3.3e-12, 3, "Ion", ionCode);
    db->AddAntiParticle("AntiCTriton", -ionCode);
  }

  ionCode = 2010010020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("CDeuteron", "CDeuteron", 3.226, kFALSE,
                    3.3e-12, 3, "Ion", ionCode);
    db->AddAntiParticle("AntiCDeuteron", -ionCode);
  }

  // Special resonances

  ionCode = 9010221;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("f0_980", "f0_980", 0.980, kFALSE,
                    0.07, 0, "Resonance", ionCode);
  }

  ionCode = 225;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("f2_1270", "f2_1270", 1.275, kFALSE,
                    0.185, 0, "Resonance", ionCode);
  }

  // Xi-/+ (1820)
  ionCode = 123314;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Xi_Minus_1820", "Xi_Minus_1820", 1.8234, kFALSE, 0.024, -3, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Xi_Plus_1820", "Xi_Plus_1820", 1.8234, kFALSE, 0.024, 3, "Resonance", -ionCode);
  }

  // Xi0 (1820)
  ionCode = 123324;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Xi_0_1820", "Xi_0_1820", 1.8234, kFALSE, 0.024, 0, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Xi_0_Bar_1820", "Xi_0_Bar_1820", 1.8234, kFALSE, 0.024, 0, "Resonance", -ionCode);
  }

  // Ps - hidden strange (s-sbar) pentaquarks

  ionCode = 9322134;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("ps_2100", "ps_2100", 2.100, kFALSE,
                    0.040, 3, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("anti-ps_2100", "anti-ps_2100", 2.100, kFALSE,
                    0.040, -3, "Resonance", -ionCode);
  }

  ionCode = 9322136;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("ps_2500", "ps_2500", 2.500, kFALSE,
                    0.040, 3, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("anti-ps_2500", "anti-ps_2500", 2.500, kFALSE,
                    0.040, -3, "Resonance", -ionCode);
  }

  //Additional Hidden Strangeness Pentaquarks

  //Ps +/-
  ionCode = 9322132;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_Plus_1870", "Ps_Plus_1870", 1.870, kFALSE,
                    0.10, 3, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_Minus_1870", "Anti-Ps_Minus_1870", 1.870, kFALSE,
                    0.10, -3, "Resonance", -ionCode);
  }
  ionCode = 9322312;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_Plus_2065", "Ps_Plus_2065", 2.065, kFALSE,
                    0.10, 3, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_Minus_2065", "Anti-Ps_Minus_2065", 2.065, kFALSE,
                    0.10, -3, "Resonance", -ionCode);
  }
  ionCode = 9323212;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_Plus_2255", "Ps_Plus_2255", 2.255, kFALSE,
                    0.10, 3, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_Minus_2255", "Anti-Ps_Minus_2255", 2.255, kFALSE,
                    0.10, -3, "Resonance", -ionCode);
  }
  ionCode = 9332212;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_Plus_2455", "Ps_Plus_2455", 2.455, kFALSE,
                    0.10, 3, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_Minus_2455", "Anti-Ps_Minus_2455", 2.455, kFALSE,
                    0.10, -3, "Resonance", -ionCode);
  }

  //Ps0
  ionCode = 9322131;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_0_1870", "Ps_0_1870", 1.870, kFALSE,
                    0.10, 0, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_0_1870", "Anti-Ps_0_1870", 1.870, kFALSE,
                    0.10, 0, "Resonance", -ionCode);
  }
  ionCode = 9322311;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_0_2065", "Ps_0_2065", 2.065, kFALSE,
                    0.10, 0, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_0_2065", "Anti-Ps_0_2065", 2.065, kFALSE,
                    0.10, 0, "Resonance", -ionCode);
  }
  ionCode = 9323211;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_0_2255", "Ps_0_2255", 2.255, kFALSE,
                    0.10, 0, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_0_2255", "Anti-Ps_0_2255", 2.255, kFALSE,
                    0.10, 0, "Resonance", -ionCode);
  }
  ionCode = 9332211;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Ps_0_2455", "Ps_0_2455", 2.455, kFALSE,
                    0.10, 0, "Resonance", ionCode);
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Anti-Ps_0_2455", "Anti-Ps_0_2455", 2.455, kFALSE,
                    0.10, 0, "Resonance", -ionCode);
  }

  // Charm pentaquarks
  // Theta_c: isospin singlet with J=1/2+ (see https://arxiv.org/abs/hep-ph/0409121)
  ionCode = 9422111;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Anti-Theta_c_3100", "Anti-Theta_c_3100", 3.099, kFALSE,
                    83.e-6, 0, "Resonance", ionCode); // same width as D*+ (83 keV)
  }
  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("Theta_c_3100", "Theta_c_3100", 3.099, kFALSE,
                    83.e-6, 0, "Resonance", -ionCode); // same width as D*+ (83 keV)
  }

  // d*(2380) - dibaryon resonance

  ionCode = 900010020; //Arbitrary choice - as deuteron but with leading 9 instead of 10
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("d*_2380", "d*_2380", 2.38, kFALSE,
                    0.070, 3, "Resonance", ionCode);
  }
  db->AddAntiParticle("d*_2380_bar", -ionCode);

  //Sexaquark (uuddss): compact, neutral and stable hypothetical bound state (arxiv.org/abs/1708.08951)
  ionCode = 900000020;
  if (!db->GetParticle(ionCode)) {
    db->AddParticle("Sexaquark", "Sexaquark", 2.0, kTRUE, 0.0, 0, "Special", ionCode);
  }

  if (!db->GetParticle(-ionCode)) {
    db->AddParticle("AntiSexaquark", "AntiSexaquark", 2.0, kTRUE, 0.0, 0, "Special", -ionCode);
  }
}

} // namespace o2

#endif //O2_O2DATABASEPDG_H
