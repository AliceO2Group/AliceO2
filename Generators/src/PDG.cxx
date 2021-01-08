// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Class to encapsulate the ALICE updates to TDatabasePDG.h
// Can be used by TGeant3 and TGeant4
// It contains also the constants for the PDG particle IDs.
// Should evolve towards dynamical loading from external data base.
// Comments to: andreas.morsch@cern.ch

#include "Generators/PDG.h"
#include <TDatabasePDG.h>
#include <FairLogger.h>

void o2::PDG::addParticle(const char* name, const char* title, double mass, bool stable, double width, double charge,
                          const char* particleClass, int pdgCode, int verbose)
{
  TDatabasePDG* pdgDB = TDatabasePDG::Instance();
  if (verbose > 1) {
    LOG(INFO) << "Adding particle " << name << " with pdg code " << pdgCode;
  }
  TParticlePDG* part = pdgDB->GetParticle(pdgCode);
  if (part) {
    if (verbose) {
      LOG(WARNING) << "PDG code " << pdgCode << " is already defined (" << part->GetName() << ")";
    }
    return;
  }
  pdgDB->AddParticle(name, title, mass, stable, width, charge, particleClass, pdgCode);
}

void o2::PDG::addAntiParticle(const char* name, int pdgCode, int verbose)
{
  TDatabasePDG* pdgDB = TDatabasePDG::Instance();
  if (verbose > 1) {
    LOG(INFO) << "Adding anti-particle " << name << " with pdg code " << pdgCode;
  }
  TParticlePDG* part = pdgDB->GetParticle(pdgCode);
  if (part) {
    if (verbose) {
      LOG(WARNING) << "PDG code " << pdgCode << " is already defined (" << part->GetName() << ")";
    }
    return;
  }
  pdgDB->AddAntiParticle(name, pdgCode);
}

void o2::PDG::addParticlesToPdgDataBase(int verbose)
{

  //
  // Add particles to the PDG data base
  //

  static bool bAdded = false;
  // Check if already called
  if (bAdded) {
    return;
  }
  bAdded = true;

  const int kspe = 50000000;

  // PDG nuclear states are 10-digit numbers
  // 10LZZZAAAI e.g. deuteron is
  // 1000010020
  const int kion = 1000000000;

  const double khSlash = 1.0545726663e-27;
  const double kErg2Gev = 1 / 1.6021773349e-3;
  const double khShGev = khSlash * kErg2Gev;
  const double kYear2Sec = 3600 * 24 * 365.25;

  //
  // Bottom mesons
  // mass and life-time from PDG
  //
  addParticle("Upsilon(3S)", "Upsilon(3S)", 10.3552, kTRUE, 0, 1, "Bottonium", 200553, verbose);

  // QCD diffractive states
  addParticle("rho_diff0", "rho_diff0", 0, kTRUE, 0, 0, "QCD diffr. state", 9900110, verbose);
  addParticle("pi_diffr+", "pi_diffr+", 0, kTRUE, 0, 1, "QCD diffr. state", 9900210, verbose);
  addParticle("omega_di", "omega_di", 0, kTRUE, 0, 0, "QCD diffr. state", 9900220, verbose);
  addParticle("phi_diff", "phi_diff", 0, kTRUE, 0, 0, "QCD diffr. state", 9900330, verbose);
  addParticle("J/psi_di", "J/psi_di", 0, kTRUE, 0, 0, "QCD diffr. state", 9900440, verbose);
  addParticle("n_diffr0", "n_diffr0", 0, kTRUE, 0, 0, "QCD diffr. state", 9902110, verbose);
  addParticle("p_diffr+", "p_diffr+", 0, kTRUE, 0, 1, "QCD diffr. state", 9902210, verbose);

  // From Herwig
  addParticle("PSID    ", " ", 3.7699, kFALSE, 0.0, 0, "meson", 30443, verbose);

  addParticle("A_00    ", " ", 0.9960, kFALSE, 0.0, 0, "meson", 9000111, verbose);
  addParticle("A_0+    ", " ", 0.9960, kFALSE, 0.0, +3, "meson", 9000211, verbose);
  addParticle("A_0-    ", " ", 0.9960, kFALSE, 0.0, -3, "meson", -9000211, verbose);

  // addParticle("F0P0    ", " ", 0.9960, kFALSE, 0.0, 0, "meson",  9010221, verbose);

  addParticle("KDL_2+  ", " ", 1.773, kFALSE, 0.0, +3, "meson", 10325, verbose);
  addParticle("KDL_2-  ", " ", 1.773, kFALSE, 0.0, -3, "meson", -10325, verbose);

  addParticle("KDL_20  ", " ", 1.773, kFALSE, 0.0, 0, "meson", 10315, verbose);
  addParticle("KDL_2BR0", " ", 1.773, kFALSE, 0.0, 0, "meson", -10315, verbose);

  addParticle("PI_2+   ", " ", 1.670, kFALSE, 0.0, +3, "meson", 10215, verbose);
  addParticle("PI_2-   ", " ", 1.670, kFALSE, 0.0, -3, "meson", -10215, verbose);
  addParticle("PI_20   ", " ", 1.670, kFALSE, 0.0, 0, "meson", 10115, verbose);

  addParticle("KD*+    ", " ", 1.717, kFALSE, 0.0, +3, "meson", 30323, verbose);
  addParticle("KD*-    ", " ", 1.717, kFALSE, 0.0, -3, "meson", -30323, verbose);

  addParticle("KD*0    ", " ", 1.717, kFALSE, 0.0, 0, "meson", 30313, verbose);
  addParticle("KDBR*0  ", " ", 1.717, kFALSE, 0.0, 0, "meson", -30313, verbose);

  addParticle("RHOD+   ", " ", 1.700, kFALSE, 0.0, +3, "meson", 30213, verbose);
  addParticle("RHOD-   ", " ", 1.700, kFALSE, 0.0, -3, "meson", -30213, verbose);
  addParticle("RHOD0   ", " ", 1.700, kFALSE, 0.0, 0, "meson", 30113, verbose);

  addParticle("ETA_2(L)", " ", 1.632, kFALSE, 0.0, 0, "meson", 10225, verbose);
  addParticle("ETA_2(H)", " ", 1.854, kFALSE, 0.0, 0, "meson", 10335, verbose);
  addParticle("OMEGA(H)", " ", 1.649, kFALSE, 0.0, 0, "meson", 30223, verbose);

  addParticle("KDH_2+  ", " ", 1.816, kFALSE, 0.0, +3, "meson", 20325, verbose);
  addParticle("KDH_2-  ", " ", 1.816, kFALSE, 0.0, -3, "meson", -20325, verbose);

  addParticle("KDH_20  ", " ", 1.816, kFALSE, 0.0, 0, "meson", 20315, verbose);
  addParticle("KDH_2BR0", " ", 1.816, kFALSE, 0.0, 0, "meson", -20315, verbose);

  addParticle("KD_3+   ", " ", 1.773, kFALSE, 0.0, +3, "meson", 327, verbose);
  addParticle("KD_3-   ", " ", 1.773, kFALSE, 0.0, -3, "meson", -327, verbose);

  addParticle("KD_30   ", " ", 1.773, kFALSE, 0.0, 0, "meson", 317, verbose);
  addParticle("KD_3BR0 ", " ", 1.773, kFALSE, 0.0, 0, "meson", -317, verbose);

  addParticle("RHO_3+  ", " ", 1.691, kFALSE, 0.0, +3, "meson", 217, verbose);
  addParticle("RHO_3-  ", " ", 1.691, kFALSE, 0.0, -3, "meson", -217, verbose);
  addParticle("RHO_30  ", " ", 1.691, kFALSE, 0.0, 0, "meson", 117, verbose);
  addParticle("OMEGA_3 ", " ", 1.667, kFALSE, 0.0, 0, "meson", 227, verbose);
  addParticle("PHI_3   ", " ", 1.854, kFALSE, 0.0, 0, "meson", 337, verbose);

  addParticle("CHI2P_B0", " ", 10.232, kFALSE, 0.0, 0, "meson", 110551, verbose);
  addParticle("CHI2P_B1", " ", 10.255, kFALSE, 0.0, 0, "meson", 120553, verbose);
  addParticle("CHI2P_B2", " ", 10.269, kFALSE, 0.0, 0, "meson", 100555, verbose);
  addParticle("UPSLON4S", " ", 10.580, kFALSE, 0.0, 0, "meson", 300553, verbose);

  // IONS
  //
  // Done by default now from Pythia6 table
  // Needed for other generators
  // So check if already defined

  int ionCode = kion + 10020;
  addParticle("Deuteron", "Deuteron", 1.875613, kTRUE, 0, 3, "Ion", ionCode, verbose);
  addAntiParticle("AntiDeuteron", -ionCode, verbose);

  ionCode = kion + 10030;
  addParticle("Triton", "Triton", 2.80925, kFALSE, khShGev / (12.33 * kYear2Sec), 3, "Ion", ionCode, verbose);
  addAntiParticle("AntiTriton", -ionCode, verbose);

  ionCode = kion + 20030;
  addParticle("HE3", "HE3", 2.80923, kFALSE, 0, 6, "Ion", ionCode, verbose);
  addAntiParticle("AntiHE3", -ionCode, verbose);

  ionCode = kion + 20040;
  addParticle("Alpha", "Alpha", 3.727379, kTRUE, khShGev / (12.33 * kYear2Sec), 6, "Ion", ionCode, verbose);
  addAntiParticle("AntiAlpha", -ionCode, verbose);

  // Special particles
  //
  addParticle("Cherenkov", "Cherenkov", 0, kFALSE, 0, 0, "Special", kspe + 50, verbose);
  addParticle("FeedbackPhoton", "FeedbackPhoton", 0, kFALSE, 0, 0, "Special", kspe + 51, verbose);
  // addParticle("Lambda1520", "Lambda1520", 1.5195, kFALSE, 0.0156, 0, "Resonance", 3124, verbose);
  // addAntiParticle("Lambda1520bar", -3124, verbose);

  // Hyper nuclei and exotica
  ionCode = 1010010030;
  addParticle("HyperTriton", "HyperTriton", 2.99131, kFALSE, 2.5e-15, 3, "Ion", ionCode, verbose);

  ionCode = -1010010030;
  addParticle("AntiHyperTriton", "AntiHyperTriton", 2.99131, kFALSE, 2.5e-15, 3, "Ion", ionCode, verbose);

  ionCode = 1010010040;
  addParticle("Hyperhydrog4", "Hyperhydrog4", 3.931, kFALSE, 2.5e-15, 3, "Ion", ionCode, verbose);

  ionCode = -1010010040;
  addParticle("AntiHyperhydrog4", "AntiHyperhydrog4", 3.931, kFALSE, 2.5e-15, 3, "Ion", ionCode, verbose);

  ionCode = 1010020040;
  addParticle("Hyperhelium4", "Hyperhelium4", 3.929, kFALSE, 2.5e-15, 6, "Ion", ionCode, verbose);

  ionCode = -1010020040;
  addParticle("AntiHyperhelium4", "AntiHyperhelium4", 3.929, kFALSE, 2.5e-15, 6, "Ion", ionCode, verbose);

  ionCode = 1010020050;
  addParticle("Hyperhelium5", "Hyperhelium5", 4.841, kFALSE, 2.5e-15, 6, "Ion", ionCode, verbose);

  ionCode = -1010020050;
  addParticle("AntiHyperhelium5", "AntiHyperhelium5", 4.841, kFALSE, 2.5e-15, 6, "Ion", ionCode, verbose);

  ionCode = 1020010040;
  addParticle("DoubleHyperhydrogen4", "DoubleHyperhydrogen4", 4.106, kFALSE, 2.5e-15, 6, "Ion", ionCode, verbose);

  ionCode = -1020010040;
  addParticle("DoubleAntiHyperhydrogen4", "DoubleAntiHyperhydrogen4", 4.106, kFALSE, 2.5e-15, 6, "Ion", ionCode,
              verbose);

  ionCode = 1010000020;
  addParticle("LambdaNeutron", "LambdaNeutron", 2.054, kFALSE, 2.5e-15, 0, "Special", ionCode, verbose);

  ionCode = -1010000020;
  addParticle("AntiLambdaNeutron", "AntiLambdaNeutron", 2.054, kFALSE, 2.5e-15, 0, "Special", ionCode, verbose);

  ionCode = 1020000020;
  addParticle("Hdibaryon", "Hdibaryon", 2.23, kFALSE, 2.5e-15, 0, "Special", ionCode, verbose);

  ionCode = -1020000020;
  addParticle("AntiHdibaryon", "AntiHdibaryon", 2.23, kFALSE, 2.5e-15, 0, "Special", ionCode, verbose);

  ionCode = 1010000030;
  addParticle("LambdaNeutronNeutron", "LambdaNeutronNeutron", 2.99, kFALSE, 2.5e-15, 0, "Special", ionCode, verbose);

  ionCode = -1010000030;
  addParticle("AntiLambdaNeutronNeutron", "AntiLambdaNeutronNeutron", 2.99, kFALSE, 2.5e-15, 0, "Special", ionCode,
              verbose);

  ionCode = 1020010020;
  addParticle("Xi0Proton", "Xi0Proton", 2.248, kFALSE, 5e-15, 3, "Ion", ionCode, verbose);

  ionCode = -1020010020;
  addParticle("AntiXi0Proton", "AntiXi0Proton", 2.248, kFALSE, 5e-15, 3, "Ion", ionCode, verbose);

  ionCode = 1030000020;
  addParticle("OmegaProton", "OmegaProton", 2.592, kFALSE, 2.5e-15, 0, "Special", ionCode, verbose);

  ionCode = -1030000020;
  addParticle("AntiOmegaProton", "AntiOmegaProton", 2.592, kFALSE, 2.5e-15, 0, "Special", ionCode, verbose);

  ionCode = 1030010020;
  addParticle("OmegaNeutron", "OmegaNeutron", 2.472, kFALSE, 0.003, 3, "Special", ionCode, verbose);

  ionCode = -1030010020;
  addParticle("AntiOmegaNeutron", "AntiOmegaNeutron", 2.472, kFALSE, 0.003, 3, "Special", ionCode, verbose);

  ionCode = 1060020020;
  addParticle("OmegaOmega", "OmegaOmega", 3.229, kFALSE, 2.5e-15, 6, "Special", ionCode, verbose);

  ionCode = -1060020020;
  addParticle("AntiOmegaOmega", "AntiOmegaOmega", 3.229, kFALSE, 2.5e-15, 6, "Special", ionCode, verbose);

  ionCode = 1010010021;
  addParticle("Lambda1405Proton", "Lambda1405Proton", 2.295, kFALSE, 0.05, 3, "Special", ionCode, verbose);

  ionCode = -1010010021;
  addParticle("AntiLambda1405Proton", "AntiLambda1405Proton", 2.295, kFALSE, 0.05, 3, "Special", ionCode, verbose);

  ionCode = 1020000021;
  addParticle("Lambda1405Lambda1405", "Lambda1405Lambda1405", 2.693, kFALSE, 0.05, 0, "Special", ionCode, verbose);

  ionCode = -1020000021;
  addParticle("AntiLambda1405Lambda1405", "AntiLambda1405Lambda1405", 2.693, kFALSE, 0.05, 0, "Special", ionCode,
              verbose);

  // Special resonances

  ionCode = 9010221;
  addParticle("f0_980", "f0_980", 0.980, kFALSE, 0.07, 0, "Resonance", ionCode, verbose);

  // ionCode = 225;
  // addParticle("f2_1270", "f2_1270", 1.275, kFALSE, 0.185, 0, "Resonance", ionCode, verbose);

  // Xi-/+ (1820)
  ionCode = 123314;
  addParticle("Xi_Minus_1820", "Xi_Minus_1820", 1.8234, kFALSE, 0.024, -3, "Resonance", ionCode, verbose);
  addParticle("Xi_Plus_1820", "Xi_Plus_1820", 1.8234, kFALSE, 0.024, 3, "Resonance", -ionCode, verbose);

  // Xi0 (1820)
  ionCode = 123324;
  addParticle("Xi_0_1820", "Xi_0_1820", 1.8234, kFALSE, 0.024, 0, "Resonance", ionCode, verbose);
  addParticle("Xi_0_Bar_1820", "Xi_0_Bar_1820", 1.8234, kFALSE, 0.024, 0, "Resonance", -ionCode, verbose);

  // Ps - hidden strange (s-sbar) pentaquarks
  ionCode = 9322134;
  addParticle("ps_2100", "ps_2100", 2.100, kFALSE, 0.040, 3, "Resonance", ionCode, verbose);
  addParticle("anti-ps_2100", "anti-ps_2100", 2.100, kFALSE, 0.040, -3, "Resonance", -ionCode, verbose);

  ionCode = 9322136;
  addParticle("ps_2500", "ps_2500", 2.500, kFALSE, 0.040, 3, "Resonance", ionCode, verbose);
  addParticle("anti-ps_2500", "anti-ps_2500", 2.500, kFALSE, 0.040, -3, "Resonance", -ionCode, verbose);

  // d*(2380) - dibaryon resonance
  ionCode = 900010020; // Arbitrary choice - as deuteron but with leading 9 instead of 10
  addParticle("d*_2380", "d*_2380", 2.38, kFALSE, 0.070, 3, "Resonance", ionCode, verbose);
  addAntiParticle("d*_2380_bar", -ionCode, verbose);
}
