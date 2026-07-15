// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file O2MonopolePhysics.cxx
/// \brief Opt-in magnetic-monopole ionisation physics for the O2 Geant4 engine.
/// \author M+Giacalone - July 2026
///
/// O2 defines the BSM monopole particles (PDG +-4110000 / +-4120000) via
/// O2MCApplication::AddParticles(), so they are transported by Geant4, but no
/// energy-loss process is attached to them by any of the stock reference
/// physics lists.
///
/// This file adds - the magnetic-monopole ionisation process G4mplIonisation
/// (Ahlen stopping power, Rev. Mod. Phys. 52 (1980) 121) to those particles, when requested.
/// The implementation takes Geant4 `examples/extended/exoticphysics/monopole` example (and the
/// equivalent geant4_vmc "monopole" physics builder). The main difference is that
/// here the process is attached to the pre-existing O2 monopole particles (with correct PDG ID()
/// rather than to a freshly created G4Monopole
///
/// G4mplIonisation and G4mplIonisationWithDeltaModel are taken directly from the standard
/// Geant4 libraries (libG4processes)

#include "SimSetup/O2MonopolePhysics.h"

#include <fairlogger/Logger.h>

#include "TG4RunConfiguration.h"
#include "TG4ComposedPhysicsList.h"

#include <G4VUserPhysicsList.hh>
#include <G4ParticleTable.hh>
#include <G4ParticleDefinition.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessVector.hh>
#include <G4VEnergyLossProcess.hh>
#include <G4mplIonisation.hh>

#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/PhysicalConstants.h>

#include <array>
#include <vector>

namespace o2
{
namespace g4config
{

namespace
{
// PDG codes of the O2 monopole species, as defined in
// O2MCApplication::AddParticles() and O2DatabasePDG:
//   +-4110000 : "symmetric"  monopoles
//   +-4120000 : "asymmetric" monopoles
constexpr std::array<int, 4> gMonopolePDGs = {4110000, -4110000, 4120000, -4120000};
} // namespace

//____________________________________________________________________________
/// Minimal physics list whose only job is to attach the magnetic-monopole
/// ionisation process to the already-defined O2 monopole particles.
///
/// It is implemented as a G4VUserPhysicsList so that it can be handed to VMC's
/// TG4ComposedPhysicsList::AddPhysicsList(). It intentionally does NOT create
/// any particle and does NOT add transportation (both are handled by the
/// reference physics list), it only adds one extra process.
class O2MonopolePhysics : public G4VUserPhysicsList
{
 public:
  /// \param magneticChargeEplus monopole magnetic charge in Geant4 internal
  ///                            (eplus) units
  explicit O2MonopolePhysics(double magneticChargeEplus) : mMagneticCharge(magneticChargeEplus) {}

  // Particles are already defined by O2MCApplication::AddParticles().
  void ConstructParticle() override {}

  // Cuts are handled by the reference physics list.
  void SetCuts() override {}

  void ConstructProcess() override
  {
    auto* table = G4ParticleTable::GetParticleTable();
    int nAttached = 0;
    for (int pdg : gMonopolePDGs) {
      auto* particle = table->FindParticle(pdg);
      if (particle == nullptr) {
        // Not necessarily an error: only the species actually produced by the
        // generator strictly need this. Report and continue.
        LOG(info) << "O2MonopolePhysics: monopole PDG " << pdg
                  << " not present in the particle table, skipping";
        continue;
      }
      auto* pmanager = particle->GetProcessManager();
      if (pmanager == nullptr) {
        LOG(warning) << "O2MonopolePhysics: no process manager for PDG " << pdg << ", skipping";
        continue;
      }
      // The reference EM physics list attaches the standard hadron ionisation
      // (hIoni) to the monopole, which is wrong and makes G4LossTableManager crash
      // while merging the two dE/dx tables.
      // Any pre-existing energy-loss process is removed so that mplIoni is the
      // monopole single, correct ionisation process. This is taken from Geant4
      // exoticphysics/monopole example
      {
        std::vector<G4VProcess*> toRemove;
        G4ProcessVector* plist = pmanager->GetProcessList();
        for (G4int ip = 0; ip < static_cast<G4int>(plist->size()); ++ip) {
          G4VProcess* proc = (*plist)[ip];
          if (dynamic_cast<G4VEnergyLossProcess*>(proc) != nullptr) {
            toRemove.push_back(proc);
          }
        }
        for (auto* proc : toRemove) {
          LOG(info) << "O2MonopolePhysics: removing pre-existing energy-loss process "
                    << proc->GetProcessName() << " from " << particle->GetParticleName();
          pmanager->RemoveProcess(proc);
        }
      }
      // Ordering (AtRest, AlongStep, PostStep) = (-1, 1, 1) as in the Geant4
      // monopole example: continuous-and-discrete energy loss, not active at rest.
      pmanager->AddProcess(new G4mplIonisation(mMagneticCharge), -1, 1, 1);
      ++nAttached;
      LOG(info) << "O2MonopolePhysics: attached G4mplIonisation to "
                << particle->GetParticleName() << " (PDG " << pdg
                << "), magnetic charge = " << mMagneticCharge / CLHEP::eplus << " eplus";
    }
    if (nAttached == 0) {
      LOG(warning) << "O2MonopolePhysics: no monopole particle found; no ionisation attached";
    }
  }

 private:
  double mMagneticCharge; ///< magnetic charge in Geant4 internal (eplus) units
};

//____________________________________________________________________________
/// TG4RunConfiguration that appends O2MonopolePhysics to the composed physics
/// list which VMC builds for the requested reference list.
class O2G4RunConfiguration : public TG4RunConfiguration
{
 public:
  O2G4RunConfiguration(const TString& userGeometry, const TString& physicsList,
                       const TString& specialProcess, Bool_t specialStacking,
                       Bool_t mtApplication, double magneticChargeEplus)
    : TG4RunConfiguration(userGeometry, physicsList, specialProcess, specialStacking, mtApplication),
      mMagneticCharge(magneticChargeEplus)
  {
  }

  G4VUserPhysicsList* CreatePhysicsList() override
  {
    G4VUserPhysicsList* physicsList = TG4RunConfiguration::CreatePhysicsList();
    if (auto* composed = dynamic_cast<TG4ComposedPhysicsList*>(physicsList)) {
      composed->AddPhysicsList(new O2MonopolePhysics(mMagneticCharge));
      LOG(info) << "O2G4RunConfiguration: monopole ionisation physics registered "
                   "on the composed physics list";
    } else {
      LOG(error) << "O2G4RunConfiguration: physics list is not a TG4ComposedPhysicsList, "
                    "monopole ionisation could NOT be enabled";
    }
    return physicsList;
  }

 private:
  double mMagneticCharge; ///< magnetic charge in Geant4 internal (eplus) units
};

//____________________________________________________________________________
TG4RunConfiguration* createMonopoleRunConfiguration(const TString& userGeometry,
                                                    const TString& physicsList,
                                                    const TString& specialProcess,
                                                    bool specialStacking,
                                                    bool mtApplication,
                                                    double magneticChargeDirac)
{
  // One Dirac charge g_D = eplus / (2*alpha) ~= 68.5 eplus (Dirac quantisation).
  // magneticChargeDirac counts Dirac charges (1.0 == classic single monopole),
  // matching the "g_1" convention used by the O2 monopole generator input.
  const double gDirac = CLHEP::eplus / (2.0 * CLHEP::fine_structure_const);
  const double magneticChargeEplus = magneticChargeDirac * gDirac;
  LOG(info) << "Monopole ionisation enabled: magnetic charge = " << magneticChargeDirac
            << " Dirac charge(s) = " << magneticChargeEplus / CLHEP::eplus << " eplus";
  return new O2G4RunConfiguration(userGeometry, physicsList, specialProcess,
                                  specialStacking, mtApplication, magneticChargeEplus);
}

} // namespace g4config
} // namespace o2
