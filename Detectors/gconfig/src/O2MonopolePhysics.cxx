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
#include "CommonUtils/ConfigurableParam.h"

#include <boost/property_tree/ptree.hpp> // needed to instantiate getValueAs<>

#include <TGeoManager.h>

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

#include <G4ChargeState.hh>
#include <G4ChordFinder.hh>
#include <G4ClassicalRK4.hh>
#include <G4EquationOfMotion.hh>
#include <G4EventManager.hh>
#include <G4FieldManager.hh>
#include <G4MagIntegratorStepper.hh>
#include <G4MagneticField.hh>
#include <G4Track.hh>
#include <G4TrackingManager.hh>
#include <G4Transportation.hh>
#include <G4TransportationManager.hh>

#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/PhysicalConstants.h>

#include <array>
#include <cmath>
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

inline bool isMonopolePDG(int pdg)
{
  const int abspdg = std::abs(pdg);
  return abspdg == 4110000 || abspdg == 4120000;
}

/// GEANT4 only queries the magnetic field when a particle has non-zero electric
/// charge or non-zero magnetic moment (μ) and the monopole has neither. So this is a workaround
/// so that G4Transportation queries the field for it
constexpr G4double gMonopoleFieldGateMoment = 1.0e-20;

// Extent of the TPC drift region
constexpr G4double gTPCFieldCageRMin = 83.5 * CLHEP::cm;
constexpr G4double gTPCFieldCageRMax = 254.5 * CLHEP::cm;
constexpr G4double gTPCFieldCageZMax = 249.525 * CLHEP::cm;

/// Magnitude of the TPC drift field, in Geant4 units; 0 when there is no TPC.
inline double tpcDriftFieldMagnitude()
{
  if (gGeoManager == nullptr || gGeoManager->GetVolume("TPC_Drift") == nullptr) {
    LOG(info) << "O2MonopolePhysics: no TPC in the geometry of this run, "
                 "the monopole is not coupled to a drift field";
    return 0.;
  }
  try {
    const double valueKVPerCm = o2::conf::ConfigurableParam::getValueAs<float>("TPCGEMParam.ElectricField[0]");
    return valueKVPerCm * CLHEP::kilovolt / CLHEP::cm;
  } catch (...) {
    LOG(warn) << "O2MonopolePhysics: the TPC is in the geometry but TPCGEMParam is not "
                 "registered; no drift-field coupling for the monopole";
    return 0.;
  }
}

/// Electric field of the TPC drift region at a specific position.
/// Returns false outside the field cage.
inline bool tpcDriftField(const G4double position[3], G4double driftField, G4double E[3])
{
  const G4double z = position[2];
  if (std::fabs(z) >= gTPCFieldCageZMax) {
    return false;
  }
  const G4double r2 = position[0] * position[0] + position[1] * position[1];
  if (r2 < gTPCFieldCageRMin * gTPCFieldCageRMin || r2 > gTPCFieldCageRMax * gTPCFieldCageRMax) {
    return false;
  }
  E[0] = 0.;
  E[1] = 0.;
  E[2] = (z >= 0.) ? -driftField : driftField;
  return true;
}
} // namespace

//____________________________________________________________________________
/// Equation of motion implementing the dual Lorentz force of a magnetic charge.
class O2MonopoleEquation : public G4EquationOfMotion
{
 public:
  /// \param field                    the magnetic field to integrate in
  /// \param magneticChargeEplusUnits monopole magnetic charge expressed in eplus
  ///                                 units (one Dirac charge = 1/(2*alpha) ~ 68.5)
  /// \param tpcDriftFieldGeant4      TPC drift field in Geant4 units
  O2MonopoleEquation(G4Field* field, double magneticChargeEplusUnits, double tpcDriftFieldGeant4)
    : G4EquationOfMotion(field),
      mMagneticChargeEplus(magneticChargeEplusUnits),
      mTPCDriftField(tpcDriftFieldGeant4)
  {
  }

  void SetChargeMomentumMass(G4ChargeState particleChargeState,
                             G4double /*momentum*/, G4double particleMass) override
  {
    mElCharge = CLHEP::eplus * particleChargeState.GetCharge() * CLHEP::c_light;
    mMassCof = particleMass * particleMass;

    // Non-zero only while an O2 monopole is being transported. The sign follows
    // the PDG sign, so monopole and anti-monopole are pushed in opposite
    // directions along B.
    double signedMagneticCharge = 0.;
    int pdg = 0;
    if (const G4Track* track = currentTrack()) {
      pdg = track->GetDefinition()->GetPDGEncoding();
      if (isMonopolePDG(pdg)) {
        signedMagneticCharge = (pdg > 0) ? mMagneticChargeEplus : -mMagneticChargeEplus;
      }
    }
    mMagCharge = CLHEP::eplus * signedMagneticCharge * CLHEP::c_light;
  }

  void EvaluateRhsGivenB(const G4double y[], const G4double B[3], G4double dydx[]) const override
  {
    const G4double pSquared = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    if (pSquared <= 0.) {
      for (int i = 0; i < 8; ++i) {
        dydx[i] = 0.;
      }
      return;
    }
    const G4double energy = std::sqrt(pSquared + mMassCof);
    const G4double pModuleInverse = 1.0 / std::sqrt(pSquared);
    const G4double cofEl = mElCharge * pModuleInverse;
    const G4double cofMag = mMagCharge * energy * pModuleInverse;

    dydx[0] = y[3] * pModuleInverse;
    dydx[1] = y[4] * pModuleInverse;
    dydx[2] = y[5] * pModuleInverse;

    // magnetic charge -> force along B; electric charge -> the usual v x B
    dydx[3] = cofMag * B[0] + cofEl * (y[4] * B[2] - y[5] * B[1]);
    dydx[4] = cofMag * B[1] + cofEl * (y[5] * B[0] - y[3] * B[2]);
    dydx[5] = cofMag * B[2] + cofEl * (y[3] * B[1] - y[4] * B[0]);

    // Coupling of the magnetic charge to an electric field: the full dual
    // Lorentz force is F = g*(B - v x E/c^2).
    // mMagCharge is zero for every non-monopole, so nothing else is affected.
    if (mMagCharge != 0. && mTPCDriftField > 0.) {
      G4double E[3] = {0., 0., 0.};
      if (tpcDriftField(y, mTPCDriftField, E)) {
        // Relative to cofMag this carries 1/(c*energy), so the term is of order
        // beta*E/(c*B) compared with the g*B term (similar to G4MonopoleEq, which uses
        // d(p)/ds = g*(c*energy*B - p x E)/(p*c)).
        const G4double cofMagE = mMagCharge * pModuleInverse / CLHEP::c_light;
        const G4double dEx = cofMagE * (y[4] * E[2] - y[5] * E[1]);
        const G4double dEy = cofMagE * (y[5] * E[0] - y[3] * E[2]);
        const G4double dEz = cofMagE * (y[3] * E[1] - y[4] * E[0]);
        dydx[3] -= dEx;
        dydx[4] -= dEy;
        dydx[5] -= dEz;
      }
    }

    dydx[6] = 0.;                                       // not used
    dydx[7] = energy * pModuleInverse / CLHEP::c_light; // inverse velocity
  }

 private:
  static const G4Track* currentTrack()
  {
    auto* eventManager = G4EventManager::GetEventManager();
    if (eventManager == nullptr) {
      return nullptr;
    }
    auto* trackingManager = eventManager->GetTrackingManager();
    return trackingManager != nullptr ? trackingManager->GetTrack() : nullptr;
  }

  double mMagneticChargeEplus;  ///< |g| in eplus units (1 g_D = 1/(2*alpha) ~ 68.5)
  G4double mTPCDriftField = 0.; ///< TPC drift field, Geant4 units; 0 disables the coupling
  G4double mElCharge = 0.;
  G4double mMagCharge = 0.;
  G4double mMassCof = 0.;
};

//____________________________________________________________________________
/// Make the global field integrate O2MonopoleEquation instead of the default
/// electric-charge-only equation.
///
/// SetUserEquationOfMotion() in Geant4-VMC is not usable here: it
/// registers the object with TG4GeometryManager, and the field integrator is
/// built before that registration is consulted, so the equation is never
/// actually called.
///
/// \param magneticChargeEplusUnits monopole magnetic charge in eplus units
/// \param tpcDriftFieldGeant4      TPC drift field in Geant4 units (0 = disabled)
inline void installMonopoleFieldIntegrator(double magneticChargeEplusUnits, double tpcDriftFieldGeant4)
{
  auto* transportationManager = G4TransportationManager::GetTransportationManager();
  auto* fieldManager = transportationManager != nullptr ? transportationManager->GetFieldManager() : nullptr;
  if (fieldManager == nullptr) {
    LOG(error) << "O2MonopolePhysics: no G4FieldManager, monopole equation of motion NOT installed";
    return;
  }
  auto* magneticField =
    const_cast<G4MagneticField*>(dynamic_cast<const G4MagneticField*>(fieldManager->GetDetectorField()));
  if (magneticField == nullptr) {
    LOG(error) << "O2MonopolePhysics: no magnetic field attached to the field manager, "
                  "monopole equation of motion NOT installed";
    return;
  }

  auto* equation = new O2MonopoleEquation(magneticField, magneticChargeEplusUnits, tpcDriftFieldGeant4);
  // A generic (non-helix) integrator is required: the helix steppers hard-code
  // the constant-curvature motion of an electric charge. 8 variables so that the
  // time-of-flight component is integrated too.
  auto* stepper = new G4ClassicalRK4(equation, 8);

  // keep the accuracy Geant4-VMC configured for this field
  auto* previous = fieldManager->GetChordFinder();
  const G4double stepMinimum = 1.0e-2 * CLHEP::mm;
  auto* chordFinder = new G4ChordFinder(magneticField, stepMinimum, stepper);
  if (previous != nullptr) {
    chordFinder->SetDeltaChord(previous->GetDeltaChord());
  }
  fieldManager->SetChordFinder(chordFinder);

  if (tpcDriftFieldGeant4 > 0.) {
    LOG(info) << "O2MonopolePhysics: monopole equation of motion installed (F = g*(B - v x E/c^2)), "
                 "magnetic charge = "
              << magneticChargeEplusUnits << " eplus, TPC drift field = "
              << tpcDriftFieldGeant4 / (CLHEP::volt / CLHEP::cm) << " V/cm";
  } else {
    LOG(info) << "O2MonopolePhysics: monopole equation of motion installed (F = g*B), magnetic charge = "
              << magneticChargeEplusUnits << " eplus (TPC drift field coupling disabled)";
  }
}

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

      // G4Transportation only looks up the field when the particle has a
      // non-zero electric charge or a non-zero magnetic moment (μ) and a monopole has
      // neither, so the field would never be queried. Hence a gate magnetic moment is set
      // to allow the query to happen.
      if (particle->GetPDGMagneticMoment() == 0.) {
        particle->SetPDGMagneticMoment(gMonopoleFieldGateMoment);
      }
      ++nAttached;
      LOG(info) << "O2MonopolePhysics: attached G4mplIonisation to "
                << particle->GetParticleName() << " (PDG " << pdg
                << "), magnetic charge = " << mMagneticCharge / CLHEP::eplus << " eplus";
    }
    if (nAttached == 0) {
      LOG(warning) << "O2MonopolePhysics: no monopole particle found; no ionisation attached";
    }

    // This static switch is what makes G4Transportation consider the μ of the monopole
    // It is global, so electrically neutral particles that already carry a momentum (neutrons) are
    // now propagated through the field as well; O2MonopoleEquation gives them
    // exactly zero force, so their trajectories are unchanged.
    G4Transportation::EnableMagneticMoment(true);

    // Deflect the monopole in the field as well; without this only the energy
    // loss above would act and the monopole would fly straight through, since
    // its electric charge (and hence the usual Lorentz force) is zero.
    installMonopoleFieldIntegrator(mMagneticCharge / CLHEP::eplus, tpcDriftFieldMagnitude());
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
