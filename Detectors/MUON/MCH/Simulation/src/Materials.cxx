// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Materials.cxx
/// \brief  Implementation of the MCH materials definitions
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   22 march 2018

#include "Materials.h"

#include "DetectorsBase/Detector.h" // for the magnetic field
#include "DetectorsBase/MaterialManager.h"

using namespace std;

namespace o2
{
namespace mch
{

/// Definition of constants for the elements
/// The atomic number and the atomic masse values are taken from the 2016 PDG booklet
/// For the radiation and absorption lengths, we let the Virtual Monte-Carlo compute them internally

// Hydrogen
const float kZHydrogen = 1.;
const float kAHydrogen = 1.00794;

// Carbon
const float kZCarbon = 6.;
const float kACarbon = 12.0107;
const float kDensCarbon = 2.265;

// Nitrogen
const float kZNitrogen = 7.;
const float kANitrogen = 14.0067;

// Oxygen
const float kZOxygen = 8.;
const float kAOxygen = 15.9994;

// Aluminium
const float kZAluminium = 13.;
const float kAAluminium = 26.9815385;
const float kDensAluminium = 2.699;

// Silicon
const float kZSilicon = 14.;
const float kASilicon = 28.0855;

// Argon
const float kZArgon = 18.;
const float kAArgon = 39.948;

// Chromium
const float kZChromium = 24.;
const float kAChromium = 51.9961;

// Iron
const float kZIron = 26.;
const float kAIron = 55.845;

// Nickel
const float kZNickel = 28.;
const float kANickel = 58.6934;

// Copper
const float kZCopper = 29.;
const float kACopper = 63.546;
const float kDensCopper = 8.96;

/// Tracking parameters (values taken from AliMUONCommonGeometryBuilder)
const float kEpsil = 0.001; // Tracking precision [cm]

// negative values below means "let the MC transport code compute the values"
const float kMaxfd = -20.;  // Maximum deflection angle due to magnetic field
const float kStemax = -1.;  // Maximum displacement for multiple scattering [cm]
const float kDeemax = -0.3; // Maximum fractional energy loss, DLS
const float kStmin = -0.8;  // Minimum step due to continuous processes [cm]

const char* kModuleName = "MCH";

void createMaterials()
{

  /// Create all the materials needed to build the MCH geometry

  int imat = 0;                                 // counter of material ID
  const bool kIsSens = true, kIsUnsens = false; // (un)sensitive medium
  int fieldType;                                // magnetic field type
  float maxField;                               // maximum magnetic field value

  // get the magnetic field parameters
  base::Detector::initFieldTrackingParams(fieldType, maxField);

  auto& mgr = o2::base::MaterialManager::Instance();

  /// Tracking gas : Ar 80% + CO2 20%
  const int nGas = 3;
  float aGas[nGas] = {kAArgon, kACarbon, kAOxygen};
  float zGas[nGas] = {kZArgon, kZCarbon, kZOxygen};
  float wGas[nGas] = {0.8, 1. / 15, 2. / 15}; // Relative weight of each atom in the gas
  float dGas = 0.001821;                      // according to AliMUONCommonGeometryBuilder

  mgr.Mixture(kModuleName, ++imat, "Ar 80% + CO2 20%", aGas, zGas, dGas, nGas, wGas);
  mgr.Medium(kModuleName, Medium::Gas, "Tracking gas", imat, kIsSens, fieldType, maxField, kMaxfd, kStemax, kDeemax,
             kEpsil, kStmin);

  /// Carbon
  mgr.Material(kModuleName, ++imat, "Carbon", kACarbon, kZCarbon, kDensCarbon, 0., 0.);
  mgr.Medium(kModuleName, Medium::Carbon, "Carbon", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil,
             kStmin);

  /// Nomex : C14 H10 N2 O2 (changed w.r.t AliMUONCommonGeometryBuilder)
  const int nNomex = 4;
  float aNomex[nNomex] = {kACarbon, kAHydrogen, kANitrogen, kAOxygen};
  float zNomex[nNomex] = {kZCarbon, kZHydrogen, kZNitrogen, kZOxygen};
  float wNomex[nNomex] = {14., 10., 2., 2.};
  // honey comb
  float dHoneyNomex = 0.024; // according to AliMUONCommonGeometryBuilder
  mgr.Mixture(kModuleName, ++imat, "Nomex (honey comb)", aNomex, zNomex, dHoneyNomex, -nNomex, wNomex);
  mgr.Medium(kModuleName, Medium::HoneyNomex, "Nomex (honey comb)", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax,
             kDeemax, kEpsil, kStmin);
  // bulk
  float dBulkNomex = 1.43; // according to AliMUONCommonGeometryBuilder
  mgr.Mixture(kModuleName, ++imat, "Nomex (bulk)", aNomex, zNomex, dBulkNomex, -nNomex, wNomex);
  mgr.Medium(kModuleName, Medium::BulkNomex, "Nomex (bulk)", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax,
             kEpsil, kStmin);

  /// Noryl 731 (ALICE-INT-2002-17) : C8 H8 O
  const int nNoryl = 3;
  float aNoryl[nNoryl] = {kACarbon, kAHydrogen, kAOxygen};
  float zNoryl[nNoryl] = {kZCarbon, kZHydrogen, kZOxygen};
  float wNoryl[nNoryl] = {8., 8., 1.};
  float dNoryl = 1.06;
  mgr.Mixture(kModuleName, ++imat, "Noryl", aNoryl, zNoryl, dNoryl, -nNoryl, wNoryl);
  mgr.Medium(kModuleName, Medium::Noryl, "Noryl", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil,
             kStmin);

  /// Copper
  mgr.Material(kModuleName, ++imat, "Copper", kACopper, kZCopper, kDensCopper, 0., 0.);
  mgr.Medium(kModuleName, Medium::Copper, "Copper", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil,
             kStmin);

  /// FR4 : O292 Si68 C462 H736 (from AliRoot)
  const int nFR4 = 4;
  float aFR4[nFR4] = {kAOxygen, kASilicon, kACarbon, kAHydrogen};
  float zFR4[nFR4] = {kZOxygen, kZSilicon, kZCarbon, kZHydrogen};
  float wFR4[nFR4] = {292, 68, 462, 736}; // Relative weight of each atom
  float dFR4 = 1.8;                       // changed w.r.t AliRoot after investigation
  mgr.Mixture(kModuleName, ++imat, "FR4", aFR4, zFR4, dFR4, -nFR4, wFR4);
  mgr.Medium(kModuleName, Medium::FR4, "FR4", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil, kStmin);

  /// Rohacell : C9 H13 N1 O2
  const int nRoha = 4;
  float aRoha[nRoha] = {kACarbon, kAHydrogen, kANitrogen, kAOxygen};
  float zRoha[nRoha] = {kZCarbon, kZHydrogen, kZNitrogen, kZOxygen};
  float wRoha[nRoha] = {9., 13., 1., 2.};

  float dRoha = 0.03; // from AliMUONCommonGeometryBuilder
  mgr.Mixture(kModuleName, ++imat, "Rohacell", aRoha, zRoha, dRoha, -nRoha, wRoha);
  mgr.Medium(kModuleName, Medium::Rohacell, "Rohacell", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax,
             kDeemax, kEpsil, kStmin);
  // for station 1
  float dSt1Roha = 0.053; // from AliMUONCommonGeometryBuilder
  mgr.Mixture(kModuleName, ++imat, "Rohacell (st 1)", aRoha, zRoha, dSt1Roha, -nRoha, wRoha);
  mgr.Medium(kModuleName, Medium::St1Rohacell, "Rohacell (st 1)", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax,
             kDeemax, kEpsil, kStmin);

  /// Glue (Araldite 2011, ALICE-INT-2002-17) : C10 H25 N3
  const int nGlue = 3;
  float aGlue[nGlue] = {kACarbon, kAHydrogen, kANitrogen};
  float zGlue[nGlue] = {kZCarbon, kZHydrogen, kZNitrogen};
  float wGlue[nGlue] = {10., 25., 3.};
  float dGlue = 1.066;
  mgr.Mixture(kModuleName, ++imat, "Glue", aGlue, zGlue, dGlue, -nGlue, wGlue);
  mgr.Medium(kModuleName, Medium::Glue, "Glue", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil, kStmin);

  /// Plastic (definition taken from AliMUONSt1GeometryBuilder)
  const int nPlastic = 2;
  float aPlastic[nPlastic] = {kACarbon, kAHydrogen};
  float zPlastic[nPlastic] = {kZCarbon, kZHydrogen};
  float wPlastic[nPlastic] = {1, 1};
  float dPlastic = 1.107;
  mgr.Mixture(kModuleName, ++imat, "Plastic", aPlastic, zPlastic, dPlastic, -nPlastic, wPlastic);
  mgr.Medium(kModuleName, Medium::Plastic, "Plastic", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil,
             kStmin);

  /// Epoxy : C18 H19 O3 (to be confirmed)
  const int nEpoxy = 3;
  float aEpoxy[nEpoxy] = {kACarbon, kAHydrogen, kAOxygen};
  float zEpoxy[nEpoxy] = {kZCarbon, kZHydrogen, kZOxygen};
  float wEpoxy[nEpoxy] = {18, 19, 3};
  float dEpoxy = 1.23; // from MFT, to be confirmed
  mgr.Mixture(kModuleName, ++imat, "Epoxy", aEpoxy, zEpoxy, dEpoxy, -nEpoxy, wEpoxy);
  mgr.Medium(kModuleName, Medium::Epoxy, "Epoxy", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil,
             kStmin);

  /// Stainless steel : Fe(73%) Cr(18%) Ni(9%)
  const int nInox = 3;
  float aInox[nInox] = {kAIron, kAChromium, kANickel};
  float zInox[nInox] = {kZIron, kZChromium, kZNickel};
  float wInox[nInox] = {73., 18., 9.};
  float dInox = 7.93; // from AliMUONSt1GeometryBuilder
  mgr.Mixture(kModuleName, ++imat, "Inox", aInox, zInox, dInox, -nInox, wInox);
  mgr.Medium(kModuleName, Medium::Inox, "Inox", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil, kStmin);

  /// Aluminium
  mgr.Material(kModuleName, ++imat, "Aluminium", kAAluminium, kZAluminium, kDensAluminium, 0., 0.);
  mgr.Medium(kModuleName, Medium::Aluminium, "Aluminium", imat, kIsUnsens, fieldType, maxField, kMaxfd, kStemax, kDeemax,
             kEpsil, kStmin);
}

TGeoMedium* assertMedium(int imed)
{
  auto& mgr = o2::base::MaterialManager::Instance();
  auto med = mgr.getTGeoMedium(kModuleName, imed);
  if (med == nullptr) {
    throw runtime_error("Could not retrieve medium " + to_string(imed) + " for " + kModuleName);
  }
  return med;
}

} // namespace mch
} // namespace o2
