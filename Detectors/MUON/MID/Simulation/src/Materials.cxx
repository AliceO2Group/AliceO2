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
/// \brief  Implementation of the MID materials definitions
/// \author Florian Damas <florian.damas@cern.ch>
/// \date   20 june 2018

#include "Materials.h"

#include "DetectorsBase/Detector.h" // for the magnetic field
#include "DetectorsBase/MaterialManager.h"

namespace o2
{
namespace mid
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

// Nitrogen
const float kZNitrogen = 7.;
const float kANitrogen = 14.0067;

// Oxygen
const float kZOxygen = 8.;
const float kAOxygen = 15.9994;

// Fluorine
const float kZFluorine = 9.;
const float kAFluorine = 18.998403163;

// Aluminium
const float kZAluminium = 13.;
const float kAAluminium = 26.9815385;
const float kDensAluminium = 2.699;

// Sulfur
const float kZSulfur = 16.;
const float kASulfur = 32.06;

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

const char* kModuleName = "MID";

void createMaterials()
{

  /// Create all the materials needed to build the MID geometry

  int imat = 0;   // counter of material ID
  int fieldType;  // magnetic field type
  float maxField; // maximum magnetic field value

  // get the magnetic field parameters
  base::Detector::initFieldTrackingParams(fieldType, maxField);

  auto& mgr = o2::base::MaterialManager::Instance();

  /// Trigger gas : C2 H2 F4 - Isobutane(C4 H10) - SF6 (89.7%+10%+0.3%)
  const int nGas = 4;
  float aGas[nGas] = {kACarbon, kAHydrogen, kAFluorine, kASulfur};
  float zGas[nGas] = {kZCarbon, kZHydrogen, kZFluorine, kZSulfur};
  float wGas[nGas] = {89.7 * 2 + 10 * 4, 89.7 * 2 + 10 * 4, 89.7 * 4 + 0.3 * 6, 0.3};
  float dGas = 0.0031463;

  mgr.Mixture(kModuleName, ++imat, "Gas", aGas, zGas, dGas, -nGas, wGas);
  mgr.Medium(kModuleName, Medium::Gas, "Gas", imat, 1, fieldType, maxField, kMaxfd, kStemax, kDeemax,
             kEpsil, kStmin);

  /// Bakelite : C6 H6-O.C H2 O
  const int nBake = 3;
  float aBake[nBake] = {kACarbon, kAHydrogen, kAOxygen};
  float zBake[nBake] = {kZCarbon, kZHydrogen, kZOxygen};
  float wBake[nBake] = {7., 8., 2.};
  float dBake = 1.4;

  mgr.Mixture(kModuleName, ++imat, "Bakelite", aBake, zBake, dBake, -nBake, wBake);
  mgr.Medium(kModuleName, Medium::Bakelite, "Bakelite", imat, 0, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil, kStmin);

  /// Stainless steel : Fe(73%) Cr(18%) Ni(9%)
  const int nInox = 3;
  float aInox[nInox] = {kAIron, kAChromium, kANickel};
  float zInox[nInox] = {kZIron, kZChromium, kZNickel};
  float wInox[nInox] = {73., 18., 9.};
  float dInox = 7.93;

  mgr.Mixture(kModuleName, ++imat, "Stainless steel", aInox, zInox, dInox, -nInox, wInox);
  mgr.Medium(kModuleName, Medium::Inox, "Inox", imat, 0, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil, kStmin);

  /// Aluminium
  mgr.Material(kModuleName, ++imat, "Aluminium", kAAluminium, kZAluminium, kDensAluminium, 0., 0.);
  mgr.Medium(kModuleName, Medium::Aluminium, "Aluminium", imat, 0, fieldType, maxField, kMaxfd, kStemax, kDeemax,
             kEpsil, kStmin);

  /// Copper
  mgr.Material(kModuleName, ++imat, "Copper", kACopper, kZCopper, kDensCopper, 0., 0.);
  mgr.Medium(kModuleName, Medium::Copper, "Copper", imat, 0, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil,
             kStmin);

  /// Mylar PET (C8 H10 O4)
  const int nMylar = 3;
  float aMylar[nMylar] = {kACarbon, kAHydrogen, kAOxygen};
  float zMylar[nMylar] = {kZCarbon, kZHydrogen, kZOxygen};
  float wMylar[nMylar] = {8., 10., 4.};
  float dMylar = 1.38;

  mgr.Mixture(kModuleName, ++imat, "Mylar", aMylar, zMylar, dMylar, -nMylar, wMylar);
  mgr.Medium(kModuleName, Medium::Mylar, "Mylar", imat, 0, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil, kStmin);

  /// Styrofoam (C8 H8)
  const int nStyro = 2;
  float aStyro[nStyro] = {kACarbon, kAHydrogen};
  float zStyro[nStyro] = {kZCarbon, kZHydrogen};
  float wStyro[nStyro] = {8., 8.};
  float dStyro = 0.028;

  mgr.Mixture(kModuleName, ++imat, "Styrofoam", aStyro, zStyro, dStyro, -nStyro, wStyro);
  mgr.Medium(kModuleName, Medium::Styrofoam, "Styrofoam", imat, 0, fieldType, maxField, kMaxfd, kStemax, kDeemax, kEpsil, kStmin);

  /// Nomex : C14 H10 N2 O2
  const int nNomex = 4;
  float aNomex[nNomex] = {kACarbon, kAHydrogen, kANitrogen, kAOxygen};
  float zNomex[nNomex] = {kZCarbon, kZHydrogen, kZNitrogen, kZOxygen};
  float wNomex[nNomex] = {14., 10., 2., 2.};
  float dNomex = 1.38;
  mgr.Mixture(kModuleName, ++imat, "Nomex", aNomex, zNomex, dNomex, -nNomex, wNomex);
  mgr.Medium(kModuleName, Medium::Nomex, "Nomex", imat, 0, fieldType, maxField, kMaxfd, kStemax,
             kDeemax, kEpsil, kStmin);
}

TGeoMedium* assertMedium(int imed)
{
  auto& mgr = o2::base::MaterialManager::Instance();
  auto med = mgr.getTGeoMedium(kModuleName, imed);
  if (med == nullptr) {
    throw std::runtime_error("Could not retrieve medium " + std::to_string(imed) + " for " + kModuleName);
  }
  return med;
}

} // namespace mid
} // namespace o2
