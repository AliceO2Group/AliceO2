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

/// \file ECalBaseParam.h
/// \brief Geometry parameters configurable via o2-sim --configKeyValues
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef O2_ECAL_BASEPARAM_H
#define O2_ECAL_BASEPARAM_H

#include <CommonUtils/ConfigurableParam.h>
#include <CommonUtils/ConfigurableParamHelper.h>

namespace o2
{
namespace ecal
{
struct ECalBaseParam : public o2::conf::ConfigurableParamHelper<ECalBaseParam> {
  bool enableFwdEndcap = false;
  // general ecal barrel settings
  double rMin = 125;    // cm
  double rMax = 155;    // cm
  double zLength = 350; // cm
  int nSuperModules = 4;
  // crystal module specification
  int nCrystalModulesZ = 31;
  int nCrystalModulesPhi = 96;
  double crystalAlphaDeg = 0.4;    // degrees
  double crystalModuleWidth = 1.9; // cm
  double crystalModuleLength = 18; // cm
  // sampling module specification
  int nSamplingModulesZ = 56;
  int nSamplingModulesPhi = 67;
  double samplingAlphaDeg = 0.4;    // degrees
  double samplingModuleWidth = 2.7; // cm
  double frontPlateThickness = 1.;  // cm
  double pbLayerThickness = 0.12;   // cm
  double scLayerThickness = 0.15;   // cm
  int nSamplingLayers = 80;
  // margin in z between crystal modules and sampling modules
  double marginCrystalToSampling = 0.1; // cm

  O2ParamDef(ECalBaseParam, "ECalBase");
};

} // namespace ecal
} // end namespace o2

#endif // O2_ECAL_BASEPARAM_H
