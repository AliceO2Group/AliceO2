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

#ifndef ALICEO2_FT3_BASEPARAM_H_
#define ALICEO2_FT3_BASEPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace ft3
{

// **
// ** Parameters for FT3 base configuration
// **

enum FT3Geometry {
  Default = 0,
  Telescope = 1
};

struct FT3BaseParam : public o2::conf::ConfigurableParamHelper<FT3BaseParam> {
  // Geometry Builder parameters

  Int_t geoModel = FT3Geometry::Default;

  // FT3Geometry::Telescope parameters
  Int_t nLayers = 10;
  Float_t z0 = -16.0;      // First layer z position
  Float_t zLength = 263.0; // Distance between first and last layers
  Float_t etaIn = 4.5;
  Float_t etaOut = 1.5;
  Float_t Layerx2X0 = 0.01;

  // FT3Geometry::External file
  std::string configFile = ""; // Overrides geoModel parameter when provided

  O2ParamDef(FT3BaseParam, "FT3Base");
};

} // end namespace ft3
} // end namespace o2

#endif // ALICEO2_FT3_BASEPARAM_H_
