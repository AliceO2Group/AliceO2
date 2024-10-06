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

#ifndef ALICEO2_FVD_FVDBASEPARAM_
#define ALICEO2_FVD_FVDBASEPARAM_

#include "FVDBase/GeometryTGeo.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace fvd
{
struct FVDBaseParam : public o2::conf::ConfigurableParamHelper<FVDBaseParam> {

  int nsect = 8;

  std::vector<float> ringsA = {3., 14.8, 26.6, 38.4, 50.2, 62.};
  std::vector<float> ringsC = {3.5, 14.75, 26.0, 37.25, 48.5, 59.75, 71.};

  float zmodA = 1700;
  float zmodC = -1950;
  float dzscint = 4.;

  O2ParamDef(FVDBaseParam, "FVDBase");
};

} // namespace fvd
} // namespace o2

#endif
