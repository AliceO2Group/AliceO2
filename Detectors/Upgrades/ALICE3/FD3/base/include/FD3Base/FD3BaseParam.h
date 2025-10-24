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

#ifndef ALICEO2_FD3_FD3BASEPARAM_
#define ALICEO2_FD3_FD3BASEPARAM_

#include "FD3Base/GeometryTGeo.h"
#include "FD3Base/Constants.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace fd3
{
struct FD3BaseParam : public o2::conf::ConfigurableParamHelper<FD3BaseParam> {

  float zmodA = 1700.0f;
  float zmodC = -1850.0f;
  float dzscint = 4.0f;

  bool withMG = false; // modified geometry with 3 rings on A side

  bool plateBehindA = false;
  bool fullContainer = false;
  float dzplate = 1.0f; // Aluminium plate width

  O2ParamDef(FD3BaseParam, "FD3Base");
};

} // namespace fd3
} // namespace o2

#endif
