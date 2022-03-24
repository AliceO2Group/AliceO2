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

/// \author A+Morsch

#include <string>
#ifndef ALICEO2_SIMCONFIG_MATMAPPARAMS_H_
#define ALICEO2_SIMCONFIG_MATMAPPARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace conf
{ 
/**
 ** A parameter class/struct holding values for
 ** material budget maps
 **/
struct MatMapParams : public o2::conf::ConfigurableParamHelper<MatMapParams> {
  int nphi = 360;
  float phimin = 0;
  float phimax = 360.;
  int ntheta = 0;
  float thetamin = -45.;
  float thetamax = 45.;
  int neta = 0;
  float etamin = -2.;
  float etamax = 2.;  
  int nzv = 0;
  float zvmin = -50.;
  float zvmax = 50.;  
  float rmin = 0.;
  float rmax = 290.;
  float zmax = 2000;
  O2ParamDef(MatMapParams, "matm");
};
} // end namespace o2
} // end namespace conf
#endif // ALICEO2_SIMCONFIG_MATMAPPARAMS_H_
