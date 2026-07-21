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

/// \author ruben.shahoyan@cern.ch

/// parameters to bias the origin of the magnetic field

#ifndef ALICEO2_FIELDORIGIN_BIAS_PARAM_H
#define ALICEO2_FIELDORIGIN_BIAS_PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace field
{

struct FieldOriginBiasParam : public o2::conf::ConfigurableParamHelper<FieldOriginBiasParam> {
  double x = 0.;
  double y = 0.;
  double z = 0.;

  O2ParamDef(FieldOriginBiasParam, "FieldOriginBias");
};

} // namespace field
} // end namespace o2

#endif
