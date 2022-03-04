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

#ifndef DETECTORS_BASE_INCLUDE_MATERIALMANAGERPARAM_H_
#define DETECTORS_BASE_INCLUDE_MATERIALMANAGERPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{

struct MaterialManagerParam : public o2::conf::ConfigurableParamHelper<MaterialManagerParam> {
  std::string inputFile = "";
  std::string outputFile = "";
  // boilerplate stuff + make principal key "MaterialManagerParam"
  O2ParamDef(MaterialManagerParam, "MaterialManagerParam");
};

} // namespace o2

#endif /* DETECTORS_BASE_INCLUDE_MATERIALMANAGERPARAM_H_ */
