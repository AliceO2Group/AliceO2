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

#ifndef ALICEO2_SIMDATAFORMAT_STACKPARAM_H_
#define ALICEO2_SIMDATAFORMAT_STACKPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace sim
{

// configuration parameters for simulation Stack
struct StackParam : public o2::conf::ConfigurableParamHelper<StackParam> {
  bool storeSecondaries = true;
  bool pruneKine = true;
  std::string transportPrimary = "all";
  std::string transportPrimaryFileName = "";
  std::string transportPrimaryFuncName = "";
  bool transportPrimaryInvert = false;

  // boilerplate stuff + make principal key "Stack"
  O2ParamDef(StackParam, "Stack");
};

} // namespace sim
} // namespace o2

#endif // ALICEO2_SIMDATAFORMAT_STACKPARAM_H_
