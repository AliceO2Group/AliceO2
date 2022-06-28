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

#ifndef O2_MCH_RAW_CODEC_PARAM_H
#define O2_MCH_RAW_CODEC_PARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

struct CoDecParam : public o2::conf::ConfigurableParamHelper<CoDecParam> {

  // default minimum allowed digit time, in orbit units
  int minDigitOrbitAccepted = -10;
  // default maximum allowed digit time, in orbit units
  // a negative value forces the value to be equal to the time-frame length
  int maxDigitOrbitAccepted = -1;

  O2ParamDef(CoDecParam, "MCHCoDecParam")
};

} // namespace o2::mch

#endif
