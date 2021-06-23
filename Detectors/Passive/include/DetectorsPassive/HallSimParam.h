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

#ifndef DETECTORS_PASSIVE_INCLUDE_DETECTORSPASSIVE_HALLSIMPARAM_H_
#define DETECTORS_PASSIVE_INCLUDE_DETECTORSPASSIVE_HALLSIMPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace passive
{

struct HallSimParam : public o2::conf::ConfigurableParamHelper<HallSimParam> {
  float mCUTGAM = 1.e00;
  float mCUTELE = 1.e00;
  float mCUTNEU = 1.e-1;
  float mCUTHAD = 1.e-3;

  bool fastYoke = true;  // if we treat compensator yoke in fast manner
  float yokeDelta = 2.;  // thickness of
                         // full physics outer layer of compensator yoke (in cm)

  // boilerplate stuff + make principal key "HallSim"
  O2ParamDef(HallSimParam, "HallSim");
};

} // namespace passive
} // namespace o2

#endif /* DETECTORS_PASSIVE_INCLUDE_DETECTORSPASSIVE_HALLSIMPARAM_H_ */
