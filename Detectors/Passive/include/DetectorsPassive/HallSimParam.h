// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_PASSIVE_INCLUDE_DETECTORSPASSIVE_HALLSIMPARAM_H_
#define DETECTORS_PASSIVE_INCLUDE_DETECTORSPASSIVE_HALLSIMPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace passive
{

struct HallSimParam : public o2::conf::ConfigurableParamHelper<HallSimParam> {
  float mCUTGAM = 1.e00;
  float mCUTELE = 1.e00;
  float mCUTNEU = 1.e-1;
  float mCUTHAD = 1.e-3;

  // boilerplate stuff + make principal key "HallSim"
  O2ParamDef(HallSimParam, "HallSim");
};

} // namespace passive
} // namespace o2

#endif /* DETECTORS_PASSIVE_INCLUDE_DETECTORSPASSIVE_HALLSIMPARAM_H_ */
