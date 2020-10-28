// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CONF_TOFDIGIPARAMS_H_
#define O2_CONF_TOFDIGIPARAMS_H_

// Global parameters for TOF simulation / digitization

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace tof
{

// Global parameters for TOF simulation / digitization
struct TOFSimParams : public o2::conf::ConfigurableParamHelper<TOFSimParams> {

  int time_resolution = 60; // TOF resolution in ps

  // efficiency parameters
  float eff_center = 0.995;    // efficiency in the center of the fired pad
  float eff_boundary1 = 0.94;  // efficiency in mBound2
  float eff_boundary2 = 0.833; // efficiency in the pad border
  float eff_boundary3 = 0.1;   // efficiency in mBound3

  O2ParamDef(TOFSimParams, "TOFSimParams");
};

} // namespace tof
} // namespace o2

#endif
