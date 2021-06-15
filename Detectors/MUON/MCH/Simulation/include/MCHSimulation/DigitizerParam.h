// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_DIGITIZER_PARAM_H_
#define O2_MCH_SIMULATION_DIGITIZER_PARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::mch
{

struct DigitizerParam : public o2::conf::ConfigurableParamHelper<DigitizerParam> {

  bool continuous = true;           // whether we assume continuous mode or not
  float noiseProba = 3.1671242e-05; // by default = proba to be above 4*sigma of a gaussian noise

  O2ParamDef(DigitizerParam, "MCHDigitizerParam")
};

} // namespace o2::mch

#endif
