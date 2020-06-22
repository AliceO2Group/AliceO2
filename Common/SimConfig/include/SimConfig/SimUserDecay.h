// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMCONFIG_SIMUSERDECAY_H_
#define O2_SIMCONFIG_SIMUSERDECAY_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace conf
{
struct SimUserDecay : public o2::conf::ConfigurableParamHelper<SimUserDecay> {
  std::string pdglist = "";

  O2ParamDef(SimUserDecay, "SimUserDecay");
};
} // namespace conf
} // namespace o2

#endif /* O2_SIMCONFIG_SIMUSERDECAY_H_ */
