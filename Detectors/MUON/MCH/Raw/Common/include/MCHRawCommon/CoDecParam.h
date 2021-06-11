// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  int sampaBcOffset = 339986; // default global sampa bunch-crossing offset

  O2ParamDef(CoDecParam, "MCHCoDecParam")
};

} // namespace o2::mch

#endif
