// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ITS3PARAMS_H_
#define ITS3PARAMS_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::its3
{

struct ITS3Params : public o2::conf::ConfigurableParamHelper<ITS3Params> {
  // Chip studies
  bool useDeadChannelMap{false};            // Query for a dead channel map to study disabling individual tiles
  std::string chipResponseFunction{"APTS"}; // Chip response function one of "Alpide", "APTS" or "Mosaix" (not yet available)

  O2ParamDef(ITS3Params, "ITS3Params");
};

} // namespace o2::its3

#endif // ITS3PARAMS_H_
