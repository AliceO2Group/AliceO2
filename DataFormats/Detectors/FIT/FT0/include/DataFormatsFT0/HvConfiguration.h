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

#ifndef O2_FT0_HV_CONFIGURATION
#define O2_FT0_HV_CONFIGURATION

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFIT/Configuration.h"
#include <Rtypes.h>

namespace o2::ft0
{
struct Ft0HvConfiguration {
  static constexpr int NChannels = 208;
  o2::fit::HvChannelsConfig<NChannels> channels;
  ClassDefNV(Ft0HvConfiguration, 1);
};
} // namespace o2::ft0
#endif