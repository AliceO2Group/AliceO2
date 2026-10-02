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

#ifndef O2_FT0_FEE_CONFIGURATION
#define O2_FT0_FEE_CONFIGURATION

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFIT/Configuration.h"
#include <Rtypes.h>

namespace o2::ft0
{
struct TriggersConfig {
  int16_t vertexTimeLowThreshold{o2::fit::config_helpers::DefaultValue};
  int16_t vertexTimeHighThreshold{o2::fit::config_helpers::DefaultValue};
  uint16_t semicentralA{o2::fit::config_helpers::DefaultValue};
  uint16_t semicentralC{o2::fit::config_helpers::DefaultValue};
  uint16_t centralA{o2::fit::config_helpers::DefaultValue};
  uint16_t centralC{o2::fit::config_helpers::DefaultValue};
  uint8_t sidesCombinationMode{o2::fit::config_helpers::DefaultValue};
  ClassDefNV(TriggersConfig, 1);
};

struct Ft0FeeConfiguration {
  static constexpr int NChannels = 208;
  TriggersConfig triggers;
  o2::fit::ChannelsConfig<NChannels> channels;
  o2::fit::TcmConfig tcm;
  o2::fit::PmConfig pmA[10];
  o2::fit::PmConfig pmC[10];
  ClassDefNV(Ft0FeeConfiguration, 1);
};
} // namespace o2::ft0
#endif