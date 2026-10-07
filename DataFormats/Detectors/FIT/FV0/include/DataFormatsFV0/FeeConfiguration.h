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

#ifndef O2_FV0_FEE_CONFIGURATION
#define O2_FV0_FEE_CONFIGURATION

#include "DataFormatsFIT/Configuration.h"

namespace o2::fv0
{
struct TriggersConfig {
  uint16_t nChannels{o2::fit::config_helpers::DefaultValue};
  uint16_t innerRings{o2::fit::config_helpers::DefaultValue};
  uint16_t charge{o2::fit::config_helpers::DefaultValue};
  uint16_t outerRings{o2::fit::config_helpers::DefaultValue};
  uint8_t sidesCombinationMode{o2::fit::config_helpers::DefaultValue};

  ClassDefNV(TriggersConfig, 1);
};

struct Fv0FeeConfiguration {
  static constexpr int NChannels = 48;
  TriggersConfig triggers;
  o2::fit::ChannelsConfig<NChannels> channels;
  o2::fit::TcmConfig tcm;
  o2::fit::PmConfig pmA[10];

  ClassDefNV(Fv0FeeConfiguration, 1);
};
} // namespace o2::fv0
#endif