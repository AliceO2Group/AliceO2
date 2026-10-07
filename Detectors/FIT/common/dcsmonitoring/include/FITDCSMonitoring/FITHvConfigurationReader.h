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

#ifndef O2_FIT_DCS_HV_CONFIGURATION_DATA_READER_H
#define O2_FIT_DCS_HV_CONFIGURATION_DATA_READER_H

#include <rapidjson/document.h>
#include <rapidjson/schema.h>
#include "DataFormatsFIT/Configuration.h"
#include "DetectorsCalibration/Utils.h"
#include "FITDCSMonitoring/FITDCSBaseConfigReader.h"

namespace o2::fit
{
class FITHvConfigurationReader : public FITDCSBaseConfigReader
{
 public:
  FITHvConfigurationReader()
  {
    loadSchema(configurationSchema);
  }

  template <typename ConfigType>
  ConfigType parseHvConfiguration(gsl::span<const char> buffer)
  {
    rapidjson::Document root = parseJsonBuffer(buffer);
    ConfigType configuration;
    parseHvChannelData(root, "hv_channels", configuration.channels);
    return configuration;
  }

  template <size_t NChannels>
  void parseHvChannelData(rapidjson::Value& root, const char* hvChannelsNode, HvChannelsConfig<NChannels>& config)
  {
    const auto& hvChannels = root[hvChannelsNode];
    parseJsonArray(hvChannels, "gain", config.gain);
  }

  std::string_view getSchemaString() const
  {
    return configurationSchema;
  }

 private:
  inline static constexpr std::string_view configurationSchema = R"json(
  {
    "type": "object",
    "properties": {
      "hv_channels": {
        "type": "object",
        "properties": {
          "gain": {
            "type": "array",
            "items": {
              "type": "number"
            }
          }
        },
        "additionalProperties": false,
        "required": ["gain"]
      }
    },
    "additionalProperties": false,
    "required": ["hv_channels"]
  }
  )json";
};
} // namespace o2::fit
#endif