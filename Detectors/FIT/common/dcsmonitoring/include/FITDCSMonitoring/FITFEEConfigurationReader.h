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

#ifndef O2_FIT_DCS_CONFIGURATION_DATA_READER_H
#define O2_FIT_DCS_CONFIGURATION_DATA_READER_H

#include <rapidjson/document.h>
#include <rapidjson/schema.h>
#include "DataFormatsFIT/Configuration.h"
#include "DetectorsCalibration/Utils.h"
#include "FITDCSMonitoring/FITDCSBaseConfigReader.h"

namespace o2::fit
{
template <typename ConfigurationReaderType>
class FITFEEConfigurationReader : public FITDCSBaseConfigReader
{
 public:
  FITFEEConfigurationReader()
  {
    loadSchema(configurationSchema);
  }

  virtual ~FITFEEConfigurationReader() = default;

 protected:
  template <typename ConfigType>
  ConfigType parseFeeConfiguration(gsl::span<const char> buffer)
  {
    rapidjson::Document document = parseJsonBuffer(buffer);
    ConfigType configuration;

    parseChannelData(document, "channels", configuration.channels);
    parseTcmConfig(document, "tcm", configuration.tcm);
    if constexpr (requires { configuration.pmA; }) {
      parsePmsArray(document, "pm_a", configuration.pmA);
    }
    if constexpr (requires { configuration.pmC; }) {
      parsePmsArray(document, "pm_c", configuration.pmC);
    }
    static_cast<ConfigurationReaderType*>(this)->parseTriggers(document, "triggers", configuration.triggers);

    return configuration;
  }

  template <typename T, int Size>
  void parseJsonArray(const rapidjson::Value& node, const char* childName, T (&array)[Size])
  {
    if (node.HasMember(childName) == false) {
      throw std::runtime_error(std::string("Failed to find node of name ") + childName);
    }
    const auto& childNode = node[childName];
    if (childNode.IsArray() == false) {
      throw std::runtime_error(std::format("Node {} is not an array!", childName));
    }
    auto jsonArray = childNode.GetArray();
    if (jsonArray.Size() != Size) {
      throw std::runtime_error(std::format("Array {}. Expected array of size {}, parsed array of size {}", childName, Size, jsonArray.Size()));
    }
    for (int idx = 0; idx < Size; idx++) {
      const auto& node = jsonArray[idx];
      if constexpr (std::is_same_v<T, bool>) {
        if (!node.IsBool()) {
          throw std::runtime_error(std::format("{} is not a bool array", childName));
        }
        array[idx] = node.GetBool();
      } else if constexpr (std::is_floating_point_v<T>) {
        if (!node.IsNumber()) {
          throw std::runtime_error(std::format("{} is not an floating point array", childName));
        }
        array[idx] = node.GetFloat();
      } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        if (!node.IsUint()) {
          throw std::runtime_error(std::format("{} is not an unsigned integer array", childName));
        }
        array[idx] = static_cast<T>(node.GetUint());
      } else if constexpr (std::is_integral_v<T>) {
        if (!node.IsInt()) {
          throw std::runtime_error(std::format("{} is not an integer array", childName));
        }
        array[idx] = static_cast<T>(node.GetInt());
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported type");
      }
    }
  }

  template <int Size>
  void parseChannelData(const rapidjson::Value& root, const char* channelsNodeName, ChannelsConfig<Size>& channelsConfiguration)
  {
    const auto& channelsNode = root[channelsNodeName];
    parseJsonArray(channelsNode, "time_alignments", channelsConfiguration.timeAligments);
    parseJsonArray(channelsNode, "cfd_thresholds", channelsConfiguration.cfdThresholds);
    parseJsonArray(channelsNode, "cfd_zeros", channelsConfiguration.cfdZeros);
    parseJsonArray(channelsNode, "adc_zeros", channelsConfiguration.adcZeros);
    parseJsonArray(channelsNode, "adc_delays", channelsConfiguration.adcDelays);
    parseJsonArray(channelsNode, "range_correction_adc0", channelsConfiguration.rangeCorrectionAdc0);
    parseJsonArray(channelsNode, "range_correction_adc1", channelsConfiguration.rangeCorrectionAdc1);
    parseJsonArray(channelsNode, "channel_mask_data", channelsConfiguration.channelMaskData);
    parseJsonArray(channelsNode, "channel_mask_triggers", channelsConfiguration.channelMaskTriggers);
    parseJsonArray(channelsNode, "threshold_calibration", channelsConfiguration.thresholdCalibration);
  }

  void parseTcmConfig(const rapidjson::Value& root, const char* tcmNodeName, TcmConfig& tcmConfig)
  {
    const auto& tcmNode = root[tcmNodeName];
    const auto& phaseDelayANode = tcmNode["phase_delay_a"];
    const auto& phaseDelayCNode = tcmNode["phase_delay_c"];
    tcmConfig.phaseDelayA = phaseDelayANode.GetDouble();
    tcmConfig.phaseDelayC = phaseDelayCNode.GetDouble();
  }

  void parsePmConfig(const rapidjson::Value& pmNode, PmConfig& pmConfig)
  {
    const auto& orGateNode = pmNode["or_gate"];
    const auto& trgChargeHighLevelNode = pmNode["trg_charge_high_level"];
    const auto& trgChargeLowLevelNode = pmNode["trg_charge_low_level"];
    pmConfig.orGate = orGateNode.GetUint();
    pmConfig.trgChargeHighLevel = trgChargeHighLevelNode.GetUint();
    pmConfig.trgChargeLowLevel = trgChargeLowLevelNode.GetUint();
  }

  template <int Size>
  void parsePmsArray(const rapidjson::Value& root, const char* pmArrayNodeName, PmConfig (&pmConfig)[Size])
  {
    if (root.HasMember(pmArrayNodeName) == false) {
      throw std::runtime_error(std::format("Failed to find {} node", pmArrayNodeName));
    }
    const auto& pmArrayNode = root[pmArrayNodeName];
    const auto& pmArray = pmArrayNode.GetArray();
    if (pmArray.Size() != Size) {
      throw std::runtime_error(std::format("Received data for {} PMs, but expected {}", pmArray.Size(), Size));
    }
    for (int idx = 0; idx < pmArray.Size(); idx++) {
      const auto& pm = pmArray[idx];
      if (pm.IsNull()) {
        continue;
      }
      parsePmConfig(pm, pmConfig[idx]);
    }
  }

  std::string_view getSchemaString() const
  {
    return configurationSchema;
  }

 private:
  inline static constexpr std::string_view configurationSchema = R"SCH(
        {
        "type": "object",
        "properties": {
            "channels": {
                "type": "object",
                "properties": {
                    "time_alignments": {"type": "array", "items": {"type": "integer"}},
                    "cfd_thresholds": {"type": "array", "items": {"type": "integer"}},
                    "cfd_zeros": {"type": "array", "items": {"type": "integer"}},
                    "adc_zeros": {"type": "array", "items": {"type": "integer"}},
                    "adc_delays": {"type": "array", "items": {"type": "integer"}},
                    "range_correction_adc0": {"type": "array", "items": {"type": "integer"}},
                    "range_correction_adc1": {"type": "array", "items": {"type": "integer"}},
                    "channel_mask_data": {"type": "array", "items": {"type": "boolean"}},
                    "channel_mask_triggers": {"type": "array", "items": {"type": "boolean"}},
                    "threshold_calibration": {"type": "array", "items": {"type": "integer"}}
                },
                "additionalProperties": false,
                "required": [
                  "time_alignments", "cfd_thresholds", "cfd_zeros",
                  "adc_zeros", "adc_delays", "range_correction_adc0", "range_correction_adc1", 
                  "channel_mask_data", "channel_mask_triggers"
                ]
            },
            "tcm" : {
                "type": "object",
                "properties": {
                    "phase_delay_a": {"type": "number"},
                    "phase_delay_c": {"type": "number"}
                },
                "additionalProperties": false,
                "required": ["phase_delay_a", "phase_delay_c"]
            },
            "pm_a":{
                "type": "array",
                "items": {
                  "type": ["object", "null"],
                  "properties": {
                    "or_gate": {"type": "number"},
                    "trg_charge_low_level": {"type": "number"},
                    "trg_charge_high_level": {"type": "number"},
                    "fdd_coincidence_mode": {"type": "boolean"}
                  },
                  "additionalProperties": false,
                  "required": ["or_gate", "trg_charge_low_level", "trg_charge_high_level"]
                }
            },
            "pm_c": {
                "type": "array",
                "items": {
                  "type": ["object", "null"],
                  "properties": {
                    "or_gate": {"type": "number"},
                    "trg_charge_low_level": {"type": "number"},
                    "trg_charge_high_level": {"type": "number"},
                    "fddCoincidenceMode": {"type": "boolean}
                  },
                  "additionalProperties": false,
                  "required": ["or_gate", "trg_charge_low_level", "trg_charge_high_level"]
                }
            },
            "triggers": {
                "type": "object",
                "additionalProperties" : { "type": "number" }
            }
        },
        "required": ["channels", "tcm", "triggers"]
    }
    )SCH";
};
} // namespace o2::fit

#endif