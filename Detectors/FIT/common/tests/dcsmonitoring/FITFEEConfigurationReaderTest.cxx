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

#define BOOST_TEST_MODULE Test FITDCSMonitoring FITFEEConfigurationReader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <rapidjson/document.h>

#include "DataFormatsFIT/Configuration.h"
#include "FITDCSMonitoring/FITFEEConfigurationReader.h"

using namespace o2::fit;

BOOST_AUTO_TEST_SUITE(o2_fit_dcs_monitoring)

// Dummy structure for trigger data for complete definition of FEE configuration structure
struct SimpleTriggers {
  int32_t triggerA;
};

// Test configuration structure
struct SimpleConfiguration {
  static constexpr int NChannels = 5;
  SimpleTriggers triggers;
  o2::fit::ChannelsConfig<NChannels> channels;
  o2::fit::TcmConfig tcm;
  o2::fit::PmConfig pmA[2];
  o2::fit::PmConfig pmC[2];
};

rapidjson::Document createEmptyPayload()
{
  rapidjson::Document doc;
  doc.SetObject();
  return doc;
}

rapidjson::Document createDocumentFromString(const char* data)
{
  rapidjson::Document doc;
  doc.Parse(data);
  return doc;
}

rapidjson::Document createPmConfigNode(uint8_t orGateValue, uint16_t trgChargeLowLevel, uint16_t trgChargeHighLevel)
{
  auto doc = createEmptyPayload();
  auto& allocator = doc.GetAllocator();
  doc.AddMember("or_gate", orGateValue, allocator);
  doc.AddMember("trg_charge_low_level", trgChargeLowLevel, allocator);
  doc.AddMember("trg_charge_high_level", trgChargeHighLevel, allocator);
  return doc;
}

void addTcmConfigNode(rapidjson::Document& doc,
                      const char* nodeName,
                      float phaseDelayA,
                      float phaseDelayC)
{
  auto& allocator = doc.GetAllocator();

  rapidjson::Value tcmConfigJson(rapidjson::kObjectType);
  tcmConfigJson.AddMember("phase_delay_a", phaseDelayA, allocator);
  tcmConfigJson.AddMember("phase_delay_c", phaseDelayC, allocator);

  doc.AddMember(rapidjson::StringRef(nodeName), tcmConfigJson, allocator);
}

template <size_t NChannels>
void addPmConfigsNode(rapidjson::Document& doc, const char* nodeName, const uint8_t (&orGateValues)[NChannels], const uint16_t (&trgChargeLowLevel)[NChannels], const uint16_t (&trgChargeHighLevel)[NChannels])
{
  auto& allocator = doc.GetAllocator();

  rapidjson::Value pmJson(rapidjson::kArrayType);

  for (size_t idx = 0; idx < NChannels; ++idx) {
    rapidjson::Value pm(rapidjson::kObjectType);
    pm.AddMember("or_gate", orGateValues[idx], allocator);
    pm.AddMember("trg_charge_low_level", trgChargeLowLevel[idx], allocator);
    pm.AddMember("trg_charge_high_level", trgChargeHighLevel[idx], allocator);
    pmJson.PushBack(pm, allocator);
  }

  doc.AddMember(rapidjson::StringRef(nodeName), pmJson, allocator);
}

template <typename T, size_t N>
void addArrayMember(rapidjson::Value& node, const char* memberName, const T (&values)[N], rapidjson::Document::AllocatorType& allocator)
{
  rapidjson::Value arrayJson(rapidjson::kArrayType);

  for (size_t idx = 0; idx < N; ++idx) {
    arrayJson.PushBack(values[idx], allocator);
  }

  node.AddMember(rapidjson::StringRef(memberName), arrayJson, allocator);
}

template <size_t NChannels>
void addChannelsConfigNode(rapidjson::Document& doc,
                           const char* nodeName,
                           const int16_t (&timeAligments)[NChannels],
                           const uint16_t (&cfdThresholds)[NChannels],
                           const int16_t (&cfdZeros)[NChannels],
                           const int16_t (&adcZeros)[NChannels],
                           const uint16_t (&adcDelays)[NChannels],
                           const uint16_t (&rangeCorrectionAdc0)[NChannels],
                           const uint16_t (&rangeCorrectionAdc1)[NChannels],
                           const bool (&channelMaskData)[NChannels],
                           const bool (&channelMaskTriggers)[NChannels])
{
  auto& allocator = doc.GetAllocator();

  rapidjson::Value channelsJson(rapidjson::kObjectType);

  addArrayMember(channelsJson, "time_alignments", timeAligments, allocator);
  addArrayMember(channelsJson, "cfd_thresholds", cfdThresholds, allocator);
  addArrayMember(channelsJson, "cfd_zeros", cfdZeros, allocator);
  addArrayMember(channelsJson, "adc_zeros", adcZeros, allocator);
  addArrayMember(channelsJson, "adc_delays", adcDelays, allocator);
  addArrayMember(channelsJson, "range_correction_adc0", rangeCorrectionAdc0, allocator);
  addArrayMember(channelsJson, "range_correction_adc1", rangeCorrectionAdc1, allocator);
  addArrayMember(channelsJson, "channel_mask_data", channelMaskData, allocator);
  addArrayMember(channelsJson, "channel_mask_triggers", channelMaskTriggers, allocator);

  doc.AddMember(rapidjson::StringRef(nodeName), channelsJson, allocator);
}

class SimpleFITFEEConfigurationReader
  : public FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>
{
 public:
  // Dummy method to parse dummy triggers payload
  SimpleTriggers parseTriggers(const rapidjson::Value&,
                               const char*,
                               SimpleTriggers& triggers)
  {
    triggers.triggerA = 32;
    return triggers;
  }

  using FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>::parseFeeConfiguration;
  using FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>::parseJsonArray;
  using FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>::parseChannelData;
  using FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>::parseTcmConfig;
  using FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>::parsePmConfig;
  using FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>::parsePmsArray;
  using FITFEEConfigurationReader<SimpleFITFEEConfigurationReader>::validateSchema;
};

BOOST_AUTO_TEST_CASE(shouldParseSinglePmConfig)
{
  const uint8_t orGateValue = 20;
  const uint16_t rangeCorrectionAdc0 = 1024;
  const uint16_t rangeCorrectionAdc1 = 2048;

  rapidjson::Document doc = createPmConfigNode(orGateValue, rangeCorrectionAdc0, rangeCorrectionAdc1);

  PmConfig pmConfig;
  SimpleFITFEEConfigurationReader reader;
  reader.parsePmConfig(doc, pmConfig);

  BOOST_CHECK_EQUAL(pmConfig.orGate, orGateValue);
}

BOOST_AUTO_TEST_CASE(shouldParseTcmConfig)
{
  const float phaseDelayAValue = 1.2f;
  const float phaseDelayCValue = -1.3f;

  rapidjson::Document doc = createEmptyPayload();
  addTcmConfigNode(doc, "tcm_config", phaseDelayAValue, phaseDelayCValue);

  TcmConfig tcmConfig;

  SimpleFITFEEConfigurationReader reader;
  reader.parseTcmConfig(doc, "tcm_config", tcmConfig);

  TcmConfig expectedConfig = {
    .phaseDelayA = phaseDelayAValue,
    .phaseDelayC = phaseDelayCValue};

  BOOST_CHECK(tcmConfig == expectedConfig);
}

BOOST_AUTO_TEST_CASE(shouldParseArrayOfPmConfigs)
{
  const uint8_t pmAOrGate[] = {3, 4};
  const uint16_t pmATrgChargeLowLevel[] = {2047, 2048};
  const uint16_t pmATrgChargeHighLevel[] = {1047, 1048};
  const uint8_t pmCOrGate[] = {5, 6};
  const uint16_t pmCTrgChargeLowLevel[] = {2045, 2046};
  const uint16_t pmCTrgChargeHighLevel[] = {1045, 1046};

  rapidjson::Document doc = createEmptyPayload();
  addPmConfigsNode(doc, "pm_a", pmAOrGate, pmATrgChargeLowLevel, pmATrgChargeHighLevel);
  addPmConfigsNode(doc, "pm_c", pmCOrGate, pmCTrgChargeLowLevel, pmCTrgChargeHighLevel);

  PmConfig pmAParsed[2];
  PmConfig pmCParsed[2];

  SimpleFITFEEConfigurationReader reader;
  reader.parsePmsArray(doc, "pm_a", pmAParsed);
  reader.parsePmsArray(doc, "pm_c", pmCParsed);

  for (size_t idx = 0; idx < std::size(pmAParsed); ++idx) {
    BOOST_CHECK_EQUAL(pmAParsed[idx].orGate, pmAOrGate[idx]);
  }

  for (size_t idx = 0; idx < std::size(pmCParsed); ++idx) {
    BOOST_CHECK_EQUAL(pmCParsed[idx].orGate, pmCOrGate[idx]);
  }
}

BOOST_AUTO_TEST_CASE(shouldThrowOnInconsistentPmsArraySize)
{
  const uint8_t pmAOrGate[] = {0, 0};
  const uint16_t pmATrgChargeLowLevel[] = {2047, 2048};
  const uint16_t pmATrgChargeHighLevel[] = {1047, 1048};

  rapidjson::Document doc = createEmptyPayload();
  addPmConfigsNode(doc, "pm_a", pmAOrGate, pmATrgChargeLowLevel, pmATrgChargeHighLevel);

  PmConfig pmAParsed[3];

  SimpleFITFEEConfigurationReader reader;

  BOOST_CHECK_THROW(reader.parsePmsArray(doc, "pm_a", pmAParsed), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(shouldParseChannelsConfig)
{
  constexpr size_t NTestChannels = 4;

  const int16_t timeAligments[NTestChannels] = {1, 2, 3, 4};
  const uint16_t cfdThresholds[NTestChannels] = {10, 20, 30, 40};
  const int16_t cfdZeros[NTestChannels] = {-1, -2, -3, -4};
  const int16_t adcZeros[NTestChannels] = {100, 200, 300, 400};
  const uint16_t adcDelays[NTestChannels] = {5, 6, 7, 8};
  const uint16_t rangeCorrectionAdc0[NTestChannels] = {2045, 2046, 2047, 2048};
  const uint16_t rangeCorrectionAdc1[NTestChannels] = {2048, 2022, 2021, 2022};
  const bool channelMaskData[NTestChannels] = {true, false, true, false};
  const bool channelMaskTriggers[NTestChannels] = {false, true, false, true};

  rapidjson::Document doc = createEmptyPayload();

  addChannelsConfigNode(
    doc,
    "channels",
    timeAligments,
    cfdThresholds,
    cfdZeros,
    adcZeros,
    adcDelays,
    rangeCorrectionAdc0,
    rangeCorrectionAdc1,
    channelMaskData,
    channelMaskTriggers);

  ChannelsConfig<NTestChannels> parsed{};

  SimpleFITFEEConfigurationReader reader;
  reader.parseChannelData(doc, "channels", parsed);

  for (size_t idx = 0; idx < NTestChannels; ++idx) {
    BOOST_CHECK_EQUAL(parsed.timeAligments[idx], timeAligments[idx]);
    BOOST_CHECK_EQUAL(parsed.cfdThresholds[idx], cfdThresholds[idx]);
    BOOST_CHECK_EQUAL(parsed.cfdZeros[idx], cfdZeros[idx]);
    BOOST_CHECK_EQUAL(parsed.adcZeros[idx], adcZeros[idx]);
    BOOST_CHECK_EQUAL(parsed.adcDelays[idx], adcDelays[idx]);
    BOOST_CHECK_EQUAL(parsed.channelMaskData[idx], channelMaskData[idx]);
    BOOST_CHECK_EQUAL(parsed.channelMaskTriggers[idx], channelMaskTriggers[idx]);
  }
}

BOOST_AUTO_TEST_CASE(shouldAcceptValidFeeConfigurationPayload)
{
  const char* jsonPayload = R"json(
  {
    "channels": {
      "time_alignments": [1, 2, 3, 4],
      "cfd_thresholds": [10, 20, 30, 40],
      "cfd_zeros": [11, 22, 33, 44],
      "adc_zeros": [21, 22, 23, 24],
      "adc_delays": [31, 32, 33, 34],
      "range_correction_adc0": [2048, 2048, 1025, 1024],
      "range_correction_adc1": [2048, 2047, 2046, 2045],
      "channel_mask_data": [true, false, true, false],
      "channel_mask_triggers": [true, true, false, false]
    },
    "pm_a": [
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      {
        "or_gate": 1.22,
        "trg_charge_low_level": 4,
        "trg_charge_high_level": 7
      }
    ],
    "pm_c": [
       {
        "or_gate": 1.22,
        "trg_charge_low_level": 4,
        "trg_charge_high_level": 7
      },
       {
        "or_gate": 1.22,
        "trg_charge_low_level": 4,
        "trg_charge_high_level": 7
      }
    ],
    "tcm": {
      "phase_delay_a": 3.3,
      "phase_delay_c": 4.4
    },
    "triggers": {
      "trigger_a": 123
    }
  }
  )json";

  rapidjson::Document doc = createDocumentFromString(jsonPayload);

  SimpleFITFEEConfigurationReader reader;
  std::string errorMessage;

  BOOST_CHECK_MESSAGE(reader.validateSchema(doc, errorMessage), errorMessage);
}

BOOST_AUTO_TEST_CASE(shouldAcceptValidFeeConfigurationPayloadWithNullPmData)
{
  const char* jsonPayload = R"json(
  {
    "channels": {
      "time_alignments": [1, 2, 3, 4],
      "cfd_thresholds": [10, 20, 30, 40],
      "cfd_zeros": [11, 22, 33, 44],
      "adc_zeros": [21, 22, 23, 24],
      "adc_delays": [31, 32, 33, 34],
      "range_correction_adc0": [2048, 2048, 1025, 1024],
      "range_correction_adc1": [2048, 2047, 2046, 2045],
      "channel_mask_data": [true, false, true, false],
      "channel_mask_triggers": [true, true, false, false]
    },
    "pm_a": [
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      null,
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      }
    ],
    "pm_c": [
      null,
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      null,
      null
    ],
    "tcm": {
      "phase_delay_a": 3.3,
      "phase_delay_c": 4.4
    },
    "triggers": {
      "trigger_a": 123
    }
  }
  )json";

  rapidjson::Document doc = createDocumentFromString(jsonPayload);

  SimpleFITFEEConfigurationReader reader;
  std::string errorMessage;

  BOOST_CHECK_MESSAGE(reader.validateSchema(doc, errorMessage), errorMessage);
}

BOOST_AUTO_TEST_CASE(shouldDetectInvalidPayload)
{
  SimpleFITFEEConfigurationReader reader;
  std::string errorMessage;

  const char* missingChannelData = R"json(
  {
    "channels": {
      "time_alignments": [1, 2, 3, 4],
      "cfd_thresholds": [10, 20, 40, 44],
      "cfd_zeros": [11, 22, 33, 44],
      "adc_delays": [31, 32, 33, 34],
      "range_correction_adc0": [2048, 2048, 1025, 1024],
      "range_correction_adc1": [2048, 2047, 2046, 2045],
      "channel_mask_data": [true, false, true, false],
      "channel_mask_triggers": [true, true, false, false]
    },
    "pm_a": [
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      null,
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      }
    ],
    "pm_c": [
      null,
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      null,
      null
    ],
    "tcm": {
      "phase_delay_a": 3.3,
      "phase_delay_c": 4.4
    },
    "triggers": {
      "trigger_a": 123
    }
  }
  )json";

  rapidjson::Document doc = createDocumentFromString(missingChannelData);
  BOOST_CHECK(reader.validateSchema(doc, errorMessage) == false);

  const char* invalidPmConfig = R"json(
  {
    "channels": {
      "time_alignments": [1, 2, 3, 5, 4],
      "cfd_thresholds": [10, 20, 30, 40],
      "cfd_zeros": [11, 22, 33, 44],
      "adc_zeros": [21, 22, 23, 24],
      "range_correction_adc0": [2048, 2048, 1025, 1024],
      "range_correction_adc1": [2048, 2047, 2046, 2045],
      "channel_mask_data": [true, false, true, false],
      "channel_mask_triggers": [true, true, false, false]
    },
    "pm_a": [
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      null,
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      }
    ],
    "pm_c": [
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      }
    ],
    "tcm": {
      "phase_delay_a": 3.3,
      "phase_delay_c": 4.4
    },
    "triggers": {
      "trigger_a": 123
    }
  }
  )json";

  doc = createDocumentFromString(invalidPmConfig);
  BOOST_CHECK(reader.validateSchema(doc, errorMessage) == false);

  const char* invalidTcmConfig = R"json(
  {
    "channels": {
      "time_alignments": [1, 2, 3, 4],
      "cfd_thresholds": [10, 20, 30, 40],
      "cfd_zeros": [11, 22, 33, 44],
      "adc_zeros": [21, 22, 23, 24],
      "adc_delays": [31, 32, 33, 34],
      "range_correction_adc0": [2048, 2048, 1025, 1024],
      "range_correction_adc1": [2048, 2047, 2046, 2045],
      "channel_mask_data": [true, false, true, false],
      "channel_mask_triggers": [true, true, false, false]
    },
    "pm_a": [
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      null,
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      }
    ],
    "pm_c": [
      null,
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      {
        "or_gate": 1.21,
        "trg_charge_low_level": 2,
        "trg_charge_high_level": 5
      },
      null,
      null
    ],
    "tcm": {
      "phase_delay_c": 4.4
    },
    "triggers": {
      "trigger_a": 123
    }
  }
  )json";

  doc = createDocumentFromString(invalidTcmConfig);
  BOOST_CHECK(reader.validateSchema(doc, errorMessage) == false);
}

BOOST_AUTO_TEST_SUITE_END()
