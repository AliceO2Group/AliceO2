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

#define BOOST_TEST_MODULE Test FITDCSMonitoring FITHvConfigurationReader
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <rapidjson/document.h>
#include "FITDCSMonitoring/FITHvConfigurationReader.h"
#include "DataFormatsFIT/Configuration.h"

using namespace o2::fit;

BOOST_AUTO_TEST_SUITE(o2_fit_dcs_monitoring)

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

template <size_t NChannels>
void addToPayloadHvChannels(rapidjson::Document& doc, float (&gainValues)[NChannels])
{
  auto& allocator = doc.GetAllocator();

  rapidjson::Value gainJson(rapidjson::kArrayType);
  for (int idx = 0; idx < NChannels; idx++) {
    gainJson.PushBack(gainValues[idx], allocator);
  }
  rapidjson::Value hvChannelsJson(rapidjson::kObjectType);
  hvChannelsJson.AddMember("gain", gainJson, allocator);
  doc.AddMember("hv_channels", hvChannelsJson, allocator);
}

template <size_t NChannels>
HvChannelsConfig<NChannels> createExpectedConfig(float (&gainValues)[NChannels])
{
  HvChannelsConfig<NChannels> expectedConfig;
  std::memcpy(std::begin(expectedConfig.gain), gainValues, sizeof(float) * NChannels);
  return expectedConfig;
}

BOOST_AUTO_TEST_CASE(shouldParseHvChannels)
{
  constexpr size_t NChannels = 5;
  float gainValues[NChannels] = {1.0, 2.1, 3.2, 4.3, 5.4};

  rapidjson::Document doc = createEmptyPayload();
  addToPayloadHvChannels(doc, gainValues);

  HvChannelsConfig<NChannels> channelsConfig;
  FITHvConfigurationReader reader;
  reader.parseHvChannelData(doc, "hv_channels", channelsConfig);

  HvChannelsConfig<NChannels> expectedConfig = createExpectedConfig(gainValues);

  BOOST_CHECK(expectedConfig == channelsConfig);
}

BOOST_AUTO_TEST_CASE(shouldAcceptValidJsonPayload)
{
  const char* jsonPayload = R"json(
    {
        "hv_channels" : {
            "gain": [1,2,3,4]
        }
    }
    )json";

  rapidjson::Document doc = createDocumentFromString(jsonPayload);

  FITHvConfigurationReader reader;
  std::string errorMessage;
  BOOST_CHECK_MESSAGE(reader.validateSchema(doc, errorMessage), errorMessage);
}

BOOST_AUTO_TEST_SUITE_END()