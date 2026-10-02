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

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "CCDB/CcdbApi.h"
#include "DataFormatsFT0/FeeConfiguration.h"

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/document.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#endif

void fetchFt0FeeConfig(const std::string ccdbUrl = "http://alice-ccdb.cern.ch", long timestamp = -1, const std::string fileName = "ft0-fee-config.json", const std::string ccdbPath = "FT0/Config/FeeConfiguration")
{
  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);

  std::map<std::string, std::string> metadata;
  std::unique_ptr<o2::ft0::Ft0FeeConfiguration> config(ccdbApi.retrieveFromTFileAny<o2::ft0::Ft0FeeConfiguration>(ccdbPath, metadata, timestamp));

  if (!config) {
    throw std::runtime_error("Cannot retrieve Ft0FeeConfiguration from CCDB path " + ccdbPath);
  }

  rapidjson::Document doc;
  doc.SetObject();
  auto& allocator = doc.GetAllocator();

  rapidjson::Value channels(rapidjson::kObjectType);
  rapidjson::Value timeAligments(rapidjson::kArrayType);
  rapidjson::Value cfdThresholds(rapidjson::kArrayType);
  rapidjson::Value cfdZeros(rapidjson::kArrayType);
  rapidjson::Value adcZeros(rapidjson::kArrayType);
  rapidjson::Value adcDelays(rapidjson::kArrayType);
  rapidjson::Value rangeCorrectionAdc0(rapidjson::kArrayType);
  rapidjson::Value rangeCorrectionAdc1(rapidjson::kArrayType);
  rapidjson::Value channelMaskData(rapidjson::kArrayType);
  rapidjson::Value channelMaskTriggers(rapidjson::kArrayType);

  for (int i = 0; i < o2::ft0::Ft0FeeConfiguration::NChannels; ++i) {
    timeAligments.PushBack(config->channels.timeAligments[i], allocator);
    cfdThresholds.PushBack(config->channels.cfdThresholds[i], allocator);
    cfdZeros.PushBack(config->channels.cfdZeros[i], allocator);
    adcZeros.PushBack(config->channels.adcZeros[i], allocator);
    adcDelays.PushBack(config->channels.adcDelays[i], allocator);
    rangeCorrectionAdc0.PushBack(config->channels.rangeCorrectionAdc0[i], allocator);
    rangeCorrectionAdc1.PushBack(config->channels.rangeCorrectionAdc1[i], allocator);
    channelMaskData.PushBack(config->channels.channelMaskData[i], allocator);
    channelMaskTriggers.PushBack(config->channels.channelMaskTriggers[i], allocator);
  }

  channels.AddMember("time_aligments", timeAligments, allocator);
  channels.AddMember("cfd_thresholds", cfdThresholds, allocator);
  channels.AddMember("cfd_zeros", cfdZeros, allocator);
  channels.AddMember("adc_zeros", adcZeros, allocator);
  channels.AddMember("adc_delays", adcDelays, allocator);
  channels.AddMember("range_correction_adc0", rangeCorrectionAdc0, allocator);
  channels.AddMember("range_correction_adc1", rangeCorrectionAdc1, allocator);
  channels.AddMember("channel_mask_data", channelMaskData, allocator);
  channels.AddMember("channel_mask_triggers", channelMaskTriggers, allocator);
  doc.AddMember("channels", channels, allocator);

  rapidjson::Value tcm(rapidjson::kObjectType);
  tcm.AddMember("phase_delay_a", config->tcm.phaseDelayA, allocator);
  tcm.AddMember("phase_delay_c", config->tcm.phaseDelayC, allocator);
  doc.AddMember("tcm", tcm, allocator);

  rapidjson::Value pmA(rapidjson::kArrayType);

  for (int i = 0; i < 10; ++i) {
    rapidjson::Value pm(rapidjson::kObjectType);
    pm.AddMember("or_gate", config->pmA[i].orGate, allocator);
    pm.AddMember("trg_charge_low_level", config->pmA[i].trgChargeLowLevel, allocator);
    pm.AddMember("trg_charge_high_level", config->pmA[i].trgChargeHighLevel, allocator);
    pmA.PushBack(pm, allocator);
  }
  doc.AddMember("pm_a", pmA, allocator);

  rapidjson::Value pmC(rapidjson::kArrayType);
  for (int i = 0; i < 10; ++i) {
    rapidjson::Value pm(rapidjson::kObjectType);
    pm.AddMember("or_gate", config->pmC[i].orGate, allocator);
    pm.AddMember("trg_charge_low_level", config->pmC[i].trgChargeLowLevel, allocator);
    pm.AddMember("trg_charge_high_level", config->pmC[i].trgChargeHighLevel, allocator);
    pmC.PushBack(pm, allocator);
  }
  doc.AddMember("pm_c", pmC, allocator);

  rapidjson::Value triggers(rapidjson::kObjectType);
  triggers.AddMember("vertex_time_low_threshold", config->triggers.vertexTimeLowThreshold, allocator);

  triggers.AddMember("vertex_time_high_threshold", config->triggers.vertexTimeHighThreshold, allocator);
  triggers.AddMember("semicentral_a", config->triggers.semicentralA, allocator);
  triggers.AddMember("semicentral_c", config->triggers.semicentralC, allocator);
  triggers.AddMember("central_a", config->triggers.centralA, allocator);
  triggers.AddMember("central_c", config->triggers.centralC, allocator);
  triggers.AddMember("sides_combination_mode", config->triggers.sidesCombinationMode, allocator);
  doc.AddMember("triggers", triggers, allocator);

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);

  doc.Accept(writer);

  if (fileName.empty()) {
    std::cout << buffer.GetString() << '\n';
    return;
  }

  std::ofstream output(fileName);

  if (!output) {
    throw std::runtime_error(
      "Cannot open output file " + fileName);
  }

  output << buffer.GetString() << '\n';

  std::cout << "FT0 FEE configuration written to " << fileName << '\n';
}
