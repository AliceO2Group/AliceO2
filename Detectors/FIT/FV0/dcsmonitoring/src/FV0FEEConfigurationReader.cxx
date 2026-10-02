#include "FV0DCSMonitoring/FV0FEEConfigurationReader.h"

namespace o2::fv0
{
Fv0FeeConfiguration FV0FEEConfigurationReader::parseFeeConfiguration(gsl::span<const char> configBuf)
{
  return FITFEEConfigurationReader<FV0FEEConfigurationReader>::parseFeeConfiguration<Fv0FeeConfiguration>(configBuf);
}

void FV0FEEConfigurationReader::parseTriggers(const rapidjson::Value& root, const char* triggersNodeName, TriggersConfig& config)
{
  const auto& triggersNode = root["triggers"];
  
  if(triggersNode.HasMember("n_channels_level") == false) {
    throw std::runtime_error("Missing n_channels_level trigger node!");
  }
  const auto& nChannelsLevelNode = triggersNode["n_channels_level"];

  if(triggersNode.HasMember("inner_rings_level") == false) {
    throw std::runtime_error("Missing inner_rings_level trigger node!");
  }
  const auto& innerRingsLevelNode = triggersNode["inner_rings_level"];

  if(triggersNode.HasMember("charge_level") == false) {
    throw std::runtime_error("Missing charge_level trigger node!");
  }
  const auto& chargeLevelNode = triggersNode["charge_level"];

  if(triggersNode.HasMember("outer_rings_level") == false) {
    throw std::runtime_error("Missing outer_rings_level trigger node!");
  }
  const auto& outerRingsLevelNode = triggersNode["outer_rings_level"];

  if(triggersNode.HasMember("sides_combination_mode") == false) {
    throw std::runtime_error("Missing sides_combination_mode node!");
  }
  const auto& sidesCombinationMode = triggersNode["sides_combination_mode"];

  config.innerRings = innerRingsLevelNode.GetInt();
  config.nChannels = nChannelsLevelNode.GetInt();
  config.charge = chargeLevelNode.GetInt();
  config.outerRings = chargeLevelNode.GetInt();
  config.sidesCombinationMode = sidesCombinationMode.GetUint();
}
} // namespace o2::fv0