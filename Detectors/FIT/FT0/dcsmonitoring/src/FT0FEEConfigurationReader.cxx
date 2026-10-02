#include "FT0DCSMonitoring/FT0FEEConfigurationReader.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>

namespace o2::ft0
{
Ft0FeeConfiguration FT0FEEConfigurationReader::parseFeeConfiguration(gsl::span<const char> buffer)
{
  return FITFEEConfigurationReader<FT0FEEConfigurationReader>::parseFeeConfiguration<Ft0FeeConfiguration>(buffer);
}

void FT0FEEConfigurationReader::parseTriggers(const rapidjson::Value& root, const char* triggersNodeName, TriggersConfig& config)
{
  const auto& triggersNode = root["triggers"];

  if(triggersNode.HasMember("vertex_time_low_threshold") == false) {
    throw std::runtime_error("Missing vertex_time_low_threshold trigger node!");
  }
  const auto& vertexTimeLowThresholdNode = triggersNode["vertex_time_low_threshold"];

  if(triggersNode.HasMember("vertex_time_high_threshold") == false) {
    throw std::runtime_error("Missing vertex_time_high_threshold trigger node!");
  }
  const auto& vertexTimeHighThresholdNode = triggersNode["vertex_time_high_threshold"];

  if(triggersNode.HasMember("semi_central_a") == false) {
    throw std::runtime_error("Missing semi_central_a trigger node!");
  }
  const auto& semicentralANode = triggersNode["semi_central_a"];

  if(triggersNode.HasMember("semi_central_c") == false) {
    throw std::runtime_error("Missing semi_central_c trigger node!");
  }
  const auto& semicentralCNode = triggersNode["semi_central_c"];

  if(triggersNode.HasMember("central_a") == false) {
    throw std::runtime_error("Missing central_a trigger node!");
  }
  const auto& centralANode = triggersNode["central_a"];

  if(triggersNode.HasMember("central_c") == false) {
    throw std::runtime_error("Missing central_c trigger node!");
  }
  const auto& centralCNode = triggersNode["central_c"];

  if(triggersNode.HasMember("sides_combination_mode") == false) {
    throw std::runtime_error("Missing sides_combination_mode node!");
  }
  const auto& sidesCombinationMode = triggersNode["sides_combination_mode"];

  config.vertexTimeLowThreshold = vertexTimeLowThresholdNode.GetInt();
  config.vertexTimeHighThreshold = vertexTimeHighThresholdNode.GetInt();
  config.semicentralA = semicentralANode.GetUint();
  config.semicentralC = semicentralCNode.GetUint();
  config.centralA = centralANode.GetUint();
  config.centralC = centralCNode.GetUint();
  config.sidesCombinationMode = sidesCombinationMode.GetUint();
}
} // namespace o2::ft0