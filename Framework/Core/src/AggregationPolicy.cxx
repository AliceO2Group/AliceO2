// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/AggregationPolicy.h"
#include "Framework/Logger.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <stdexcept>

using namespace o2::framework::metricaggregator;

std::vector<std::string> AggregationPolicy::split(std::string_view input, char delim) const
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream{std::string(input)};
  while (std::getline(tokenStream, token, delim)) {
    tokens.push_back(token);
  }
  return tokens;
}

void AggregationPolicy::configureFromEnv()
{
  const char* envPolicy = std::getenv("ALIEN_JDL_AGGREGATOR_POLICY");
  if (!envPolicy) {
    LOGP(warn, "[AggregationPolicy] ALIEN_JDL_AGGREGATOR_POLICY is not set. Using default 'all:simple'.");
    mSelection = AggregationSelectionType::All;
    mReduction = AggregationMetricType::Simple;
    return;
  }

  try {
    std::string policyStr(envPolicy);
    std::vector<std::string> parts = split(policyStr, ':');

    if (parts.size() < 2) {
      LOGP(error, "[AggregationPolicy] Invalid ALIEN_JDL_AGGREGATOR_POLICY format");
      return;
    }

    mSelection = parseSelectionType(parts[0]);
    mReduction = parseReductionType(parts[1]);

    if (mSelection == AggregationSelectionType::Specific) {
      const char* envDevices = std::getenv("ALIEN_JDL_AGGREGATOR_DEVICES");
      if (!envDevices) {
        throw std::invalid_argument("ALIEN_JDL_AGGREGATOR_DEVICES environment variable is required when selection type is 'specific'");
      }
      mSpecificDevices = split(std::string(envDevices), ',');
    }
    if (mReduction == AggregationMetricType::Specific) {
      const char* envMetrics = std::getenv("ALIEN_JDL_AGGREGATOR_METRICS");
      if (!envMetrics) {
        LOGP(warn, "[AggregationPolicy] ALIEN_JDL_AGGREGATOR_METRICS environment variable missing for 'specific' reduction type. Using default.");
        mSpecificMetricRules.push_back({std::regex(".*"), AggregationMetricType::Sum});
        return;
      }

      std::stringstream metricsStream(envMetrics);
      std::string metricRuleStr;
      while (std::getline(metricsStream, metricRuleStr, ';')) {
        auto pos = metricRuleStr.find(':');
        if (pos == std::string::npos) {
          throw std::invalid_argument("Invalid metric rule format: " + metricRuleStr);
        }
        std::string typeStr = metricRuleStr.substr(0, pos);
        std::string patternStr = metricRuleStr.substr(pos + 1);
        AggregationMetricType type = parseReductionType(typeStr);
        mSpecificMetricRules.push_back({std::regex(patternStr), type});
      }
    }
  } catch (std::exception const& e) {
    LOGP(error, "[AggregationPolicy] Failed to parse ALIEN_JDL_AGGREGATOR_POLICY: {}", e.what());
  }
}

AggregationMetricType AggregationPolicy::getAggregationTypeForMetric(std::string_view metricName) const
{
  if (mReduction != AggregationMetricType::Specific) {
    return mReduction;
  }
  for (const auto& rule : mSpecificMetricRules) {
    if (std::regex_match(std::string(metricName), rule.metricPattern)) {
      return rule.type;
    }
  }
  if (mReduction == AggregationMetricType::Specific) {
    LOGP(error, "[AggregationPolicy] No specific aggregation type found for metric '{}'", metricName);
  }
  throw std::invalid_argument("No specific aggregation type found for metric: " + std::string(metricName));
}

AggregationSelectionType AggregationPolicy::getSelection() const noexcept
{
  return mSelection;
}

AggregationMetricType AggregationPolicy::getReduction() const noexcept
{
  return mReduction;
}

AggregationSelectionType AggregationPolicy::parseSelectionType(const std::string& str)
{
  if (str == "all") {
    return AggregationSelectionType::All;
  } else if (str == "specific") {
    return AggregationSelectionType::Specific;
  }
  throw std::invalid_argument("Invalid selection type: " + str);
}

AggregationMetricType AggregationPolicy::parseReductionType(const std::string& str)
{
  if (str == "sum") {
    return AggregationMetricType::Sum;
  } else if (str == "average") {
    return AggregationMetricType::Average;
  } else if (str == "rate") {
    return AggregationMetricType::Rate;
  } else if (str == "simple") {
    return AggregationMetricType::Simple;
  } else if (str == "specific") {
    return AggregationMetricType::Specific;
  }
  throw std::invalid_argument("Invalid reduction type: " + str);
}

bool AggregationPolicy::selectDevice(std::string_view deviceId) const
{
  if (mSelection == AggregationSelectionType::Specific) {
    return std::find(mSpecificDevices.begin(), mSpecificDevices.end(), deviceId) != mSpecificDevices.end();
  }
  return true;
}
