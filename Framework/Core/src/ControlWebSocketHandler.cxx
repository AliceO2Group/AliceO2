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

#include "ControlWebSocketHandler.h"
#include "DriverServerContext.h"
#include "Framework/DeviceMetricsHelper.h"
#include "Framework/ServiceMetricsInfo.h"
#include <regex>
#include "Framework/Logger.h"
#include "Framework/DeviceConfigInfo.h"

namespace o2::framework
{
void ControlWebSocketHandler::frame(char const* frame, size_t s)
{
  bool hasNewMetric = false;
  std::array<Metric2DViewIndex*, 2> model = {&(*mContext.infos)[mIndex].inputChannelMetricsViewIndex,
                                             &(*mContext.infos)[mIndex].outputChannelMetricsViewIndex};
  auto updateMetricsViews = Metric2DViewIndex::getUpdater();

  auto newMetricCallback = [&](std::string const& name, MetricInfo const& metric, int value, size_t metricIndex) {
    updateMetricsViews(model, name, metric, value, metricIndex);
    hasNewMetric = true;
  };
  std::string_view tokenSV(frame, s);
  ParsedMetricMatch metricMatch;

  auto doParseConfig = [](std::string_view const& token, ParsedConfigMatch& configMatch, DeviceInfo& info) -> bool {
    if (DeviceConfigHelper::parseConfig(token, configMatch)) {
      DeviceConfigHelper::processConfig(configMatch, info);
      return true;
    }
    return false;
  };
  LOG(debug3) << "Data received: " << std::string_view(frame, s);
  if (DeviceMetricsHelper::parseMetric(tokenSV, metricMatch)) {
    // We use this callback to cache which metrics are needed to provide a
    // the DataRelayer view.
    assert(mContext.metrics);
    DeviceMetricsHelper::processMetric(metricMatch, (*mContext.metrics)[mIndex], newMetricCallback);
    didProcessMetric = true;
    didHaveNewMetric |= hasNewMetric;
    return;
  }

  ParsedConfigMatch configMatch;
  std::string_view const token(frame, s);
  std::match_results<std::string_view::const_iterator> match;

  if (ControlServiceHelpers::parseControl(token, match) && mContext.infos) {
    ControlServiceHelpers::processCommand(*mContext.infos, *mContext.states, mPid, match[1].str(), match[2].str());
  } else if (doParseConfig(token, configMatch, (*mContext.infos)[mIndex]) && mContext.infos) {
    LOG(debug2) << "Found configuration information for pid " << mPid;
  } else {
    LOG(error) << "Unexpected control data: " << std::string_view(frame, s);
  }
}

ControlWebSocketHandler::ControlWebSocketHandler(DriverServerContext& context)
  : mContext{context}
{
}
void ControlWebSocketHandler::endChunk()
{
  if (!didProcessMetric) {
    return;
  }
  size_t timestamp = uv_now(mContext.loop);
  assert(mContext.metrics);
  assert(mContext.infos);
  assert(mContext.states);
  assert(mContext.specs);
  assert(mContext.driver);

  for (auto& callback : *mContext.metricProcessingCallbacks) {
    callback(mContext.registry, ServiceMetricsInfo{*mContext.metrics, *mContext.specs, *mContext.infos, mContext.driver->metrics}, timestamp);
  }
  for (auto& metricsInfo : *mContext.metrics) {
    std::fill(metricsInfo.changed.begin(), metricsInfo.changed.end(), false);
  }
}

void ControlWebSocketHandler::headers(std::map<std::string, std::string> const& headers)
{
  if (headers.count("x-dpl-pid")) {
    auto s = headers.find("x-dpl-pid");
    this->mPid = std::stoi(s->second);
    for (size_t di = 0; di < mContext.infos->size(); ++di) {
      if ((*mContext.infos)[di].pid == mPid) {
        mIndex = di;
        return;
      }
    }
  }
}
} // namespace o2::framework
