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
#ifndef O2_FRAMEWORK_CONTROLWEBSOCKETHANDLER_H_
#define O2_FRAMEWORK_CONTROLWEBSOCKETHANDLER_H_
#include "HTTPParser.h"
#include "DriverServerContext.h"
#include "Framework/DeviceConfigInfo.h"
#include "ControlServiceHelpers.h"
#include "Framework/Logger.h"
#include "Framework/DeviceMetricsHelper.h"
#include <regex>

namespace o2::framework
{
/// An handler for a websocket message stream.
struct ControlWebSocketHandler : public WebSocketHandler {
  ControlWebSocketHandler(DriverServerContext& context)
    : mContext{context}
  {
  }

  ~ControlWebSocketHandler() override = default;

  /// Invoked at the end of the headers.
  /// as a special header we have "x-dpl-pid" which devices can use
  /// to identify themselves.
  /// FIXME: No effort is done to guarantee their identity. Maybe each device
  ///        should be started with a unique secret if we wanted to provide
  ///        some secutity.
  void headers(std::map<std::string, std::string> const& headers) override
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
  /// FIXME: not implemented by the backend.
  void beginFragmentation() override {}

  /// Invoked when a frame it's parsed. Notice you do not own the data and you must
  /// not free the memory.
  void frame(char const* frame, size_t s) override
  {
    bool hasNewMetric = false;
    auto updateMetricsViews = Metric2DViewIndex::getUpdater({&(*mContext.infos)[mIndex].dataRelayerViewIndex,
                                                             &(*mContext.infos)[mIndex].variablesViewIndex,
                                                             &(*mContext.infos)[mIndex].queriesViewIndex,
                                                             &(*mContext.infos)[mIndex].outputsViewIndex});

    auto newMetricCallback = [&updateMetricsViews, &metrics = mContext.metrics, &hasNewMetric](std::string const& name, MetricInfo const& metric, int value, size_t metricIndex) {
      updateMetricsViews(name, metric, value, metricIndex);
      hasNewMetric = true;
    };
    std::string token(frame, s);
    std::smatch match;
    ParsedConfigMatch configMatch;
    ParsedMetricMatch metricMatch;

    auto doParseConfig = [](std::string const& token, ParsedConfigMatch& configMatch, DeviceInfo& info) -> bool {
      auto ts = "                 " + token;
      if (DeviceConfigHelper::parseConfig(ts, configMatch)) {
        DeviceConfigHelper::processConfig(configMatch, info);
        return true;
      }
      return false;
    };
    LOG(debug3) << "Data received: " << std::string_view(frame, s);
    if (DeviceMetricsHelper::parseMetric(token, metricMatch)) {
      // We use this callback to cache which metrics are needed to provide a
      // the DataRelayer view.
      assert(mContext.metrics);
      DeviceMetricsHelper::processMetric(metricMatch, (*mContext.metrics)[mIndex], newMetricCallback);
      didProcessMetric = true;
      didHaveNewMetric |= hasNewMetric;
    } else if (ControlServiceHelpers::parseControl(token, match) && mContext.infos) {
      ControlServiceHelpers::processCommand(*mContext.infos, mPid, match[1].str(), match[2].str());
    } else if (doParseConfig(token, configMatch, (*mContext.infos)[mIndex]) && mContext.infos) {
      LOG(debug2) << "Found configuration information for pid " << mPid;
    } else {
      LOG(error) << "Unexpected control data: " << std::string_view(frame, s);
    }
  }

  /// FIXME: not implemented
  void endFragmentation() override{};
  /// FIXME: not implemented
  void control(char const* frame, size_t s) override{};

  /// Invoked at the beginning of some incoming data. We simply
  /// reset actions which need to happen on a per chunk basis.
  void beginChunk() override
  {
    didProcessMetric = false;
    didHaveNewMetric = false;
  }

  /// Invoked after we have processed all the available incoming data.
  /// In this particular case we must handle the metric callbacks, if
  /// needed.
  void endChunk() override
  {
    if (!didProcessMetric) {
      return;
    }
    size_t timestamp = uv_now(mContext.loop);
    for (auto& callback : *mContext.metricProcessingCallbacks) {
      callback(*mContext.registry, *mContext.metrics, *mContext.specs, *mContext.infos, mContext.driver->metrics, timestamp);
    }
    for (auto& metricsInfo : *mContext.metrics) {
      std::fill(metricsInfo.changed.begin(), metricsInfo.changed.end(), false);
    }
    if (didHaveNewMetric) {
      DeviceMetricsHelper::updateMetricsNames(*mContext.driver, *mContext.metrics);
    }
  }

  /// The driver context were we want to accumulate changes
  /// which we got from the websocket.
  DriverServerContext& mContext;
  /// The pid of the remote process actually associated to this
  /// handler. Notice that this information comes as part of
  /// the HTTP headers via x-dpl-pid.
  pid_t mPid = 0;
  /// The index of the remote process associated to this handler.
  size_t mIndex = (size_t)-1;
  /// Wether any frame operation between beginChunk and endChunk
  /// actually processed some metric.
  bool didProcessMetric = false;
  bool didHaveNewMetric = false;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONTROLWEBSOCKETHANDLER_H_
