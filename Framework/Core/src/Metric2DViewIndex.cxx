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

#include "Framework/Metric2DViewIndex.h"

#include "Framework/DeviceMetricsInfo.h"

#include "Framework/Logger.h"

#include <algorithm>
#include <functional>

namespace o2::framework
{

Metric2DViewIndex::Updater Metric2DViewIndex::getUpdater()
{
  return [](std::array<Metric2DViewIndex*, 2> const& views, std::string const& name, MetricInfo const& metric, int value, int metricsIndex) -> void {
    for (auto viewPtr : views) {
      auto& view = *viewPtr;
      if (view.prefix.size() > name.size()) {
        continue;
      }
      if (std::mismatch(view.prefix.begin(), view.prefix.end(), name.begin()).first != view.prefix.end()) {
        continue;
      }

      auto extra = name;

      // +1 is to remove the /
      extra.erase(0, view.prefix.size() + 1);
      if (extra == "w") {
        view.w = value;
        view.indexes.resize(view.w * view.h, -1);
        return;
      } else if (extra == "h") {
        view.h = value;
        view.indexes.resize(view.w * view.h, -1);
        return;
      }
      int idx = -1;
      try {
        idx = std::stoi(extra, nullptr, 10);
      } catch (...) {
        LOG(error) << "Badly formatted metric";
      }
      if (idx < 0) {
        LOG(error) << "Negative metric";
        return;
      }
      if (view.indexes.size() <= idx) {
        view.indexes.resize(std::max(idx + 1, view.w * view.h), -1);
      }
      view.indexes[idx] = metricsIndex;
    }
  };
}

} // namespace o2
