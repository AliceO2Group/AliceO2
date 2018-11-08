// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Metric2DViewIndex.h"

#include "Framework/DeviceMetricsInfo.h"

#include <fairmq/FairMQLogger.h>

#include <algorithm>
#include <functional>

namespace o2
{
namespace framework
{

Metric2DViewIndex::Updater Metric2DViewIndex::getUpdater(Metric2DViewIndex& view)
{
  return [&view](std::string const& name, MetricInfo const& metric, int value) -> void {
    if (view.prefix.size() > name.size()) {
      return;
    }
    if (std::mismatch(view.prefix.begin(), view.prefix.end(), name.begin()).first != view.prefix.end()) {
      return;
    }

    auto extra = name;

    // +1 is to remove the /
    extra.erase(0, view.prefix.size() + 1);
    if (extra == "w") {
      view.w = value;
      view.indexes.resize(view.w * view.h);
      return;
    } else if (extra == "h") {
      view.h = value;
      view.indexes.resize(view.w * view.h);
      return;
    }
    int idx = -1;
    try {
      idx = std::stoi(extra, nullptr, 10);
    } catch (...) {
      LOG(ERROR) << "Badly formatted metric";
    }
    if (idx < 0) {
      LOG(ERROR) << "Negative metric";
      return;
    }
    if (view.indexes.size() <= idx) {
      view.indexes.resize(std::max(idx + 1, view.w * view.h));
    }
    view.indexes[idx] = metric.storeIdx;
  };
}

} // namespace framework
} // namespace o2
