// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/RemoteMetricsService.h"
#include "FairMQLogger.h"

#include <cassert>

namespace o2 {
namespace framework {

// FIXME: for the moment, simply make sure we print out the metric.
//        this should really send a message to an aggregator device
void RemoteMetricsService::commit() {
  assert(mValues.size() == 1);
  assert(mMetrics.size() == 1);
  LOG(DEBUG) << "metric:" << mLabelsIdx[mMetrics[0].first] << ":" << mValues[0];
  mValues.clear();
  mMetrics.clear();
}

size_t RemoteMetricsService::getLabelIdx(const char *label) {
  // FIXME: requires one memory allocation per metric being posted, find a
  // better way of doing it.
  std::string l{label};
  auto li = mLabels.find(l);
  // Every time we insert a new label, we do a vector insert. Hopefully
  // this happens only seldomly.
  size_t idx;
  if (li == mLabels.end()) {
    idx = mLabels.size();
    mLabels.insert(std::make_pair(l, mLabels.size()));
    mLabelsIdx.insert(std::make_pair(mLabelsIdx.size(), l));
  } else {
    idx = li->second;
  }

  return idx;
}

void RemoteMetricsService::post(const char *label, float value) {
  auto idx = getLabelIdx(label);
  mMetrics.emplace_back(std::make_pair(idx, mValues.size()));
  mValues.emplace_back(Variant{value});
  commit();
}

void RemoteMetricsService::post(char const*label, int value) {
  auto idx = getLabelIdx(label);
  mMetrics.emplace_back(std::make_pair(idx, mValues.size()));
  mValues.emplace_back(Variant{value});
  commit();
}

void RemoteMetricsService::post(const char *label, const char *value) {
  auto idx = getLabelIdx(label);
  mMetrics.emplace_back(std::make_pair(idx, mValues.size()));
  mValues.emplace_back(Variant{value});
  commit();
}

} // framework
} // o2
