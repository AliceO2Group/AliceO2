// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_REMOTEMETRICSSERVICE_H
#define FRAMEWORK_REMOTEMETRICSSERVICE_H

#include "Framework/MetricsService.h"
#include "Framework/Variant.h"
#include <map>
#include <string>
#include <vector>

namespace o2 {
namespace framework {

// This is a metrics service which sends metrics
// to a separate MetricsDevice which funnels them
// to the appropriate backend
class RemoteMetricsService : public MetricsService {
public:
  void post(const char *label, float value) final;
  void post(const char *label, int value) final;
  void post(const char *label, const char *value);
private:
  size_t getLabelIdx(const char *label);
  void commit(void);

  std::map<std::string, size_t> mLabels;
  std::map<size_t, std::string> mLabelsIdx;
  std::vector<Variant> mValues;
  std::vector<std::pair<size_t, size_t>> mMetrics;
  size_t mCurrentIdx = 0;
};

} // framework
} // o2
#endif // FRAMEWORK_REMOTEMETRICSSERVICE_H
