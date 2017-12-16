// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/SimpleMetricsService.h"
#include <FairMQLogger.h>

#include <chrono>

#include <cassert>

namespace o2 {
namespace framework {

// All we do is to printout
void SimpleMetricsService::post(const char *label, float value) {
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  LOG(DEBUG) << "METRIC:float:" << label << ":" << now_c << ":" << value;
}

void SimpleMetricsService::post(char const*label, int value) {
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  LOG(DEBUG) << "METRIC:int:" << label << ":" << now_c << ":" << value;
}

void SimpleMetricsService::post(const char *label, const char *value) {
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  LOG(DEBUG) << "METRIC:string:" << label << ":" << now_c << ":" << value;
}

} // framework
} // o2
