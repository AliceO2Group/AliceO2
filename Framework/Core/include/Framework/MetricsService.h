// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_METRICSSERVICE_H
#define FRAMEWORK_METRICSSERVICE_H

#include "Framework/Variant.h"
#include <map>
#include <string>
#include <vector>

namespace o2 {
namespace framework {

/// A simple service which relays metrics and pushes them
/// to the device which will actually be responsible for publishing
/// them. This way the metrics themselves go through the same
/// path as the rest of the communication rather than being "out of bound".
class MetricsService {
public:
  virtual void post(const char *label, float value) = 0;
  virtual void post(const char *label, int value) = 0;
  virtual void post(const char *label, const char *value) = 0;
};

} // framework
} // o2
#endif // FRAMEWORK_METRICSSERVICE_H
