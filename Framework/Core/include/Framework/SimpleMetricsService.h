// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_SIMPLEMETRICSSERVICE_H
#define FRAMEWORK_SIMPLEMETRICSSERVICE_H

#include "Framework/MetricsService.h"
#include "Framework/Variant.h"
#include <map>
#include <string>
#include <vector>

namespace o2 {
namespace framework {

/// A simple metrics service which prints out metrics so tha
/// they can be collected by the driver process.
class SimpleMetricsService : public MetricsService{
public:
  void post(const char *label, float value) final;
  void post(const char *label, int value) final;
  void post(const char *label, const char *value) final;
};

} // framework
} // o2
#endif // FRAMEWORK_SIMPLEMETRICSSERVICE_H
