// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DUMMYMETRICSSERVICE_H
#define FRAMEWORK_DUMMYMETRICSSERVICE_H

#include "Framework/MetricsService.h"

namespace o2 {
namespace framework {

/// A dummy service for metrics which does nothing. For tests.
class DummyMetricsService : public MetricsService {
public:
  void post(const char *label, float value) final {}
  void post(const char *label, int value) final {}
  void post(const char *label, const char *value) final {}
};

} // framework
} // o2
#endif // FRAMEWORK_DUMMYMETRICSSERVICE_H
