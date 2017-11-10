// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma once

#include <fstream>
#include <string>

#include <dds_intercom.h>
#include <boost/property_tree/ptree.hpp>

namespace o2
{
namespace qc
{
class MetricsExtractor
{
 public:
  MetricsExtractor(const char* testName);
  ~MetricsExtractor();

  void runMetricsExtractor();

 private:
  dds::intercom_api::CIntercomService mService;
  std::unique_ptr<dds::intercom_api::CCustomCmd> ddsCustomCmd;
  std::ofstream metricsOutputFile;
  std::ofstream stateOutputFile;

  unsigned int mInternalMetricId{ 0 };
  unsigned int mInternalStateId{ 0 };
  const char* mTestName;

  std::string convertToString(boost::property_tree::ptree& command);
  void sendCheckStateDdsCustomCommand();
  void sendGetMetrics();
  void subscribeDdsCommands();
};
}
}