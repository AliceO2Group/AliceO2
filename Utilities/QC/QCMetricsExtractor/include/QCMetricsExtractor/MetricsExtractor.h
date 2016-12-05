#pragma once

#include <fstream>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <dds_intercom.h>

class MetricsExtractor
{
public:
  MetricsExtractor(const char * testName);
  ~MetricsExtractor();
  void runMetricsExtractor();
  
private:
  dds::intercom_api::CIntercomService mService;
  std::unique_ptr<dds::intercom_api::CCustomCmd> ddsCustomCmd;
  std::ofstream metricsOutputFile;
  std::ofstream stateOutputFile;

  unsigned int mInternalMetricId {0};
  unsigned int mInternalStateId {0};
  const char * mTestName;

  std::string convertToString(boost::property_tree::ptree & command); 
  void sendCheckStateDdsCustomCommand();
  void sendGetMetrics();
  void subscribeDdsCommands();
};
