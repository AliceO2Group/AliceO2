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

// example to run:
// o2-dcs-proxy --dcs-proxy '--channel-config "name=dcs-proxy,type=pull,method=connect,address=tcp://10.11.28.22:60000,rateLogging=1,transport=zeromq"' -b

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/Lifetime.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DCStoDPLconverter.h"
#include "CommonUtils/StringUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "Headers/DataHeaderHelpers.h"
#include <vector>
#include <unordered_map>
#include <regex>
#include <string>
#include <unistd.h>

using namespace o2::framework;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DeliveryType = o2::dcs::DeliveryType;
using CcdbManager = o2::ccdb::BasicCCDBManager;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"verbose", VariantType::Bool, false, {"verbose output"}});
  workflowOptions.push_back(ConfigParamSpec{"fbi-report-rate", VariantType::Int, 6, {"report pet N FBI received"}});
  workflowOptions.push_back(ConfigParamSpec{"test-mode", VariantType::Bool, false, {"test mode"}});
  workflowOptions.push_back(ConfigParamSpec{"may-send-delta-first", VariantType::Bool, false, {"if true, do not wait for FBI before sending 1st output"}});
  workflowOptions.push_back(ConfigParamSpec{"ccdb-url", VariantType::String, "http://ccdb-test.cern.ch:8080", {"url of CCDB to get the detectors DPs configuration"}});
  workflowOptions.push_back(ConfigParamSpec{"detector-list", VariantType::String, "TOF, MCH", {"list of detectors for which to process DCS"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  bool verbose = config.options().get<bool>("verbose");
  bool testMode = config.options().get<bool>("test-mode");
  bool fbiFirst = !config.options().get<bool>("may-send-delta-first");
  int repRate = std::max(1, config.options().get<int>("fbi-report-rate"));
  std::string detectorList = config.options().get<std::string>("detector-list");
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  std::string url = config.options().get<std::string>("ccdb-url");

  std::unordered_map<DPID, o2h::DataDescription> dpid2DataDesc;

  if (testMode) {
    DPID dpidtmp;
    DPID::FILL(dpidtmp, "ADAPOS_LG/TEST_000100", DeliveryType::DPVAL_STRING);
    dpid2DataDesc[dpidtmp] = "COMMON"; // i.e. this will go to {DCS/COMMON/0} OutputSpec
    DPID::FILL(dpidtmp, "ADAPOS_LG/TEST_000110", DeliveryType::DPVAL_STRING);
    dpid2DataDesc[dpidtmp] = "COMMON";
    DPID::FILL(dpidtmp, "ADAPOS_LG/TEST_000200", DeliveryType::DPVAL_STRING);
    dpid2DataDesc[dpidtmp] = "COMMON1";
    DPID::FILL(dpidtmp, "ADAPOS_LG/TEST_000240", DeliveryType::DPVAL_INT);
    dpid2DataDesc[dpidtmp] = "COMMON1";
  }

  else {
    auto& mgr = CcdbManager::instance();
    mgr.setURL(url); // http://ccdb-test.cern.ch:8080 or http://localhost:8080 for a local installation
    long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::regex re("[\\s,-]+");
    std::sregex_token_iterator it(detectorList.begin(), detectorList.end(), re, -1);
    std::sregex_token_iterator reg_end;
    for (; it != reg_end; ++it) {
      std::string detStr = it->str();
      o2::utils::Str::trim(detStr);
      if (!detStr.empty()) {
        LOG(info) << "DCS DPs configured for detector " << detStr;
        std::unordered_map<DPID, std::string>* dpid2Det = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>(detStr + "/Config/DCSDPconfig", ts);
        for (auto& el : *dpid2Det) {
          o2::header::DataDescription tmpd;
          tmpd.runtimeInit(el.second.c_str(), el.second.size());
          dpid2DataDesc[el.first] = tmpd;
        }
      }
    }
  }

  // RS: here we should complete the attribution of different DPs to different outputs
  // ...

  // now collect all required outputs to define OutputSpecs for specifyExternalFairMQDeviceProxy
  std::unordered_map<o2h::DataDescription, int, std::hash<o2h::DataDescription>> outMap;
  for (auto itdp : dpid2DataDesc) {
    outMap[itdp.second]++;
  }

  Outputs dcsOutputs;
  for (auto itout : outMap) {
    dcsOutputs.emplace_back("DCS", itout.first, 0, Lifetime::Timeframe);
  }

  DataProcessorSpec dcsProxy = specifyExternalFairMQDeviceProxy(
    "dcs-proxy",
    std::move(dcsOutputs),
    "type=pull,method=connect,address=tcp://aldcsadaposactor:60000,rateLogging=1,transport=zeromq",
    dcs2dpl(dpid2DataDesc, fbiFirst, verbose, repRate));

  WorkflowSpec workflow;
  workflow.emplace_back(dcsProxy);
  return workflow;
}
