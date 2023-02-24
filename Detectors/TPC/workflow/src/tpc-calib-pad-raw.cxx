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

/// @brief  TPC Pad-wise raw data calibration
/// @author Jens Wiechula
/// @author David Silvermyr
//
#include <fmt/format.h>
#include "Algorithm/RangeTokenizer.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "DPLUtils/RawParser.h"
#include "Headers/DataHeader.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCCalibration/CalibPedestal.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include <vector>
#include <string>
#include "DetectorsRaw/RDHUtils.h"
#include "TPCBase/RDHUtils.h"
#include "TPCWorkflow/TPCCalibPadRawSpec.h"
#include "TPCWorkflow/CalDetMergerPublisherSpec.h"

using namespace o2::framework;
using RDHUtils = o2::raw::RDHUtils;

const std::string DEFAULTINPUT = "A:TPC/RAWDATA";

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::string sectorDefault = "0-" + std::to_string(o2::tpc::Sector::MAXSECTOR - 1);
  int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);

  std::vector<ConfigParamSpec> options{
    {"input-spec", VariantType::String, DEFAULTINPUT, {"selection string input specs"}},
    {"publish-after-tfs", VariantType::Int, 0, {"number of time frames after which to force publishing the objects"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCCalibPedestal.FirstTimeBin=10;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"calib-type", VariantType::String, "pedestal", {"Calibration type to run: pedestal, pulser, ce"}},
    {"no-write-ccdb", VariantType::Bool, false, {"skip sending the calibration output to CCDB"}},
    {"lanes", VariantType::Int, defaultlanes, {"Number of parallel processing lanes."}},
    {"sectors", VariantType::String, sectorDefault.c_str(), {"List of TPC sectors, comma separated ranges, e.g. 0-3,7,9-15"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

using RDH = o2::header::RAWDataHeader;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpccalibration_configuration.ini");

  std::string inputSpec = config.options().get<std::string>("input-spec");
  const auto skipCCDB = config.options().get<bool>("no-write-ccdb");
  const auto publishAfterTFs = (uint32_t)config.options().get<int>("publish-after-tfs");

  const auto tpcsectors = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("sectors"));
  const auto nSectors = (uint32_t)tpcsectors.size();
  auto nLanes = std::min((uint32_t)config.options().get<int>("lanes"), nSectors);
  const auto sectorsPerLane = nSectors / nLanes + ((nSectors % nLanes) != 0);

  CDBType rawType;
  try {
    rawType = CalibRawTypeMap.at(config.options().get<std::string>("calib-type"));
    if ((rawType == CDBType::CalCE) && (inputSpec == DEFAULTINPUT)) {
      inputSpec = "tpcdigits:TPC/CEDIGITS";
    }
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid writer-type type: ") + config.options().get<std::string>("calib-type"));
  }

  WorkflowSpec workflow;

  if (nLanes <= 0) {
    return workflow;
  }

  const bool digitInput = inputSpec.find("DIGITS") != std::string::npos;
  if (digitInput && nLanes != 1) {
    LOGP(info, "only one lane allowed for DIGIT type input");
    nLanes = 1;
  }
  for (int ilane = 0; ilane < nLanes; ++ilane) {
    auto first = tpcsectors.begin() + ilane * sectorsPerLane;
    if (first >= tpcsectors.end()) {
      break;
    }
    auto last = std::min(tpcsectors.end(), first + sectorsPerLane);
    std::vector<int> range(first, last);
    workflow.emplace_back(getTPCCalibPadRawSpec(inputSpec, ilane, range, publishAfterTFs, rawType));
  }

  workflow.emplace_back(getCalDetMergerPublisherSpec(nLanes, skipCCDB, publishAfterTFs > 0));

  return workflow;
}
