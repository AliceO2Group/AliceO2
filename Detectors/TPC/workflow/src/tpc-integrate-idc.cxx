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

#include <vector>
#include <string>
#include "Algorithm/RangeTokenizer.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/TPCIntegrateIDCSpec.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/HBFUtils.h"
#include "TPCSimulation/IDCSim.h"
#include "TPCBase/Sector.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  const std::string sectorDefault = "0-" + std::to_string(o2::tpc::Sector::MAXSECTOR - 1);
  const int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);

  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"nOrbits", VariantType::Int, 12, {"number of orbits for which the IDCs are integrated"}},
    {"outputFormat", VariantType::String, "Sim", {"setting the output format type: 'Sim'=IDC simulation format, 'Real'=real output format of CRUs (not implemented yet)"}},
    {"debug", VariantType::Bool, false, {"create debug tree"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"lanes", VariantType::Int, defaultlanes, {"Number of parallel processing lanes."}},
    {"sectors", VariantType::String, sectorDefault.c_str(), {"List of TPC sectors, comma separated ranges, e.g. 0-3,7,9-15"}},
    {"hbfutils-config", VariantType::String, std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE), {"config file for HBFUtils (or none) to get number of orbits per TF"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  std::string confDig = config.options().get<std::string>("hbfutils-config");
  if (!confDig.empty() && confDig != "none") {
    o2::conf::ConfigurableParam::updateFromFile(confDig, "HBFUtils");
  }
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcintegrateidc_configuration.ini");

  const auto& hbfu = o2::raw::HBFUtils::Instance();
  LOGP(info, "Setting {} orbits per TF", hbfu.getNOrbitsPerTF());
  o2::tpc::IDCSim::setNOrbitsPerTF(hbfu.getNOrbitsPerTF());

  const auto nOrbits = config.options().get<int>("nOrbits");
  const auto outputFormatStr = config.options().get<std::string>("outputFormat");
  const TPCIntegrateIDCDevice::IDCFormat outputFormat = outputFormatStr.compare("Sim") ? TPCIntegrateIDCDevice::IDCFormat::Real : TPCIntegrateIDCDevice::IDCFormat::Sim;
  const auto debug = config.options().get<bool>("debug");
  const auto tpcsectors = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("sectors"));
  const auto nSectors = tpcsectors.size();
  const auto nLanes = std::min(static_cast<unsigned long>(config.options().get<int>("lanes")), nSectors);
  const auto sectorsPerLane = nSectors / nLanes + ((nSectors % nLanes) != 0);

  WorkflowSpec workflow;
  if (nLanes <= 0) {
    return workflow;
  }

  for (int ilane = 0; ilane < nLanes; ++ilane) {
    const auto first = tpcsectors.begin() + ilane * sectorsPerLane;
    if (first >= tpcsectors.end()) {
      break;
    }
    const auto last = std::min(tpcsectors.end(), first + sectorsPerLane);
    const std::vector<unsigned int> rangeSectors(first, last);
    workflow.emplace_back(getTPCIntegrateIDCSpec(ilane, rangeSectors, nOrbits, outputFormat, debug));
  }

  return workflow;
}
