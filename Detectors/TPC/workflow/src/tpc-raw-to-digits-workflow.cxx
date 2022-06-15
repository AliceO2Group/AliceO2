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

#include <fmt/format.h>
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Algorithm/RangeTokenizer.h"
#include "TPCWorkflow/RawToDigitsSpec.h"
#include "TPCWorkflow/RecoWorkflow.h"
#include "TPCBase/Sector.h"
#include <vector>
#include <string>

#include <thread> // to detect number of hardware threads

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-raw-to-digits-.*", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // for the TPC it is useful to take at most half of the available (logical) cores due to memory requirements
  int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);
  const std::string laneshelp("Number of tpc processing lanes. A lane is a pipeline of algorithms.");

  const std::string sectorshelp("List of TPC sectors, comma separated ranges, e.g. 0-3,7,9-15");
  const std::string sectorDefault = "0-" + std::to_string(o2::tpc::Sector::MAXSECTOR - 1);
  const std::string tpcrthelp("Run TPC reco workflow to specified output type, currently supported: 'digits,clusters,tracks'");

  std::vector<ConfigParamSpec> options{
    {"input-spec", VariantType::String, "A:TPC/RAWDATA", {"selection string input specs"}},
    {"tpc-lanes", VariantType::Int, defaultlanes, {laneshelp}},
    {"tpc-sectors", VariantType::String, sectorDefault.c_str(), {sectorshelp}},
    {"tpc-reco-output", VariantType::String, "", {tpcrthelp}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"send-ce-digits", VariantType::Bool, false, {"filter CE digits and publish them for analysis on a separate stream"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCCalibPedestal.FirstTimeBin=10;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

// extract num TPC lanes, a lane is a streaming line of processors (digitizer-clusterizer-etc)
// by default this will be std::max(the number of physical cores, numberofsectors)
// as a temporary means to fully use a machine and as a way to play with different topologies
int getNumTPCLanes(std::vector<int> const& sectors, ConfigContext const& configcontext)
{
  auto lanes = configcontext.options().get<int>("tpc-lanes");
  if (lanes < 0) {
    LOG(fatal) << "tpc-lanes needs to be positive\n";
    return 0;
  }
  // crosscheck with sectors
  return std::min(lanes, (int)sectors.size());
}

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(configcontext.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2-tpc-raw-reco-workflow_configuration.ini");

  WorkflowSpec specs;

  // for the moment no parallel workflow, so just one lane hard coded
  auto tpcSectors = o2::RangeTokenizer::tokenize<int>(configcontext.options().get<std::string>("tpc-sectors"));
  auto lanes = 1; // getNumTPCLanes(tpcSectors, configcontext);
  const bool sendCEdigits = configcontext.options().get<bool>("send-ce-digits");

  int fanoutsize = 0;
  for (int l = 0; l < lanes; ++l) {
    specs.emplace_back(o2::tpc::getRawToDigitsSpec(fanoutsize, configcontext.options().get<std::string>("input-spec"), configcontext.options().get<bool>("ignore-dist-stf"), tpcSectors, sendCEdigits));
    fanoutsize++;
  }

  // ===| attach the TPC reco workflow |========================================
  // default is to dump digits
  std::string_view recoOuput = "digits";
  const auto tpcRecoOutputType = configcontext.options().get<std::string>("tpc-reco-output");
  if (!tpcRecoOutputType.empty()) {
    recoOuput = tpcRecoOutputType.c_str();
  }

  return specs;
}
