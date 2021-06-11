// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include <vector>
#include <string>
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "TPCReconstruction/KrCluster.h"
#include "TPCBase/Sector.h"
#include "TPCWorkflow/KryptonClustererSpec.h"
#include "TPCWorkflow/KryptonClustererSpec.h"

using namespace o2::framework;
using namespace o2::tpc;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-krypton-clusterer-.*", CompletionPolicy::CompletionOp::Consume));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::string sectorDefault = "0-" + std::to_string(o2::tpc::Sector::MAXSECTOR - 1);
  int defaultlanes = std::max(1u, std::thread::hardware_concurrency() / 2);

  std::vector<ConfigParamSpec> options{
    {"input-spec", VariantType::String, "A:TPC/RAWDATA", {"selection string input specs"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCCalibPedestal.FirstTimeBin=10;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"outputFile", VariantType::String, "./tpcBoxClusters.root", {"output file name for the box cluster root file"}},
    {"lanes", VariantType::Int, defaultlanes, {"Number of parallel processing lanes."}},
    {"sectors", VariantType::String, sectorDefault.c_str(), {"List of TPC sectors, comma separated ranges, e.g. 0-3,7,9-15"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpccalibration_configuration.ini");

  const std::string inputSpec = config.options().get<std::string>("input-spec");
  const std::string outputFile = config.options().get<std::string>("outputFile");

  const auto tpcSectors = o2::RangeTokenizer::tokenize<int>(config.options().get<std::string>("sectors"));
  const auto nSectors = (int)tpcSectors.size();
  const auto nLanes = std::min(config.options().get<int>("lanes"), nSectors);
  const auto sectorsPerLane = nSectors / nLanes + ((nSectors % nLanes) != 0);

  WorkflowSpec workflow;

  if (nLanes <= 0) {
    return workflow;
  }

  for (int ilane = 0; ilane < nLanes; ++ilane) {
    auto first = tpcSectors.begin() + ilane * sectorsPerLane;
    if (first >= tpcSectors.end()) {
      break;
    }
    auto last = std::min(tpcSectors.end(), first + sectorsPerLane);
    std::vector<int> range(first, last);
    workflow.emplace_back(getKryptonClustererSpec(inputSpec, ilane, range));
  }

  std::vector<int> laneConfiguration = tpcSectors; // Currently just a copy of the tpcSectors, why?

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // generation of processor specs for various types of outputs
  // based on generic RootTreeWriter and MakeRootTreeWriterSpec generator
  //
  // -------------------------------------------------------------------------------------------
  // the callbacks for the RootTreeWriter
  //
  // The generic writer needs a way to associate incoming data with the individual branches for
  // the TPC sectors. The sector number is transmitted as part of the sector header, the callback
  // finds the corresponding index in the vector of configured sectors
  auto getIndex = [tpcSectors](o2::framework::DataRef const& ref) {
    auto const* tpcSectorHeader = o2::framework::DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
    if (!tpcSectorHeader) {
      throw std::runtime_error("TPC sector header missing in header stack");
    }
    if (tpcSectorHeader->sector() < 0) {
      // special data sets, don't write
      return ~(size_t)0;
    }
    size_t index = 0;
    for (auto const& sector : tpcSectors) {
      if (sector == tpcSectorHeader->sector()) {
        return index;
      }
      ++index;
    }
    throw std::runtime_error("sector " + std::to_string(tpcSectorHeader->sector()) + " not configured for writing");
  };
  auto getName = [tpcSectors](std::string base, size_t index) {
    return base + "_" + std::to_string(tpcSectors.at(index));
  };

  auto makeWriterSpec = [tpcSectors, laneConfiguration, getIndex, getName](const char* processName,
                                                                           const char* defaultFileName,
                                                                           const char* defaultTreeName,
                                                                           auto&& databranch,
                                                                           bool singleBranch = false) {
    if (tpcSectors.size() == 0) {
      throw std::invalid_argument(std::string("writer process configuration needs list of TPC sectors"));
    }

    auto amendInput = [tpcSectors, laneConfiguration](InputSpec& input, size_t index) {
      input.binding += std::to_string(laneConfiguration[index]);
      DataSpecUtils::updateMatchingSubspec(input, laneConfiguration[index]);
    };
    auto amendBranchDef = [laneConfiguration, amendInput, tpcSectors, getIndex, getName, singleBranch](auto&& def, bool enableMC = true) {
      if (!singleBranch) {
        def.keys = mergeInputs(def.keys, laneConfiguration.size(), amendInput);
        // the branch is disabled if set to 0
        def.nofBranches = enableMC ? tpcSectors.size() : 0;
        def.getIndex = getIndex;
        def.getName = getName;
      } else {
        // instead of the separate sector branches only one is going to be written
        def.nofBranches = enableMC ? 1 : 0;
      }
      return std::move(def);
    };

    return std::move(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                            std::move(amendBranchDef(databranch)))());
  };

  //////////////////////////////////////////////////////////////////////////////////////////////
  //
  // a writer process for digits
  //
  // selected by output type 'difits'
  using KrClusterOutputType = std::vector<o2::tpc::KrCluster>;
  workflow.push_back(makeWriterSpec("tpc-krcluster-writer",
                                    outputFile.data(),
                                    "Clusters",
                                    BranchDefinition<KrClusterOutputType>{InputSpec{"data", "TPC", "KRCLUSTERS", 0},
                                                                          "TPCBoxCluster",
                                                                          "boxcluster-branch-name"}));

  return workflow;
}
