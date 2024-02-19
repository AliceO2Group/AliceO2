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

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"detector", VariantType::String, "ITS", {"detector name"}}};

  std::swap(workflowOptions, options);
}
// ------------------------------------------------------------------

namespace o2
{
namespace dcs
{
class DCSConfigConsumer : public o2::framework::Task
{
 public:
  void run(o2::framework::ProcessingContext& pc) final
  {
    auto fileBuff = pc.inputs().get<gsl::span<char>>("confFile");
    auto fileName = pc.inputs().get<std::string>("confFileName");
    LOG(info) << "got input file " << fileName << " of size " << fileBuff.size();
  }
};
} // namespace dcs
} // namespace o2

DataProcessorSpec getDCSConsumerSpec(DetID det)
{
  std::string procName = "dcs-config-consumer-";
  procName += det.getName();
  return DataProcessorSpec{
    procName,
    Inputs{{"confFile", ConcreteDataTypeMatcher{det.getDataOrigin(), "DCS_CONFIG_FILE"}, Lifetime::Sporadic},
           {"confFileName", ConcreteDataTypeMatcher{det.getDataOrigin(), "DCS_CONFIG_NAME"}, Lifetime::Sporadic}},
    Outputs{},
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSConfigConsumer>()},
    Options{}};
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto detName = configcontext.options().get<std::string>("detector");
  auto detID = DetID::nameToID(detName.c_str(), DetID::First);
  if (detID < 0) {
    throw std::runtime_error(fmt::format("{} is not a valid detector name", detName));
  }
  specs.emplace_back(getDCSConsumerSpec({detID}));

  return specs;
}
