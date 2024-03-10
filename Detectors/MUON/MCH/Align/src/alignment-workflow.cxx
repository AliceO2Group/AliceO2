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

/// \file alignment-workflow.cxx
/// \brief Implementation of a DPL device to perform alignment for muon spectrometer
///
/// \author Chi ZHANG, CEA-Saclay

#include "MCHAlign/AlignmentSpec.h"

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtils.h"
#include "Headers/STFHeader.h"
#include "Framework/CallbackService.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

using namespace o2::framework;
using namespace std;

namespace o2::mch
{

class SeederTask : public Task
{
 public:
  void run(ProcessingContext& pc) final
  {
    const auto& hbfu = o2::raw::HBFUtils::Instance();
    auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    if (hbfu.startTime != 0) {
      tinfo.creation = hbfu.startTime;
    } else {
      tinfo.creation = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
    }
    if (hbfu.orbitFirstSampled != 0) {
      tinfo.firstTForbit = hbfu.orbitFirstSampled;
    } else {
      tinfo.firstTForbit = 0;
    }
    auto& stfDist = pc.outputs().make<o2::header::STFHeader>(Output{"FLP", "DISTSUBTIMEFRAME", 0});
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
};

} // namespace o2::mch

o2::framework::DataProcessorSpec getSeederSpec()
{
  return DataProcessorSpec{
    "seeder",
    Inputs{},
    Outputs{{"FLP", "DISTSUBTIMEFRAME", 0}},
    AlgorithmSpec{o2::framework::adaptFromTask<o2::mch::SeederTask>()},
    Options{}};
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back("configKeyValues", VariantType::String, "",
                               ConfigParamSpec::HelpString{"Semicolon separated key=value strings"});
  workflowOptions.emplace_back("disable-input-from-ccdb", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"Do not read magnetic field and geometry from CCDB"});
}

#include "Framework/runDataProcessing.h"
WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  bool disableCCDB = configcontext.options().get<bool>("disable-input-from-ccdb");
  return WorkflowSpec{o2::mch::getAlignmentSpec(disableCCDB), getSeederSpec()};
}
