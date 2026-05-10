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

/// @file   test_cmv-trigger.cxx
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Test workflow: reads CMVTRIGGER packets from tpc-flp-cmv and logs results

#include <vector>
#include <string>
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Headers/DataHeader.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCWorkflow/TPCFLPCMVSpec.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>&) {}

#include "Framework/runDataProcessing.h"

namespace o2::tpc
{

class CMVTriggerDevice : public o2::framework::Task
{
 public:
  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto tf = processing_helpers::getCurrentTF(pc);

    for (auto& ref : o2::framework::InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* hdr = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const uint32_t firstCRU = hdr->subSpecification >> 7;
      const bool triggered = pc.inputs().get<bool>(ref);
      if (triggered) {
        LOGP(info, "TF {:6} first CRU {:3}: {}", tf, firstCRU, "triggered");
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
  }

 private:
  const std::vector<o2::framework::InputSpec> mFilter = {
    {"cmvtrigger", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, o2::tpc::TPCFLPCMVDevice::getDataDescriptionCMVTrigger()}, o2::framework::Lifetime::Timeframe}};
};

o2::framework::DataProcessorSpec getCMVTriggerSpec()
{
  std::vector<o2::framework::InputSpec> inputSpecs;
  inputSpecs.emplace_back(o2::framework::InputSpec{"cmvtrigger", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, o2::tpc::TPCFLPCMVDevice::getDataDescriptionCMVTrigger()}, o2::framework::Lifetime::Timeframe});

  return o2::framework::DataProcessorSpec{
    "tpc-cmv-trigger",
    inputSpecs,
    {},
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<CMVTriggerDevice>()}};
}

} // namespace o2::tpc

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  workflow.emplace_back(o2::tpc::getCMVTriggerSpec());
  return workflow;
}
