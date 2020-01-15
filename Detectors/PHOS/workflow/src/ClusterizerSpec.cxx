// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FairLogger.h"

#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "PHOSWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"

using namespace o2::phos::reco_workflow;

void ClusterizerSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[PHOSClusterizer - init] Initialize clusterizer ...";

  // Initialize clusterizer and link geometry
  mClusterizer.initialize();
}

void ClusterizerSpec::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[PHOSClusterizer - run] called";
  auto dataref = ctx.inputs().get("digits");
  auto const* phosheader = o2::framework::DataRefUtils::getHeader<o2::phos::PHOSBlockHeader*>(dataref);
  if (!phosheader->mHasPayload) {
    LOG(DEBUG) << "[PHOSClusterizer - run] No more digits" << std::endl;
    ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
    return;
  }

  auto digits = ctx.inputs().get<std::vector<o2::phos::Digit>>("digits");
  LOG(DEBUG) << "[PHOSClusterizer - run]  Received " << digits.size() << " digits, running clusterizer ...";
  auto digitsTR = ctx.inputs().get<std::vector<o2::phos::TriggerRecord>>("digitTriggerRecords");
  auto truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>*>("digitsmctr");

  mClusterizer.process(digits, digitsTR, truthcont.get(), &mOutputClusters, &mOutputClusterTrigRecs, &mOutputTruthCont); // Find clusters on digits (pass by ref)

  LOG(DEBUG) << "[PHOSClusterizer - run] Writing " << mOutputClusters.size() << " clusters, " << mOutputClusterTrigRecs.size() << "TR and " << mOutputTruthCont.getIndexedSize() << " Labels";
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusters);
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRIGRECS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusterTrigRecs);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
  }
  LOG(INFO) << "Finished ";
  ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getClusterizerSpec(bool propagateMC)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginPHS, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("digitTriggerRecords", o2::header::gDataOriginPHS, "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("digitsmctr", "PHS", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
  }
  outputs.emplace_back("PHS", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "CLUSTERTRIGRECS", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    outputs.emplace_back("PHS", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{"PHOSClusterizerSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::ClusterizerSpec>(propagateMC)};
}
