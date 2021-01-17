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

#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/Cluster.h"
#include "DataFormatsCPV/CPVBlockHeader.h"
#include "CPVWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"

using namespace o2::cpv::reco_workflow;

void ClusterizerSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[CPVClusterizer - init] Initialize clusterizer ...";

  // Initialize clusterizer and link geometry
  mClusterizer.initialize();
  mClusterizer.propagateMC(mPropagateMC);
}

void ClusterizerSpec::run(framework::ProcessingContext& ctx)
{
  LOG(INFO) << "Start run ";
  LOG(DEBUG) << "CPVClusterizer - run on digits called";
  auto digits = ctx.inputs().get<gsl::span<Digit>>("digits");
  // auto digitsTR = ctx.inputs().get<std::span<TriggerRecord>>("digitTriggerRecords"); //TODO:: Why span does not work???
  // auto digits = ctx.inputs().get<std::vector<o2::cpv::Digit>>("digits");
  auto digitsTR = ctx.inputs().get<std::vector<o2::cpv::TriggerRecord>>("digitTriggerRecords");

  // printf("CluSpec: digits=%d, TR=%d \n",digits.size(),digitsTR.size()) ;

  LOG(DEBUG) << "[CPVClusterizer - run]  Received " << digitsTR.size() << " TR, running clusterizer ...";
  std::unique_ptr<const o2::dataformats::MCTruthContainer<MCCompLabel>> truthcont;
  if (mPropagateMC) {
    truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("digitsmctr");
  }

  mClusterizer.process(digits, digitsTR, *truthcont, &mOutputClusters, &mOutputClusterTrigRecs, &mOutputTruthCont); // Find clusters on digits (pass by ref)

  ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusters);
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERTRIGRECS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusterTrigRecs);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
  }
  LOG(INFO) << "Finished, wrote  " << mOutputClusters.size() << " clusters, " << mOutputClusterTrigRecs.size() << "TR and " << mOutputTruthCont.getIndexedSize() << " Labels";
  ctx.services().get<o2::framework::ControlService>().endOfStream();
  ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
}
o2::framework::DataProcessorSpec o2::cpv::reco_workflow::getClusterizerSpec(bool propagateMC)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginCPV, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("digitTriggerRecords", o2::header::gDataOriginCPV, "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("digitsmctr", "CPV", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
  }
  outputs.emplace_back("CPV", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "CLUSTERTRIGRECS", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    outputs.emplace_back("CPV", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{"CPVClusterizerSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::cpv::reco_workflow::ClusterizerSpec>(propagateMC)};
}
