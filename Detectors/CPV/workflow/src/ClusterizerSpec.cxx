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
#include "FairLogger.h"

#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/Cluster.h"
#include "DataFormatsCPV/CPVBlockHeader.h"
#include "CPVWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"
#include "CPVBase/CPVSimParams.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::cpv::reco_workflow;

void ClusterizerSpec::init(framework::InitContext& ctx)
{
  LOG(debug) << "[CPVClusterizer - init] Initialize clusterizer ...";

  // Initialize clusterizer and link geometry
  mClusterizer.initialize();
  mClusterizer.propagateMC(mPropagateMC);
}

void ClusterizerSpec::run(framework::ProcessingContext& ctx)
{
  LOG(info) << "Starting ClusterizerSpec::run() ";
  LOG(debug) << "CPVClusterizer - run on digits called";

  // update config
  static bool isConfigFetched = false;
  if (!isConfigFetched) {
    LOG(info) << "ClusterizerSpec::run() : fetching o2::cpv::CPVSimParams from CCDB";
    ctx.inputs().get<o2::cpv::CPVSimParams*>("simparams");
    LOG(info) << "ClusterizerSpec::run() : o2::cpv::CPVSimParams::Instance() now is following:";
    o2::cpv::CPVSimParams::Instance().printKeyValues();
    isConfigFetched = true;
  }

  auto digits = ctx.inputs().get<std::vector<Digit>>("digits");

  if (!digits.size()) { // nothing to process
    LOG(info) << "ClusterizerSpec::run() : no digits; moving on";
    mOutputClusters.clear();
    ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusters);
    mOutputClusterTrigRecs.clear();
    ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERTRIGRECS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusterTrigRecs);
    mCalibDigits.clear();
    ctx.outputs().snapshot(o2::framework::Output{"CPV", "CALIBDIGITS", 0, o2::framework::Lifetime::Timeframe}, mCalibDigits);
    if (mPropagateMC) {
      mOutputTruthCont.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
    }
    return;
  }
  auto digitsTR = ctx.inputs().get<std::vector<o2::cpv::TriggerRecord>>("digitTriggerRecords");

  // const o2::dataformats::MCTruthContainer<MCCompLabel>* truthcont = nullptr;
  //  DO NOT TRY TO USE const pointer for MCTruthContainer, it is somehow spoiling whole array
  if (mPropagateMC) {
    auto truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("digitsmctr");
    mClusterizer.process(digits, digitsTR, truthcont.get(), &mOutputClusters, &mOutputClusterTrigRecs, &mOutputTruthCont, &mCalibDigits); // Find clusters with MC Truth
  } else {
    mClusterizer.process(digits, digitsTR, nullptr, &mOutputClusters, &mOutputClusterTrigRecs, &mOutputTruthCont, &mCalibDigits); // Find clusters without MC Truth
  }

  LOG(debug) << "CPVClusterizer::run() : Received " << digitsTR.size() << " TR, calling clusterizer ...";

  ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusters);
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERTRIGRECS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusterTrigRecs);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"CPV", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
  }
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "CALIBDIGITS", 0, o2::framework::Lifetime::Timeframe}, mCalibDigits);
  LOG(info) << "Finished, wrote  " << mOutputClusters.size() << " clusters, " << mOutputClusterTrigRecs.size() << "TR and " << mOutputTruthCont.getIndexedSize() << " Labels";
}
o2::framework::DataProcessorSpec o2::cpv::reco_workflow::getClusterizerSpec(bool propagateMC)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("simparams", "CPV", "CPV_SimPars", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CPV/Config/CPVSimParams"));
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
  outputs.emplace_back("CPV", "CALIBDIGITS", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"CPVClusterizerSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::cpv::reco_workflow::ClusterizerSpec>(propagateMC)};
}
