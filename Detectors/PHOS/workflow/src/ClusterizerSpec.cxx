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

#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "PHOSWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"

using namespace o2::phos::reco_workflow;

void ClusterizerSpec::init(framework::InitContext& ctx)
{
  LOG(debug) << "[PHOSClusterizer - init] Initialize clusterizer ...";

  // get BadMap and calibration CCDB

  mClusterizer.initialize();
  auto localccdb = ctx.options().get<std::string>("testBadMap");

  if (localccdb == "localtest") {
    // create test BadMap and Calib objects. ClusterizerSpec should be owner
    mCalibParams = std::make_unique<CalibParams>(1); // Create test calibration coefficients
    mBadMap = std::make_unique<BadChannelsMap>(1);   // Create test bad map
    mClusterizer.setBadMap(mBadMap.get());
    mClusterizer.setCalibration(mCalibParams.get()); // test calibration map
    LOG(info) << "No reading BadMap/Calibration from ccdb requested, set default";
  } else {
    // Normally CCDB manager should get and own objects
    auto& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
    ccdbManager.setURL(o2::base::NameConf::getCCDBServer());
    LOG(info) << " set-up CCDB " << o2::base::NameConf::getCCDBServer();

    BadChannelsMap* badMap = ccdbManager.get<o2::phos::BadChannelsMap>("PHS/Calib/BadMap");
    CalibParams* calibParams = ccdbManager.get<o2::phos::CalibParams>("PHS/Calib/CalibParams");
    if (badMap) {
      mClusterizer.setBadMap(badMap);
    } else {
      LOG(fatal) << "[PHOSCellConverter - run] can not get Bad Map";
    }
    if (calibParams) {
      mClusterizer.setCalibration(calibParams);
    } else {
      LOG(fatal) << "[PHOSCellConverter - run] can not get CalibParams";
    }
  }
}

void ClusterizerSpec::run(framework::ProcessingContext& ctx)
{

  if (mUseDigits) {
    LOG(debug) << "PHOSClusterizer - run on digits called";

    auto dataref = ctx.inputs().get("digits");
    auto const* phosheader = o2::framework::DataRefUtils::getHeader<o2::phos::PHOSBlockHeader*>(dataref);
    if (!phosheader->mHasPayload) {
      mOutputClusters.clear();
      ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusters);
      if (mFullCluOutput) {
        mOutputCluElements.clear();
        ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe}, mOutputCluElements);
      }
      mOutputClusterTrigRecs.clear();
      ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputClusterTrigRecs);
      if (mPropagateMC) {
        mOutputTruthCont.clear();
        ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
      }
      return;
    }
    // auto digits = ctx.inputs().get<gsl::span<o2::phos::Digit>>("digits");
    auto digits = ctx.inputs().get<std::vector<o2::phos::Digit>>("digits");
    auto digitsTR = ctx.inputs().get<std::vector<o2::phos::TriggerRecord>>("digitTriggerRecords");
    LOG(debug) << "[PHOSClusterizer - run]  Received " << digitsTR.size() << " TR, running clusterizer ...";
    // const o2::dataformats::MCTruthContainer<MCLabel>* truthcont=nullptr;
    if (mPropagateMC) {
      std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>> truthcont(ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>*>("digitsmctr"));
      mClusterizer.process(digits, digitsTR, truthcont.get(), mOutputClusters, mOutputCluElements, mOutputClusterTrigRecs, mOutputTruthCont); // Find clusters on digits (pass by ref)
    } else {
      mClusterizer.process(digits, digitsTR, nullptr, mOutputClusters, mOutputCluElements, mOutputClusterTrigRecs, mOutputTruthCont); // Find clusters on digits (pass by ref)
    }
  } else {

    LOG(debug) << "PHOSClusterizer - run run on cells called";

    auto cells = ctx.inputs().get<std::vector<o2::phos::Cell>>("cells");
    // auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
    LOG(debug) << "[PHOSClusterizer - run]  Received " << cells.size() << " cells, running clusterizer ...";
    // auto cellsTR = ctx.inputs().get<gsl::span<o2::phos::TriggerRecord>>("cellTriggerRecords");
    auto cellsTR = ctx.inputs().get<std::vector<o2::phos::TriggerRecord>>("cellTriggerRecords");
    if (mPropagateMC) {
      std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>> truthcont(ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>*>("cellsmctr"));
      // truthmap = ctx.inputs().get<gsl::span<uint>>("cellssmcmap");
      mClusterizer.processCells(cells, cellsTR, truthcont.get(), mOutputClusters, mOutputCluElements, mOutputClusterTrigRecs, mOutputTruthCont); // Find clusters on digits (pass by ref)
    } else {
      mClusterizer.processCells(cells, cellsTR, nullptr, mOutputClusters, mOutputCluElements, mOutputClusterTrigRecs, mOutputTruthCont); // Find clusters on digits (pass by ref)
    }
  }

  if (mPropagateMC) {
    LOG(debug) << "[PHOSClusterizer - run] Writing " << mOutputClusters.size() << " clusters, " << mOutputClusterTrigRecs.size() << "TR and " << mOutputTruthCont.getIndexedSize() << " Labels";
  } else {
    LOG(debug) << "[PHOSClusterizer - run] Writing " << mOutputClusters.size() << " clusters and " << mOutputClusterTrigRecs.size() << " TR";
  }
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusters);
  if (mFullCluOutput) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe}, mOutputCluElements);
  }
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputClusterTrigRecs);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
  }
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getClusterizerSpec(bool propagateMC, bool fullClu)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginPHS, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("digitTriggerRecords", o2::header::gDataOriginPHS, "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("digitsmctr", "PHS", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
  }
  outputs.emplace_back("PHS", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  if (fullClu) {
    outputs.emplace_back("PHS", "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe);
  }
  outputs.emplace_back("PHS", "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    outputs.emplace_back("PHS", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{"PHOSClusterizerSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::ClusterizerSpec>(propagateMC, true, fullClu),
                                          o2::framework::Options{
                                            {"testBadMap", o2::framework::VariantType::String, "localtest", {"use test bad map and calib objects"}}}};
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getCellClusterizerSpec(bool propagateMC, bool fullClu)
{
  // Cluaterizer with cell input
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("cells", o2::header::gDataOriginPHS, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", o2::header::gDataOriginPHS, "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("cellsmctr", "PHS", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe);
  }
  outputs.emplace_back("PHS", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  if (fullClu) {
    outputs.emplace_back("PHS", "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe);
  }
  outputs.emplace_back("PHS", "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    outputs.emplace_back("PHS", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{"PHOSClusterizerSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::ClusterizerSpec>(propagateMC, false, fullClu),
                                          o2::framework::Options{
                                            {"testBadMap", o2::framework::VariantType::String, "localtest", {"use test bad map and calib objects"}}}};
}
