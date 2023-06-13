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
#include <fairlogger/Logger.h>

#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "PHOSBase/PHOSSimParams.h"
#include "PHOSWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"

#include "Framework/DataRefUtils.h"

using namespace o2::phos::reco_workflow;

void ClusterizerSpec::init(framework::InitContext& ctx)
{
  LOG(debug) << "[PHOSClusterizer - init] Initialize clusterizer ...";

  // get BadMap and calibration CCDB

  mClusterizer.initialize();
  if (mDefBadMap) {
    LOG(info) << "No reading BadMap/Calibration from ccdb requested, set default";
    // create test BadMap and Calib objects. ClusterizerSpec should be owner
    mCalibParams = std::make_unique<CalibParams>(1); // Create test calibration coefficients
    mBadMap = std::make_unique<BadChannelsMap>();    // Create empty bad map
    mClusterizer.setBadMap(mBadMap.get());
    mClusterizer.setCalibration(mCalibParams.get()); // test calibration map
    mHasCalib = true;
  }
}

void ClusterizerSpec::run(framework::ProcessingContext& ctx)
{

  // Do not use ccdb if localtest
  if (!mHasCalib) { // Default map and calibration was not set, use CCDB
    // update BadMap and calibration if necessary
    std::decay_t<decltype(ctx.inputs().get<o2::phos::BadChannelsMap*>("badmap"))> badMapPtr{};
    badMapPtr = ctx.inputs().get<o2::phos::BadChannelsMap*>("badmap");
    mClusterizer.setBadMap(badMapPtr.get());

    std::decay_t<decltype(ctx.inputs().get<o2::phos::CalibParams*>("calib"))> calibPtr{};
    calibPtr = ctx.inputs().get<o2::phos::CalibParams*>("calib");
    mClusterizer.setCalibration(calibPtr.get());

    if (!mSkipL1phase && !mPropagateMC) {
      auto vec = ctx.inputs().get<std::vector<int>*>("l1phase");
      mClusterizer.setL1phase((*vec)[0]);
    }

    mHasCalib = true;
  }
  if (mInitSimParams) { // trigger reading sim/rec parameters from CCDB, singleton initiated in Fetcher
    ctx.inputs().get<o2::phos::PHOSSimParams*>("recoparams");
    mInitSimParams = false;
  }

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
    auto cells = ctx.inputs().get<std::vector<o2::phos::Cell>>("cells");
    //     auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
    LOG(debug) << "[PHOSClusterizer - run]  Received " << cells.size() << " cells, running clusterizer ...";
    //     auto cellsTR = ctx.inputs().get<gsl::span<o2::phos::TriggerRecord>>("cellTriggerRecords");
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

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getClusterizerSpec(bool propagateMC, bool fullClu, bool defBadMap)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginPHS, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("digitTriggerRecords", o2::header::gDataOriginPHS, "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (!defBadMap) {
    inputs.emplace_back("badmap", o2::header::gDataOriginPHS, "PHS_BadMap", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/BadMap"));
    inputs.emplace_back("calib", o2::header::gDataOriginPHS, "PHS_Calib", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/CalibParams"));
  }
  inputs.emplace_back("recoparams", o2::header::gDataOriginPHS, "PHS_RecoParams", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Config/RecoParams"));
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
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::ClusterizerSpec>(propagateMC, true, fullClu, defBadMap, true),
                                          o2::framework::Options{}};
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getCellClusterizerSpec(bool propagateMC, bool fullClu, bool defBadMap, bool skipL1phase)
{
  // Cluaterizer with cell input
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("cells", o2::header::gDataOriginPHS, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", o2::header::gDataOriginPHS, "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (!defBadMap) {
    inputs.emplace_back("badmap", o2::header::gDataOriginPHS, "PHS_BadMap", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/BadMap"));
    inputs.emplace_back("calib", o2::header::gDataOriginPHS, "PHS_Calib", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/CalibParams"));
    if (!skipL1phase && !propagateMC) {
      inputs.emplace_back("l1phase", o2::header::gDataOriginPHS, "PHS_L1phase", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/L1phase"));
    }
  }
  inputs.emplace_back("recoparams", o2::header::gDataOriginPHS, "PHS_RecoParams", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Config/RecoParams"));
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
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::ClusterizerSpec>(propagateMC, false, fullClu, defBadMap, skipL1phase),
                                          o2::framework::Options{}};
}
