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

  //get BadMap and calibration CCDB
  std::unique_ptr<CalibParams> calibParams; ///! Calibration coefficients
  std::unique_ptr<BadChannelsMap> badMap;   ///! Bad map

  // if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
  badMap.reset(new BadChannelsMap(1));   // test default map
  calibParams.reset(new CalibParams(1)); //test calibration map
  LOG(INFO) << "No reading BadMap/Calibration from ccdb requested, set default";
  // } else {
  //   LOG(INFO) << "Getting BadMap object from ccdb";
  //   o2::ccdb::CcdbApi ccdb;
  //   std::map<std::string, std::string> metadata; // do we want to store any meta data?
  //   ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
  //   long bcTime = 1;                             //TODO!!! Convert BC time to time o2::InteractionRecord bcTime = digitsTR.front().getBCData() ;
  // mBadMap = ccdb.retrieveFromTFileAny<o2::phos::BadChannelsMap>("PHOS/BadMap", metadata, bcTime);
  // mCalibParams = ccdb.retrieveFromTFileAny<o2::phos::CalibParams>("PHOS/Calib", metadata, bcTime);
  // if (!mBadMap) {
  //   LOG(FATAL) << "[PHOSCellConverter - run] can not get Bad Map";
  // }
  // if (!mCalibParams) {
  //   LOG(FATAL) << "[PHOSCellConverter - run] can not get CalibParams";
  // }
  // }

  mClusterizer.initialize();
  mClusterizer.setBadMap(badMap);
  mClusterizer.setCalibration(calibParams);
}

void ClusterizerSpec::run(framework::ProcessingContext& ctx)
{

  if (mUseDigits) {
    LOG(DEBUG) << "PHOSClusterizer - run on digits called";

    auto dataref = ctx.inputs().get("digits");
    auto const* phosheader = o2::framework::DataRefUtils::getHeader<o2::phos::PHOSBlockHeader*>(dataref);
    if (!phosheader->mHasPayload) {
      LOG(DEBUG) << "[PHOSClusterizer - run] No more digits" << std::endl;
      ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
      return;
    }

    // auto digits = ctx.inputs().get<gsl::span<o2::phos::Digit>>("digits");
    auto digits = ctx.inputs().get<std::vector<o2::phos::Digit>>("digits");
    auto digitsTR = ctx.inputs().get<std::vector<o2::phos::TriggerRecord>>("digitTriggerRecords");
    LOG(DEBUG) << "[PHOSClusterizer - run]  Received " << digitsTR.size() << " TR, running clusterizer ...";
    // const o2::dataformats::MCTruthContainer<MCLabel>* truthcont=nullptr;
    if (mPropagateMC) {
      std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>> truthcont(ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>*>("digitsmctr"));
      mClusterizer.process(digits, digitsTR, truthcont.get(), mOutputClusters, mOutputCluElements, mOutputClusterTrigRecs, mOutputTruthCont); // Find clusters on digits (pass by ref)
    } else {
      mClusterizer.process(digits, digitsTR, nullptr, mOutputClusters, mOutputCluElements, mOutputClusterTrigRecs, mOutputTruthCont); // Find clusters on digits (pass by ref)
    }
  } else {

    LOG(DEBUG) << "PHOSClusterizer - run run on cells called";

    auto cells = ctx.inputs().get<std::vector<o2::phos::Cell>>("cells");
    // auto cells = ctx.inputs().get<gsl::span<o2::phos::Cell>>("cells");
    LOG(DEBUG) << "[PHOSClusterizer - run]  Received " << cells.size() << " cells, running clusterizer ...";
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
    LOG(DEBUG) << "[PHOSClusterizer - run] Writing " << mOutputClusters.size() << " clusters, " << mOutputClusterTrigRecs.size() << "TR and " << mOutputTruthCont.getIndexedSize() << " Labels";
  } else {
    LOG(DEBUG) << "[PHOSClusterizer - run] Writing " << mOutputClusters.size() << " clusters and " << mOutputClusterTrigRecs.size() << " TR";
  }
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, mOutputClusters);
  if (mFullCluOutput) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe}, mOutputCluElements);
  }
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputClusterTrigRecs);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CLUSTERTRUEMC", 0, o2::framework::Lifetime::Timeframe}, mOutputTruthCont);
  }
  ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
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
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::ClusterizerSpec>(propagateMC, true, fullClu)};
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getCellClusterizerSpec(bool propagateMC, bool fullClu)
{
  //Cluaterizer with cell input
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
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::ClusterizerSpec>(propagateMC, false, fullClu)};
}
