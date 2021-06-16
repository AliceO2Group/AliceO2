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
#include <gsl/span>

#include "FairLogger.h"

#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALWorkflow/AnalysisClusterSpec.h"
#include "Framework/ControlService.h"
#include "EMCALBase/Geometry.h"
#include "DetectorsBase/GeometryManager.h"
#include <TGeoManager.h>

using namespace o2::emcal::reco_workflow;

template <class InputType>
void AnalysisClusterSpec<InputType>::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[EMCALClusterizer - init] Initialize clusterizer ...";

  // FIXME: Placeholder configuration -> get config from CCDB object
  double timeCut = 10000, timeMin = 0, timeMax = 10000, gradientCut = 0.03, thresholdSeedEnergy = 0.1, thresholdCellEnergy = 0.05;
  bool doEnergyGradientCut = true;

  // FIXME: Hardcoded for run II run
  // Get default geometry object if not yet set
  o2::base::GeometryManager::loadGeometry(); // for generating full clusters
  mGeometry = Geometry::GetInstanceFromRunNumber(223409);
  if (!mGeometry) {
    LOG(ERROR) << "Failure accessing geometry";
  }
  //gGeoManager->Import("/Users/hadi/Clusterizer/O2geometry.root");

  // Initialize clusterizer and link geometry
  mClusterizer.initialize(timeCut, timeMin, timeMax, gradientCut, doEnergyGradientCut, thresholdSeedEnergy, thresholdCellEnergy);
  mClusterizer.setGeometry(mGeometry);

  mEventHandler = new o2::emcal::EventHandler<InputType>();

  mClusterFactory = new o2::emcal::ClusterFactory<InputType>();

  mOutputAnaClusters = new std::vector<o2::emcal::AnalysisCluster>();
}

template <class InputType>
void AnalysisClusterSpec<InputType>::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALClusterizer - run] called";

  std::string inputname;
  std::string TrigName;

  if constexpr (std::is_same<InputType, o2::emcal::Digit>::value) {
    inputname = "digits";
    TrigName = "digitstrgr";
  } else if constexpr (std::is_same<InputType, o2::emcal::Cell>::value) {
    inputname = "cells";
    TrigName = "cellstrgr";
  }

  auto dataref = ctx.inputs().get(inputname.c_str());
  auto const* emcheader = o2::framework::DataRefUtils::getHeader<o2::emcal::EMCALBlockHeader*>(dataref);
  if (!emcheader->mHasPayload) {
    LOG(DEBUG) << "[EMCALClusterizer - run] No more cells/digits" << std::endl;
    ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
    return;
  }

  auto Inputs = ctx.inputs().get<gsl::span<InputType>>(inputname.c_str());
  LOG(DEBUG) << "[EMCALClusterizer - run]  Received " << Inputs.size() << " Cells/digits, running clusterizer ...";

  auto InputTriggerRecord = ctx.inputs().get<gsl::span<TriggerRecord>>(TrigName.c_str());
  LOG(DEBUG) << "[EMCALClusterizer - run]  Received " << InputTriggerRecord.size() << " Trigger Records, running clusterizer ...";

  mOutputAnaClusters->clear();

  std::vector<o2::emcal::Cluster> outputClusters;
  std::vector<int> outputCellDigitIndices;
  std::vector<o2::emcal::TriggerRecord> outputTriggerRecord;
  std::vector<o2::emcal::TriggerRecord> outputTriggerRecordIndices;

  int currentStartClusters = 0;
  int currentStartIndices = 0;

  for (auto iTrgRcrd : InputTriggerRecord) {

    mClusterizer.findClusters(gsl::span<const InputType>(&Inputs[iTrgRcrd.getFirstEntry()], iTrgRcrd.getNumberOfObjects())); // Find clusters on cells/digits (pass by ref)

    // Get found clusters + cell/digit indices for output
    // * A cluster contains a range that correspond to the vector of cell/digit indices
    // * The cell/digit index vector contains the indices of the clusterized cells/digits wrt to the original cell/digit array

    auto outputClustersTemp = mClusterizer.getFoundClusters();
    auto outputCellDigitIndicesTemp = mClusterizer.getFoundClustersInputIndices();

    std::copy(outputClustersTemp->begin(), outputClustersTemp->end(), std::back_inserter(outputClusters));
    std::copy(outputCellDigitIndicesTemp->begin(), outputCellDigitIndicesTemp->end(), std::back_inserter(outputCellDigitIndices));

    outputTriggerRecord.emplace_back(iTrgRcrd.getBCData(), currentStartClusters, outputClustersTemp->size());
    outputTriggerRecordIndices.emplace_back(iTrgRcrd.getBCData(), currentStartIndices, outputCellDigitIndicesTemp->size());

    currentStartClusters = outputClusters.size();
    currentStartIndices = outputCellDigitIndices.size();
  }

  mEventHandler->setClusterData(outputClusters, outputCellDigitIndices, outputTriggerRecord, outputTriggerRecordIndices);
  mEventHandler->setCellData(Inputs, InputTriggerRecord);

  //for (const auto& inputEvent : mEventHandler) {
  for (int iev = 0; iev < mEventHandler->getNumberOfEvents(); iev++) {
    auto inputEvent = mEventHandler->buildEvent(iev);

    mClusterFactory->reset();
    mClusterFactory->setClustersContainer(inputEvent.mClusters);
    mClusterFactory->setCellsContainer(Inputs);
    mClusterFactory->setCellsIndicesContainer(inputEvent.mCellIndices);

    //for (const auto& analysisCluster : mClusterFactory) {
    for (int icl = 0; icl < mClusterFactory->getNumberOfClusters(); icl++) {
      auto analysisCluster = mClusterFactory->buildCluster(icl);
      mOutputAnaClusters->push_back(analysisCluster);
    }
  }

  LOG(DEBUG) << "[EMCALClusterizer - run] Writing " << mOutputAnaClusters->size() << " clusters ...";
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "ANALYSISCLUSTERS", 0, o2::framework::Lifetime::Timeframe}, *mOutputAnaClusters);
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getAnalysisClusterSpec(bool useDigits)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;

  if (useDigits) {
    inputs.emplace_back("digits", o2::header::gDataOriginEMC, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
    inputs.emplace_back("digitstrgr", o2::header::gDataOriginEMC, "DIGITSTRGR", 0, o2::framework::Lifetime::Timeframe);
  } else {
    inputs.emplace_back("cells", o2::header::gDataOriginEMC, "CELLS", 0, o2::framework::Lifetime::Timeframe);
    inputs.emplace_back("cellstrgr", o2::header::gDataOriginEMC, "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe);
  }

  outputs.emplace_back(o2::header::gDataOriginEMC, "ANALYSISCLUSTERS", 0, o2::framework::Lifetime::Timeframe);

  if (useDigits) {
    return o2::framework::DataProcessorSpec{"EMCALAnalysisClusterSpec",
                                            inputs,
                                            outputs,
                                            o2::framework::adaptFromTask<o2::emcal::reco_workflow::AnalysisClusterSpec<o2::emcal::Digit>>()};
  } else {
    return o2::framework::DataProcessorSpec{"EMCALAnalysisClusterSpec",
                                            inputs,
                                            outputs,
                                            o2::framework::adaptFromTask<o2::emcal::reco_workflow::AnalysisClusterSpec<o2::emcal::Cell>>()};
  }
}
