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

#include <InfoLogger/InfoLogger.hxx>

#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"

using namespace o2::emcal::reco_workflow;

template <class InputType>
void ClusterizerSpec<InputType>::init(framework::InitContext& ctx)
{
  // Check if InfoLoggerContext is active and if so set the Detector field
  if (ctx.services().active<AliceO2::InfoLogger::InfoLoggerContext>()) {
    auto& ilctx = ctx.services().get<AliceO2::InfoLogger::InfoLoggerContext>();
    ilctx.setField(AliceO2::InfoLogger::InfoLoggerContext::FieldName::Detector, "EMC");
  }

  LOG(debug) << "[EMCALClusterizer - init] Initialize clusterizer ...";

  // FIXME: Placeholder configuration -> get config from CCDB object
  double timeCut = 10000, timeMin = 0, timeMax = 10000, gradientCut = 0.03, thresholdSeedEnergy = 0.1, thresholdCellEnergy = 0.05;
  bool doEnergyGradientCut = true;

  // FIXME: Hardcoded for run II run
  // Get default geometry object if not yet set
  mGeometry = Geometry::GetInstanceFromRunNumber(223409);
  if (!mGeometry) {
    LOG(error) << "Failure accessing geometry";
  }

  // Initialize clusterizer and link geometry
  mClusterizer.initialize(timeCut, timeMin, timeMax, gradientCut, doEnergyGradientCut, thresholdSeedEnergy, thresholdCellEnergy);
  mClusterizer.setGeometry(mGeometry);

  mOutputClusters = new std::vector<o2::emcal::Cluster>();
  mOutputCellDigitIndices = new std::vector<o2::emcal::ClusterIndex>();
  mOutputTriggerRecord = new std::vector<o2::emcal::TriggerRecord>();
  mOutputTriggerRecordIndices = new std::vector<o2::emcal::TriggerRecord>();
  mTimer.Stop();
  mTimer.Reset();
}

template <class InputType>
void ClusterizerSpec<InputType>::run(framework::ProcessingContext& ctx)
{
  LOG(debug) << "[EMCALClusterizer - run] called";
  mTimer.Start(false);
  std::string inputname;
  std::string TrigName;

  if constexpr (std::is_same<InputType, o2::emcal::Digit>::value) {
    inputname = "digits";
    TrigName = "digitstrgr";
  } else if constexpr (std::is_same<InputType, o2::emcal::Cell>::value) {
    inputname = "cells";
    TrigName = "cellstrgr";
  }

  auto Inputs = ctx.inputs().get<gsl::span<InputType>>(inputname.c_str());
  LOG(debug) << "[EMCALClusterizer - run]  Received " << Inputs.size() << " Cells/digits, running clusterizer ...";

  auto InputTriggerRecord = ctx.inputs().get<gsl::span<TriggerRecord>>(TrigName.c_str());
  LOG(debug) << "[EMCALClusterizer - run]  Received " << InputTriggerRecord.size() << " Trigger Records, running clusterizer ...";

  mOutputClusters->clear();
  mOutputCellDigitIndices->clear();
  mOutputTriggerRecord->clear();
  mOutputTriggerRecordIndices->clear();

  int currentStartClusters = mOutputClusters->size();
  int currentStartIndices = mOutputCellDigitIndices->size();
  for (auto iTrgRcrd : InputTriggerRecord) {
    if (Inputs.size() && iTrgRcrd.getNumberOfObjects()) {
      mClusterizer.findClusters(gsl::span<const InputType>(&Inputs[iTrgRcrd.getFirstEntry()], iTrgRcrd.getNumberOfObjects())); // Find clusters on cells/digits (pass by ref)
    } else {
      mClusterizer.clear();
    }
    // Get found clusters + cell/digit indices for output
    // * A cluster contains a range that correspond to the vector of cell/digit indices
    // * The cell/digit index vector contains the indices of the clusterized cells/digits wrt to the original cell/digit array

    auto outputClustersTemp = mClusterizer.getFoundClusters();
    auto outputCellDigitIndicesTemp = mClusterizer.getFoundClustersInputIndices();

    std::copy(outputClustersTemp->begin(), outputClustersTemp->end(), std::back_inserter(*mOutputClusters));
    std::copy(outputCellDigitIndicesTemp->begin(), outputCellDigitIndicesTemp->end(), std::back_inserter(*mOutputCellDigitIndices));

    mOutputTriggerRecord->emplace_back(iTrgRcrd.getBCData(), currentStartClusters, outputClustersTemp->size());
    mOutputTriggerRecordIndices->emplace_back(iTrgRcrd.getBCData(), currentStartIndices, outputCellDigitIndicesTemp->size());

    currentStartClusters = mOutputClusters->size();
    currentStartIndices = mOutputCellDigitIndices->size();
  }
  LOG(debug) << "[EMCALClusterizer - run] Writing " << mOutputClusters->size() << " clusters ...";
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, *mOutputClusters);
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "INDICES", 0, o2::framework::Lifetime::Timeframe}, *mOutputCellDigitIndices);

  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CLUSTERSTRGR", 0, o2::framework::Lifetime::Timeframe}, *mOutputTriggerRecord);
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "INDICESTRGR", 0, o2::framework::Lifetime::Timeframe}, *mOutputTriggerRecordIndices);
  mTimer.Stop();
}

template <class InputType>
void ClusterizerSpec<InputType>::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  LOG(info) << "EMCALClusterizer timing: CPU: " << mTimer.CpuTime() << " Real: " << mTimer.RealTime() << " in " << mTimer.Counter() - 1 << " TFs";
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getClusterizerSpec(bool useDigits)
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

  outputs.emplace_back(o2::header::gDataOriginEMC, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginEMC, "INDICES", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginEMC, "CLUSTERSTRGR", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginEMC, "INDICESTRGR", 0, o2::framework::Lifetime::Timeframe);

  if (useDigits) {
    return o2::framework::DataProcessorSpec{"EMCALClusterizerSpec",
                                            inputs,
                                            outputs,
                                            o2::framework::adaptFromTask<o2::emcal::reco_workflow::ClusterizerSpec<o2::emcal::Digit>>()};
  } else {
    return o2::framework::DataProcessorSpec{"EMCALClusterizerSpec",
                                            inputs,
                                            outputs,
                                            o2::framework::adaptFromTask<o2::emcal::reco_workflow::ClusterizerSpec<o2::emcal::Cell>>()};
  }
}
