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

#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "EMCALWorkflow/ClusterizerSpec.h"
#include "Framework/ControlService.h"

using namespace o2::emcal::reco_workflow;

template <class InputType>
void ClusterizerSpec<InputType>::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[EMCALClusterizer - init] Initialize clusterizer ...";

  // FIXME: Placeholder configuration -> get config from CCDB object
  double timeCut = 10000, timeMin = 0, timeMax = 10000, gradientCut = 0.03, thresholdSeedEnergy = 0.1, thresholdCellEnergy = 0.05;
  bool doEnergyGradientCut = true;

  // FIXME: Hardcoded for run II run
  // Get default geometry object if not yet set
  mGeometry = Geometry::GetInstanceFromRunNumber(223409);
  if (!mGeometry) {
    LOG(ERROR) << "Failure accessing geometry";
  }

  // Initialize clusterizer and link geometry
  mClusterizer.initialize(timeCut, timeMin, timeMax, gradientCut, doEnergyGradientCut, thresholdSeedEnergy, thresholdCellEnergy);
  mClusterizer.setGeometry(mGeometry);
}

template <class InputType>
void ClusterizerSpec<InputType>::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALClusterizer - run] called";

  std::string inputname;

  if constexpr (std::is_same<InputType, o2::emcal::Digit>::value) {
    inputname = "digits";
  } else if constexpr (std::is_same<InputType, o2::emcal::Cell>::value) {
    inputname = "cells";
  }

  auto dataref = ctx.inputs().get(inputname.c_str());
  auto const* emcheader = o2::framework::DataRefUtils::getHeader<o2::emcal::EMCALBlockHeader*>(dataref);
  if (!emcheader->mHasPayload) {
    LOG(DEBUG) << "[EMCALClusterizer - run] No more cells/digits" << std::endl;
    ctx.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
    return;
  }

  auto Inputs = ctx.inputs().get<std::vector<InputType>>(inputname.c_str());
  LOG(DEBUG) << "[EMCALClusterizer - run]  Received " << Inputs.size() << " Cells/digits, running clusterizer ...";

  mClusterizer.findClusters(Inputs); // Find clusters on cells/digits (pass by ref)

  // Get found clusters + cell/digit indices for output
  // * A cluster contains a range that correspond to the vector of cell/digit indices
  // * The cell/digit index vector contains the indices of the clusterized cells/digits wrt to the original cell/digit array
  mOutputClusters = mClusterizer.getFoundClusters();
  mOutputCellDigitIndices = mClusterizer.getFoundClustersInputIndices();

  LOG(DEBUG) << "[EMCALClusterizer - run] Writing " << mOutputClusters->size() << " clusters ...";
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, *mOutputClusters);
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "INDICES", 0, o2::framework::Lifetime::Timeframe}, *mOutputCellDigitIndices);
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getClusterizerSpec(bool useDigits)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;

  if (useDigits) {
    inputs.emplace_back("digits", o2::header::gDataOriginEMC, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  } else {
    inputs.emplace_back("cells", o2::header::gDataOriginEMC, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  }

  outputs.emplace_back(o2::header::gDataOriginEMC, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginEMC, "INDICES", 0, o2::framework::Lifetime::Timeframe);
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
