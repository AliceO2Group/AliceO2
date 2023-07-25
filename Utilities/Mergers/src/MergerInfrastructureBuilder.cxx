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

/// \file MergerInfrastructureBuilder.cxx
/// \brief Definition of Mergers' Infrastructure Builder
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerInfrastructureBuilder.h"
#include "Mergers/MergerAlgorithm.h"
#include "Mergers/MergerBuilder.h"

#include "Framework/DataSpecUtils.h"

using namespace o2::framework;

namespace o2::mergers
{

MergerInfrastructureBuilder::MergerInfrastructureBuilder()
  : mOutputSpecIntegral{header::gDataOriginInvalid, header::gDataDescriptionInvalid},
    mOutputSpecMovingWindow{header::gDataOriginInvalid, header::gDataDescriptionInvalid}
{
}

void MergerInfrastructureBuilder::setInfrastructureName(std::string name)
{
  mInfrastructureName = name;
}

void MergerInfrastructureBuilder::setInputSpecs(const framework::Inputs& inputs)
{
  mInputs = inputs;
}

void MergerInfrastructureBuilder::setOutputSpec(const framework::OutputSpec& outputSpec)
{
  mOutputSpecIntegral = outputSpec;
}

void MergerInfrastructureBuilder::setOutputSpecMovingWindow(const framework::OutputSpec& outputSpec)
{
  mOutputSpecMovingWindow = outputSpec;
}

void MergerInfrastructureBuilder::setConfig(MergerConfig config)
{
  mConfig = config;
}

std::string MergerInfrastructureBuilder::validateConfig()
{
  std::string error;
  const std::string preamble = "MergerInfrastructureBuilder error: ";
  if (mInfrastructureName.empty()) {
    error += preamble + "the infrastructure name is empty\n";
  }
  if (mInputs.empty()) {
    error += preamble + "no inputs specified\n";
  }
  if (DataSpecUtils::validate(mOutputSpecIntegral) == false) {
    error += preamble + "invalid output\n";
  }

  if ((mConfig.topologySize.value == TopologySize::NumberOfLayers || mConfig.topologySize.value == TopologySize::ReductionFactor) && !std::holds_alternative<int>(mConfig.topologySize.param)) {
    error += preamble + "TopologySize::NumberOfLayers and TopologySize::ReductionFactor require a single int as parameter\n";
  } else {
    if (mConfig.topologySize.value == TopologySize::NumberOfLayers && std::get<int>(mConfig.topologySize.param) < 1) {
      error += preamble + "number of layers less than 1 (" + std::to_string(std::get<int>(mConfig.topologySize.param)) + ")\n";
    }
    if (mConfig.topologySize.value == TopologySize::ReductionFactor && std::get<int>(mConfig.topologySize.param) < 2) {
      error += preamble + "reduction factor smaller than 2 (" + std::to_string(std::get<int>(mConfig.topologySize.param)) + ")\n";
    }
  }
  if (mConfig.topologySize.value == TopologySize::MergersPerLayer) {
    if (!std::holds_alternative<std::vector<size_t>>(mConfig.topologySize.param)) {
      error += preamble + "TopologySize::MergersPerLayer require std::vector<size_t> as parameter\n";
    } else {
      auto mergersPerLayer = std::get<std::vector<size_t>>(mConfig.topologySize.param);
      if (mergersPerLayer.empty()) {
        error += preamble + "TopologySize::MergersPerLayer was used, but the provided vector is empty\n";
      } else if (mergersPerLayer.back() != 1) {
        error += preamble + "Last Merger layer should consist of one Merger, " + mergersPerLayer.back() + " was used\n";
      }
    }
  }

  if (mConfig.inputObjectTimespan.value == InputObjectsTimespan::FullHistory && mConfig.parallelismType.value == ParallelismType::RoundRobin) {
    error += preamble + "ParallelismType::RoundRobin does not apply to InputObjectsTimespan::FullHistory\n";
  }

  if (mConfig.inputObjectTimespan.value == InputObjectsTimespan::FullHistory && mConfig.mergedObjectTimespan.value == MergedObjectTimespan::LastDifference) {
    error += preamble + "MergedObjectTimespan::LastDifference does not apply to InputObjectsTimespan::FullHistory\n";
  }

  if (mConfig.publishMovingWindow.value == PublishMovingWindow::Yes && mConfig.inputObjectTimespan.value == InputObjectsTimespan::FullHistory) {
    error += preamble + "PublishMovingWindow::Yes is not supported with InputObjectsTimespan::FullHistory\n";
  }

  for (const auto& input : mInputs) {
    if (DataSpecUtils::match(input, mOutputSpecIntegral)) {
      error += preamble + "output '" + DataSpecUtils::label(mOutputSpecIntegral) + "' matches input '" + DataSpecUtils::label(input) + "'. That will cause a circular dependency!";
    }
  }

  if (mConfig.detectorName.empty()) {
    error += preamble + "detector name is empty";
  }

  return error;
}

framework::WorkflowSpec MergerInfrastructureBuilder::generateInfrastructure()
{
  if (std::string error = validateConfig(); !error.empty()) {
    throw std::runtime_error(error);
  }

  framework::WorkflowSpec workflow;
  auto layerInputs = mInputs;

  // preparing some numbers
  auto mergersPerLayer = computeNumberOfMergersPerLayer(layerInputs.size());

  // topology generation
  MergerBuilder mergerBuilder;
  mergerBuilder.setName(mInfrastructureName);
  mergerBuilder.setOutputSpecMovingWindow(mOutputSpecMovingWindow);

  for (size_t layer = 1; layer < mergersPerLayer.size(); layer++) {

    size_t numberOfMergers = mergersPerLayer[layer];
    size_t splitInputsMergers = mConfig.parallelismType.value == ParallelismType::SplitInputs ? numberOfMergers : 1;
    size_t timePipelineVal = mConfig.parallelismType.value == ParallelismType::SplitInputs ? 1 : numberOfMergers;
    size_t inputsPerMerger = layerInputs.size() / splitInputsMergers;
    size_t inputsPerMergerRemainder = layerInputs.size() % splitInputsMergers;

    MergerConfig layerConfig = mConfig;
    if (layer < mergersPerLayer.size() - 1) {
      // in intermediate layers we should reset the results, so the same data is not added many times.
      layerConfig.mergedObjectTimespan = {MergedObjectTimespan::NCycles, 1};
      // we also expect moving windows to be published only by the last layer
      layerConfig.publishMovingWindow = {PublishMovingWindow::No};
    }
    mergerBuilder.setConfig(layerConfig);

    framework::Inputs nextLayerInputs;
    auto inputsRangeBegin = layerInputs.begin();

    for (size_t m = 0; m < splitInputsMergers; m++) {

      mergerBuilder.setTopologyPosition(layer, m);
      mergerBuilder.setTimePipeline(timePipelineVal);

      auto inputsRangeEnd = inputsRangeBegin + inputsPerMerger + (m < inputsPerMergerRemainder);
      mergerBuilder.setInputSpecs(framework::Inputs(inputsRangeBegin, inputsRangeEnd));
      inputsRangeBegin = inputsRangeEnd;

      if (layer == mergersPerLayer.size() - 1) {
        // the last layer => use the specified external OutputSpec
        mergerBuilder.setOutputSpec(mOutputSpecIntegral);
      }

      auto merger = mergerBuilder.buildSpec();

      auto input = DataSpecUtils::matchingInput(merger.outputs.at(0));
      input.binding = "in";
      nextLayerInputs.push_back(input);

      workflow.emplace_back(std::move(merger));
    }
    layerInputs = nextLayerInputs; // todo: could be optimised with pointers
  }

  return workflow;
}

std::vector<size_t> MergerInfrastructureBuilder::computeNumberOfMergersPerLayer(const size_t inputs) const
{
  std::vector<size_t> mergersPerLayer{inputs};
  if (mConfig.topologySize.value == TopologySize::NumberOfLayers) {
    //              _          _
    //             |    L - i   |  where:
    //             |    -----   |  L   - number of layers
    //  |V|  ---   | |V|  L     |  i   - layer index (0 - input layer)
    //  | |i ---   | | |0       |  M_i - number of mergers in i layer
    //             |            |
    //

    size_t L = std::get<int>(mConfig.topologySize.param);
    for (size_t i = 1; i <= L; i++) {
      mergersPerLayer.push_back(static_cast<size_t>(ceil(pow(inputs, (L - i) / static_cast<double>(L)))));
    }

  } else if (mConfig.topologySize.value == TopologySize::ReductionFactor) {
    //              _        _
    //             |  |V|     |  where:
    //  |V|  ---   |  | |i-1  |  R   - reduction factor
    //  | |i ---   | -------- |  i   - layer index (0 - input layer)
    //             |    R     |  M_i - number of mergers in i layer
    //

    double R = std::get<int>(mConfig.topologySize.param);
    size_t Mi, prevMi = inputs;
    do {
      Mi = static_cast<size_t>(ceil(prevMi / R));
      mergersPerLayer.push_back(Mi);
      prevMi = Mi;
    } while (Mi > 1);
  } else { // mConfig.topologySize.value == TopologySize::MergersPerLayer
    auto mergersPerLayerConfig = std::get<std::vector<size_t>>(mConfig.topologySize.param);
    mergersPerLayer.insert(mergersPerLayer.cend(), mergersPerLayerConfig.begin(), mergersPerLayerConfig.end());
  }

  return mergersPerLayer;
}

void MergerInfrastructureBuilder::generateInfrastructure(framework::WorkflowSpec& workflow)
{
  auto mergersInfrastructure = generateInfrastructure();
  workflow.insert(std::end(workflow), std::begin(mergersInfrastructure), std::end(mergersInfrastructure));
}

} // namespace o2::mergers
