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

MergerInfrastructureBuilder::MergerInfrastructureBuilder() : mOutputSpec{header::gDataOriginInvalid, header::gDataDescriptionInvalid}
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
  mOutputSpec = outputSpec;
}

void MergerInfrastructureBuilder::setConfig(MergerConfig config)
{
  mConfig = config;
}

std::string MergerInfrastructureBuilder::validateConfig()
{
  std::string error;
  const std::string preamble = "mergers::MergerInfrastructureBuilder error: ";
  if (mInfrastructureName.empty()) {
    error += preamble + "the infrastructure name is empty\n";
  }
  if (mInputs.empty()) {
    error += preamble + "no inputs specified\n";
  }
  if (DataSpecUtils::validate(mOutputSpec) == false) {
    error += preamble + "invalid output\n";
  }

  if (mConfig.topologySize.value == TopologySize::NumberOfLayers && mConfig.topologySize.param < 1) {
    error += preamble + "number of layers less than 1 (" + std::to_string(mConfig.topologySize.param) + ")\n";
  }
  if (mConfig.topologySize.value == TopologySize::ReductionFactor && mConfig.topologySize.param < 2) {
    error += preamble + "reduction factor smaller than 2 (" + std::to_string(mConfig.topologySize.param) + ")\n";
  }

  if (mConfig.inputObjectTimespan.value == InputObjectsTimespan::FullHistory && mConfig.mergedObjectTimespan.value == MergedObjectTimespan::LastDifference) {
    error += preamble + "MergedObjectTimespan::LastDifference does not apply to InputObjectsTimespan::FullHistory\n";
  }

  for (const auto& input : mInputs) {
    if (DataSpecUtils::match(input, mOutputSpec)) {
      error += preamble + "output '" + DataSpecUtils::label(mOutputSpec) + "' matches input '" + DataSpecUtils::label(input) + "'. That will cause a circular dependency!";
    }
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

  // actual topology generation
  MergerBuilder mergerBuilder;
  mergerBuilder.setName(mInfrastructureName);
  for (size_t layer = 1; layer < mergersPerLayer.size(); layer++) {

    size_t numberOfMergers = mergersPerLayer[layer];
    size_t inputsPerMerger = layerInputs.size() / numberOfMergers;
    size_t inputsPerMergerRemainder = layerInputs.size() % numberOfMergers;

    MergerConfig layerConfig = mConfig;
    if (layer > 1 && mConfig.inputObjectTimespan.value == InputObjectsTimespan::LastDifference) {
      layerConfig.inputObjectTimespan = {InputObjectsTimespan::FullHistory};     // in LastDifference mode only the first layer should integrate
      layerConfig.mergedObjectTimespan = {MergedObjectTimespan::LastDifference}; // and objects that are merged should not be used again
    }
    mergerBuilder.setConfig(layerConfig);

    framework::Inputs nextLayerInputs;
    auto inputsRangeBegin = layerInputs.begin();

    for (size_t m = 0; m < numberOfMergers; m++) {

      mergerBuilder.setTopologyPosition(layer, m);

      auto inputsRangeEnd = inputsRangeBegin + inputsPerMerger + (m < inputsPerMergerRemainder);
      mergerBuilder.setInputSpecs(framework::Inputs(inputsRangeBegin, inputsRangeEnd));
      inputsRangeBegin = inputsRangeEnd;

      if (numberOfMergers == 1) {
        assert(layer == mergersPerLayer.size() - 1);
        // the last layer => use the specified external OutputSpec
        mergerBuilder.setOutputSpec(mOutputSpec);
      }

      auto merger = mergerBuilder.buildSpec();

      auto input = DataSpecUtils::matchingInput(merger.outputs.at(0));
      input.binding = "in";
      nextLayerInputs.push_back(input);

      workflow.emplace_back(std::move(merger));
    }
    layerInputs = nextLayerInputs; //todo: could be optimised with pointers
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

    size_t L = mConfig.topologySize.param;
    for (size_t i = 1; i <= L; i++) {
      mergersPerLayer.push_back(static_cast<size_t>(ceil(pow(inputs, (L - i) / static_cast<double>(L)))));
    }

  } else { // mConfig.topologySize.value == TopologySize::ReductionFactor
    //              _        _
    //             |  |V|     |  where:
    //  |V|  ---   |  | |i-1  |  R   - reduction factor
    //  | |i ---   | -------- |  i   - layer index (0 - input layer)
    //             |    R     |  M_i - number of mergers in i layer
    //

    double R = mConfig.topologySize.param;
    size_t Mi, prevMi = inputs;
    do {
      Mi = static_cast<size_t>(ceil(prevMi / R));
      mergersPerLayer.push_back(Mi);
      prevMi = Mi;
    } while (Mi > 1);
  }

  return mergersPerLayer;
}

void MergerInfrastructureBuilder::generateInfrastructure(framework::WorkflowSpec& workflow)
{
  auto mergersInfrastructure = generateInfrastructure();
  workflow.insert(std::end(workflow), std::begin(mergersInfrastructure), std::end(mergersInfrastructure));
}

} // namespace o2::mergers
