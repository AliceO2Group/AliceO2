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

#include <algorithm>
#include <unordered_map>
#include <vector>

#include <fairlogger/Logger.h>

#include "Framework/RootSerializationSupport.h"
#include "Algorithm/RangeTokenizer.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsCPV/CPVBlockHeader.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "CPVWorkflow/RecoWorkflow.h"
#include "CPVWorkflow/DigitReaderSpec.h"
#include "CPVWorkflow/ClusterizerSpec.h"
#include "CPVWorkflow/ReaderSpec.h"
#include "CPVWorkflow/WriterSpec.h"
#include "CPVWorkflow/RawToDigitConverterSpec.h"
//#include "CPVWorkflow/RawWriterSpec.h"
#include "Framework/DataSpecUtils.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::dataformats;

namespace o2
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

namespace cpv
{

namespace reco_workflow
{

const std::unordered_map<std::string, InputType> InputMap{
  {"raw", InputType::Raw},
  {"hits", InputType::Hits},
  {"digits", InputType::Digits}};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"digits", OutputType::Digits},
  {"clusters", OutputType::Clusters}};

o2::framework::WorkflowSpec getWorkflow(bool disableRootInp,
                                        bool disableRootOut,
                                        bool propagateMC,
                                        bool askSTFDist,
                                        bool isPedestal,
                                        bool useBadChannelMap,
                                        bool useGainCalibration,
                                        std::string const& cfgInput,
                                        std::string const& cfgOutput)
{
  InputType inputType;

  try {
    inputType = InputMap.at(cfgInput);
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid input type: ") + cfgInput);
  }
  std::vector<OutputType> outputTypes;
  try {
    outputTypes = RangeTokenizer::tokenize<OutputType>(cfgOutput, [](std::string const& token) { return OutputMap.at(token); });
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid output type: ") + cfgOutput);
  }
  auto isEnabled = [&outputTypes](OutputType type) {
    return std::find(outputTypes.begin(), outputTypes.end(), type) != outputTypes.end();
  };

  o2::framework::WorkflowSpec specs;

  // //Raw to ....
  if (inputType == InputType::Raw) {
    // no explicit raw reader

    if (isEnabled(OutputType::Digits)) {
      specs.emplace_back(o2::cpv::reco_workflow::getRawToDigitConverterSpec(askSTFDist, isPedestal, useBadChannelMap, useGainCalibration));
      if (!disableRootOut) {
        specs.emplace_back(o2::cpv::getDigitWriterSpec(false));
      }
    }
    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::cpv::reco_workflow::getRawToDigitConverterSpec(askSTFDist, isPedestal, useBadChannelMap, useGainCalibration));
      specs.emplace_back(o2::cpv::reco_workflow::getClusterizerSpec(false));
      if (!disableRootOut) {
        specs.emplace_back(o2::cpv::getClusterWriterSpec(false));
      }
    }
  }

  // Digits to ....
  if (inputType == InputType::Digits) {
    if (!disableRootInp) {
      specs.emplace_back(o2::cpv::getCPVDigitReaderSpec(propagateMC));
      // specs.emplace_back(o2::cpv::getDigitsReaderSpec(propagateMC));
    }

    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::cpv::reco_workflow::getClusterizerSpec(propagateMC));
      if (!disableRootOut) {
        specs.emplace_back(o2::cpv::getClusterWriterSpec(propagateMC));
      }
    }
  }

  return specs;
}

} // namespace reco_workflow

} // namespace cpv

} // namespace o2
