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

#include "Algorithm/RangeTokenizer.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "PHOSWorkflow/RecoWorkflow.h"
#include "PHOSWorkflow/CellConverterSpec.h"
#include "PHOSWorkflow/ClusterizerSpec.h"
#include "PHOSWorkflow/ReaderSpec.h"
#include "PHOSWorkflow/WriterSpec.h"
#include "PHOSWorkflow/RawToCellConverterSpec.h"
#include "PHOSWorkflow/RawWriterSpec.h"
#include "Framework/DataSpecUtils.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

namespace phos
{

namespace reco_workflow
{

const std::unordered_map<std::string, InputType> InputMap{
  {"raw", InputType::Raw},
  {"digits", InputType::Digits},
  {"cells", InputType::Cells}};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"cells", OutputType::Cells},
  {"clusters", OutputType::Clusters}};

o2::framework::WorkflowSpec getWorkflow(bool disableRootInp,
                                        bool disableRootOut,
                                        bool propagateMC,
                                        std::string const& cfgInput,
                                        std::string const& cfgOutput,
                                        bool fullCluOut,
                                        int flpId,
                                        bool defBadMap,
                                        bool skipL1phase)
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

  // Raw to ....
  if (inputType == InputType::Raw) {
    // no explicit raw reader ??

    if (isEnabled(OutputType::Cells)) {
      specs.emplace_back(o2::phos::reco_workflow::getRawToCellConverterSpec(static_cast<unsigned int>(flpId)));
      if (!disableRootOut) {
        specs.emplace_back(o2::phos::getCellWriterSpec(false));
      }
    }
    if (isEnabled(OutputType::Clusters)) {
      specs.emplace_back(o2::phos::reco_workflow::getRawToCellConverterSpec(static_cast<unsigned int>(flpId)));
      specs.emplace_back(o2::phos::reco_workflow::getCellClusterizerSpec(false, fullCluOut, defBadMap, skipL1phase)); // no MC propagation
      if (!disableRootOut) {
        specs.emplace_back(o2::phos::getClusterWriterSpec(false));
      }
    }
  }

  // Digits to ....
  if (inputType == InputType::Digits) {
    if (!disableRootInp) {
      specs.emplace_back(o2::phos::getDigitsReaderSpec(propagateMC));
    }
    if (isEnabled(OutputType::Cells)) {
      // add converter for cells
      specs.emplace_back(o2::phos::reco_workflow::getCellConverterSpec(propagateMC, defBadMap));
      if (!disableRootOut) {
        specs.emplace_back(o2::phos::getCellWriterSpec(propagateMC));
      }
    } else {
      if (isEnabled(OutputType::Clusters)) {
        specs.emplace_back(o2::phos::reco_workflow::getClusterizerSpec(propagateMC, fullCluOut, defBadMap));
        if (!disableRootOut) {
          specs.emplace_back(o2::phos::getClusterWriterSpec(propagateMC));
        }
      }
    }
  }

  // Cells to
  if (inputType == InputType::Cells) {
    if (!disableRootInp) {
      specs.emplace_back(o2::phos::getCellReaderSpec(propagateMC));
    }
    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::phos::reco_workflow::getCellClusterizerSpec(propagateMC, fullCluOut, defBadMap, skipL1phase));
      if (!disableRootOut) {
        specs.emplace_back(o2::phos::getClusterWriterSpec(propagateMC));
      }
    }
  }

  return specs;
}

} // namespace reco_workflow

} // namespace phos

} // namespace o2
