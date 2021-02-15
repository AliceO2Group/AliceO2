// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "FairLogger.h"

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
#include "PHOSWorkflow/DigitsPrinterSpec.h"
#include "PHOSWorkflow/PublisherSpec.h"
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
  {"hits", InputType::Hits},
  {"digits", InputType::Digits},
  {"cells", InputType::Cells},
  {"raw", InputType::Raw}};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"digits", OutputType::Digits},
  {"cells", OutputType::Cells},
  {"raw", OutputType::Raw},
  {"clusters", OutputType::Clusters}};

o2::framework::WorkflowSpec getWorkflow(bool propagateMC,
                                        bool enableDigitsPrinter,
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

  // if (isEnabled(OutputType::Raw)) {
  //   // add Raw encoder
  //   specs.emplace_back(o2::phos::reco_workflow::getRawWriterSpec());
  // }

  if (inputType == InputType::Raw) {
    //no explicit raw reader

    if (isEnabled(OutputType::Cells)) {
      specs.emplace_back(o2::phos::reco_workflow::getRawToCellConverterSpec());
    }
  }

  if (inputType == InputType::Digits) {
    if (isEnabled(OutputType::Cells)) {
      specs.emplace_back(o2::phos::getPublisherSpec(PublisherConf{
                                                      "phos-digit-reader",
                                                      "o2sim",
                                                      {"digitbranch", "PHOSDigit", "Digit branch"},
                                                      {"digittrigger", "PHOSDigitTrigRecords", "TrigRecords branch"},
                                                      {"mcbranch", "PHOSDigitMCTruth", "MC label branch"},
                                                      {"mcmapbranch", "", "Dummy branch"},
                                                      o2::framework::OutputSpec{"PHS", "DIGITS"},
                                                      o2::framework::OutputSpec{"PHS", "DIGITTRIGREC"},
                                                      o2::framework::OutputSpec{"PHS", "DIGITSMCTR"},
                                                      o2::framework::OutputSpec{"PHS", ""}}, // it empty, do not create
                                                    propagateMC, false));
      // add converter for cells
      specs.emplace_back(o2::phos::reco_workflow::getCellConverterSpec(propagateMC));
    }

    if (isEnabled(OutputType::Clusters)) {
      specs.emplace_back(o2::phos::getPublisherSpec(PublisherConf{
                                                      "phos-digit-reader",
                                                      "o2sim",
                                                      {"digitbranch", "PHOSDigit", "Digit branch"},
                                                      {"digittrigger", "PHOSDigitTrigRecords", "TrigRecords branch"},
                                                      {"mcbranch", "PHOSDigitMCTruth", "MC label branch"},
                                                      {"mcmapbranch", "", "Dummy branch"},
                                                      o2::framework::OutputSpec{"PHS", "DIGITS"},
                                                      o2::framework::OutputSpec{"PHS", "DIGITTRIGREC"},
                                                      o2::framework::OutputSpec{"PHS", "DIGITSMCTR"},
                                                      o2::framework::OutputSpec{"PHS", ""}}, // it empty, do not create
                                                    propagateMC, false));
      // add clusterizer
      specs.emplace_back(o2::phos::reco_workflow::getClusterizerSpec(propagateMC));
    }

    if (enableDigitsPrinter) {
      specs.emplace_back(o2::phos::reco_workflow::getPhosDigitsPrinterSpec());
    }
  }

  if (inputType == InputType::Cells) {
    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::phos::reco_workflow::getCellClusterizerSpec(propagateMC));
    }
  }

  return std::move(specs);
}

} // namespace reco_workflow

} // namespace phos

} // namespace o2
