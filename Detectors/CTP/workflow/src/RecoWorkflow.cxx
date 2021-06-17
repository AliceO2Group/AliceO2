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

#include "Framework/RootSerializationSupport.h"
#include "Algorithm/RangeTokenizer.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsCTP/Digits.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CTPWorkflow/RecoWorkflow.h"
#include "CTPWorkflow/ReaderSpec.h"
#include "CTPWorkflow/WriterSpec.h"
#include "CTPWorkflow/RawToDigitConverterSpec.h"
#include "Framework/DataSpecUtils.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::dataformats;

namespace o2
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

namespace ctp
{

namespace reco_workflow
{

const std::unordered_map<std::string, InputType> InputMap{
  {"raw", InputType::Raw},
  {"digits", InputType::Digits}};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"digits", OutputType::Digits}};

o2::framework::WorkflowSpec getWorkflow(bool disableRootInp,
                                        bool disableRootOut,
                                        bool propagateMC,
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
    //no explicit raw reader

    if (isEnabled(OutputType::Digits)) {
      specs.emplace_back(o2::ctp::reco_workflow::getRawToDigitConverterSpec());
      if (!disableRootOut) {
        specs.emplace_back(o2::ctp::getDigitWriterSpec(false));
      }
    }
  }

  // Digits to ....
  if (inputType == InputType::Digits) {
    if (!disableRootInp) {
      specs.emplace_back(o2::ctp::getDigitsReaderSpec(propagateMC));
    }
  }
  return std::move(specs);
}

} // namespace reco_workflow

} // namespace ctp

} // namespace o2
