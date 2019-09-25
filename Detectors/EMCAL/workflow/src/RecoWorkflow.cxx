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
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALWorkflow/RecoWorkflow.h"
#include "EMCALWorkflow/DigitsPrinterSpec.h"
#include "EMCALWorkflow/PublisherSpec.h"
#include "Framework/DataSpecUtils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

namespace emcal
{

namespace reco_workflow
{

const std::unordered_map<std::string, InputType> InputMap{
  {"digitizer", InputType::Digitizer},
  {"digits", InputType::Digits},
  {"raw", InputType::Raw},
  {"clusters", InputType::Clusters},
};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"digits", OutputType::Digits},
  {"raw", OutputType::Raw},
  {"clusters", OutputType::Clusters}};

o2::framework::WorkflowSpec getWorkflow(bool propagateMC,
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

  if (inputType == InputType::Digits) {
    specs.emplace_back(o2::emcal::getPublisherSpec(PublisherConf{
                                                     "emcal-digit-reader",
                                                     "o2sim",
                                                     {"digitbranch", "EMCALDigit", "Digit branch"},
                                                     {"mcbranch", "EMCALDigitMCTruth", "MC label branch"},
                                                     o2::framework::OutputSpec{"EMC", "DIGITS"},
                                                     o2::framework::OutputSpec{"EMC", "DIGITSMCTR"}},
                                                   propagateMC));

    specs.emplace_back(o2::emcal::reco_workflow::getEmcalDigitsPrinterSpec());
  }

  // check if the process is ready to quit
  // this is decided upon the meta information in the EMCAL block header, the operation is set
  // value kNoPayload in case of no data or no operation
  // see also PublisherSpec.cxx
  // in this workflow, the EOD is sent after the last real data, and all inputs will receive EOD,
  // so it is enough to check on the first occurence
  // FIXME: this will be changed once DPL can propagate control events like EOD
  auto checkReady = [](o2::framework::DataRef const& ref) {
    auto const* emcalheader = o2::framework::DataRefUtils::getHeader<o2::emcal::EMCALBlockHeader*>(ref);
    // sector number -1 indicates end-of-data
    if (emcalheader != nullptr) {
      // indicate normal processing if not ready and skip if ready
      if (!emcalheader->mHasPayload) {
        return std::make_tuple(o2::framework::MakeRootTreeWriterSpec::TerminationCondition::Action::SkipProcessing, true);
      }
    }
    return std::make_tuple(o2::framework::MakeRootTreeWriterSpec::TerminationCondition::Action::DoProcessing, false);
  };

  auto makeWriterSpec = [propagateMC, checkReady](const char* processName, const char* defaultFileName, const char* defaultTreeName,
                                                  auto&& databranch, auto&& mcbranch) {
    // depending on the MC propagation flag, the RootTreeWriter spec is created with two
    // or one branch definition
    if (propagateMC) {
      return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                             o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                                             std::move(databranch),
                                                             std::move(mcbranch)));
    }
    return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                           o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                                           std::move(databranch)));
  };

  if (isEnabled(OutputType::Digits)) {
    using DigitOutputType = std::vector<o2::emcal::Digit>;
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
    specs.push_back(makeWriterSpec("emcal-digits-writer",
                                   inputType == InputType::Digits ? "emc-filtered-digits.root" : "emcdigits.root",
                                   "o2sim",
                                   BranchDefinition<DigitOutputType>{o2::framework::InputSpec{"data", "EMC", "DIGITS", 0},
                                                                     "EMCDigit",
                                                                     "digit-branch-name"},
                                   BranchDefinition<MCLabelContainer>{o2::framework::InputSpec{"mc", "EMC", "DIGITSMCTR", 0},
                                                                      "EMCDigitMCTruth",
                                                                      "digitmc-branch-name"})());
  }

  return std::move(specs);
}

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
