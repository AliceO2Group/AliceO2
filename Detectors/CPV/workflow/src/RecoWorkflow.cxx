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
#include "DataFormatsCPV/CPVBlockHeader.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "CPVWorkflow/RecoWorkflow.h"
#include "CPVWorkflow/ClusterizerSpec.h"
#include "CPVWorkflow/PublisherSpec.h"
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
  {"hits", InputType::Hits},
  {"digits", InputType::Digits},
  {"raw", InputType::Raw}};

const std::unordered_map<std::string, OutputType> OutputMap{
  {"digits", OutputType::Digits},
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
  //   specs.emplace_back(o2::cpv::reco_workflow::getRawWriterSpec());
  // }

  if (inputType == InputType::Raw) {
    //no explicit raw reader

    if (isEnabled(OutputType::Digits)) {
      specs.emplace_back(o2::cpv::reco_workflow::getRawToDigitConverterSpec());
    }
    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::cpv::reco_workflow::getRawToDigitConverterSpec());
      specs.emplace_back(o2::cpv::reco_workflow::getClusterizerSpec(propagateMC));
    }
  }

  if (inputType == InputType::Digits) {
    // specs.emplace_back(o2::cpv::getPublisherSpec(PublisherConf{
    //                                                "cpv-digit-reader",
    //                                                "o2sim",
    //                                                {"digitbranch", "CPVDigit", "Digit branch"},
    //                                                {"digittrigger", "CPVDigitTrigRecords", "TrigRecords branch"},
    //                                                {"mcbranch", "CPVDigitMCTruth", "MC label branch"},
    //                                                o2::framework::OutputSpec{"CPV", "DIGITS"},
    //                                                o2::framework::OutputSpec{"CPV", "DIGITTRIGREC"},
    //                                                o2::framework::OutputSpec{"CPV", "DIGITSMCTR"}},
    //                                              propagateMC));

    // if (enableDigitsPrinter) {
    //   specs.emplace_back(o2::cpv::reco_workflow::getDigitsPrinterSpec());
    // }

    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::cpv::reco_workflow::getClusterizerSpec(propagateMC));
    }

    // if (isEnabled(OutputType::Raw)) {
    //   // add Raw encoder
    //   specs.emplace_back(o2::cpv::reco_workflow::getRawWriterSpec());
    // }
  }

  // // check if the process is ready to quit
  // // this is decided upon the meta information in the CPV block header, the operation is set
  // // value kNoPayload in case of no data or no operation
  // // see also PublisherSpec.cxx
  // // in this workflow, the EOD is sent after the last real data, and all inputs will receive EOD,
  // // so it is enough to check on the first occurence
  // // FIXME: this will be changed once DPL can propagate control events like EOD
  // auto checkReady = [](o2::framework::DataRef const& ref) {
  //   auto const* cpvheader = o2::framework::DataRefUtils::getHeader<o2::cpv::CPVBlockHeader*>(ref);
  //   // sector number -1 indicates end-of-data
  //   if (cpvheader != nullptr) {
  //     // indicate normal processing if not ready and skip if ready
  //     if (!cpvheader->mHasPayload) {
  //       return std::make_tuple(o2::framework::MakeRootTreeWriterSpec::TerminationCondition::Action::SkipProcessing, true);
  //     }
  //   }
  //   return std::make_tuple(o2::framework::MakeRootTreeWriterSpec::TerminationCondition::Action::DoProcessing, false);
  // };

  // if (isEnabled(OutputType::Digits)) {
  //   using DigitOutputType = std::vector<o2::cpv::Digit>;
  //   using DTROutputType = std::vector<o2::cpv::TriggerRecord>;

  //   specs.emplace_back(o2::framework::MakeRootTreeWriterSpec("cpv-digits-writer", "cpvdigits.root", "o2sim",
  //                                                            -1,
  //                                                            o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
  //                                                            BranchDefinition<DigitOutputType>{o2::framework::InputSpec{"data", "CPV", "DIGITS", 0},
  //                                                                                              "CPVDigit",
  //                                                                                              "digit-branch-name"},
  //                                                            BranchDefinition<DTROutputType>{o2::framework::InputSpec{"data", "CPV", "DIGITTRIGREC", 0},
  //                                                                                            "CPVDigTR",
  //                                                                                            "digittr-branch-name"},
  //                                                            BranchDefinition<MCLabelContainer>{o2::framework::InputSpec{"mc", "CPV", "DIGITSMCTR", 0},
  //                                                                                               "CPVDigitMCTruth",
  //                                                                                               "digitmc-branch-name"})());
  // }

  // if (isEnabled(OutputType::Clusters)) {
  //   specs.emplace_back(o2::framework::MakeRootTreeWriterSpec("cpv-clusters-writer", "cpvclusters.root", "o2sim", -1,
  //                                                            o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
  //                                                            BranchDefinition<std::vector<o2::cpv::Cluster>>{o2::framework::InputSpec{"data", "CPV", "CLUSTERS", 0},
  //                                                                                                            "CPVCluster",
  //                                                                                                            "cluster-branch-name"},
  //                                                            BranchDefinition<std::vector<o2::cpv::TriggerRecord>>{o2::framework::InputSpec{"datatr", "CPV", "CLUSTERTRIGRECS", 0},
  //                                                                                                                  "CPVClusTR",
  //                                                                                                                  "clustertr-branch-name"},
  //                                                            BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{o2::framework::InputSpec{"mc", "CPV", "CLUSTERTRUEMC", 0},
  //                                                                                                                                 "CPVClusMC",
  //                                                                                                                                 "clustermc-branch-name"})());
  // }

  return std::move(specs);
}

} // namespace reco_workflow

} // namespace cpv

} // namespace o2
