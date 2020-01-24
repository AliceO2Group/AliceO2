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
//#include "PHOSWorkflow/RawWriterSpec.h"
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

  if (inputType == InputType::Digits) {
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

    if (enableDigitsPrinter)
      specs.emplace_back(o2::phos::reco_workflow::getPhosDigitsPrinterSpec());

    if (isEnabled(OutputType::Cells)) {
      // add converter for cells
      specs.emplace_back(o2::phos::reco_workflow::getCellConverterSpec(propagateMC));
    }

    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::phos::reco_workflow::getClusterizerSpec(propagateMC));
    }

    // if (isEnabled(OutputType::Raw)) {
    //   // add Raw encoder
    //   specs.emplace_back(o2::phos::reco_workflow::getRawWriterSpec());
    // }
  }

  if (inputType == InputType::Cells) {
    specs.emplace_back(o2::phos::getPublisherSpec(PublisherConf{
                                                    "phos-cell-reader",
                                                    "o2sim",
                                                    {"cellbranch", "PHSCell", "Cells branch"},
                                                    {"celltrigger", "PHSCellTR", "TrigRecords branch"},
                                                    {"mcbranch", "PHSCellMCTruth", "MC label branch"},
                                                    {"mcmapbranch", "PHSCellMCMap", "MC label map branch"},
                                                    o2::framework::OutputSpec{"PHS", "CELLS"},
                                                    o2::framework::OutputSpec{"PHS", "CELLTRIGREC"},
                                                    o2::framework::OutputSpec{"PHS", "CELLSMCTR"},
                                                    o2::framework::OutputSpec{"PHS", "CELLSMCMAP"}},
                                                  propagateMC, true));

    if (isEnabled(OutputType::Clusters)) {
      // add clusterizer
      specs.emplace_back(o2::phos::reco_workflow::getCellClusterizerSpec(propagateMC));
    }
  }

  // check if the process is ready to quit
  // this is decided upon the meta information in the PHOS block header, the operation is set
  // value kNoPayload in case of no data or no operation
  // see also PublisherSpec.cxx
  // in this workflow, the EOD is sent after the last real data, and all inputs will receive EOD,
  // so it is enough to check on the first occurence
  // FIXME: this will be changed once DPL can propagate control events like EOD
  auto checkReady = [](o2::framework::DataRef const& ref) {
    auto const* phosheader = o2::framework::DataRefUtils::getHeader<o2::phos::PHOSBlockHeader*>(ref);
    // sector number -1 indicates end-of-data
    if (phosheader != nullptr) {
      // indicate normal processing if not ready and skip if ready
      if (!phosheader->mHasPayload) {
        return std::make_tuple(o2::framework::MakeRootTreeWriterSpec::TerminationCondition::Action::SkipProcessing, true);
      }
    }
    return std::make_tuple(o2::framework::MakeRootTreeWriterSpec::TerminationCondition::Action::DoProcessing, false);
  };

  // auto makeWriterSpec = [propagateMC, checkReady](const char* processName,
  //                                                 const char* defaultFileName,
  //                                                 const char* defaultTreeName,
  //                                                 bool createMCMap,
  //                                                 auto&& databranch,
  //                                                 auto&& datatrbranch,
  //                                                 auto&& mcbranch=nullptr,
  //                                                 auto&& mcmapbranch=nullptr) {
  //   // depending on the MC propagation flag, the RootTreeWriter spec is created with two
  //   // or one branch definition
  //   if (propagateMC) {
  //     if(createMCMap){
  //        return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
  //                                                            o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
  //                                                            std::move(databranch),
  //                                                            std::move(datatrbranch),
  //                                                            std::move(mcbranch),
  //                                                            std::move(mcmapbranch)));
  //     }
  //     else{
  //       return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
  //                                                            o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
  //                                                            std::move(databranch),
  //                                                            std::move(datatrbranch),
  //                                                            std::move(mcbranch)));
  //    }
  //   }
  //   else{
  //   return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
  //                                                          o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
  //                                                          std::move(databranch),
  //                                                          std::move(datatrbranch)));
  //   }
  // };

  // if (isEnabled(OutputType::Raw)) {
  //   using RawOutputType = std::vector<o2::phos::Raw>;
  //   using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::phos::MCLabel>;
  //   specs.push_back(makeWriterSpec("phos-raw-writer",
  //                                  inputType == InputType::Digits ? "phos-raw.root" : "phosrawcells.root",
  //                                  "o2sim",
  //                                  BranchDefinition<DigitOutputType>{o2::framework::InputSpec{"data", "PHS", "RAW", 0},
  //                                                                    "PHSRaw",
  //                                                                    "raw-branch-name"},
  //                                  BranchDefinition<MCLabelContainer>{o2::framework::InputSpec{"mc", "PHS", "RAWMCTR", 0},
  //                                                                     "PHSRawMCTruth",
  //                                                                     "rawmc-branch-name"})());
  // }

  if (isEnabled(OutputType::Digits)) {
    using DigitOutputType = std::vector<o2::phos::Digit>;
    using DTROutputType = std::vector<o2::phos::TriggerRecord>;
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::phos::MCLabel>;

    specs.emplace_back(o2::framework::MakeRootTreeWriterSpec("phos-digits-writer", "phosdigits.root", "o2sim",
                                                             -1,
                                                             o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                                             BranchDefinition<DigitOutputType>{o2::framework::InputSpec{"data", "PHS", "DIGITS", 0},
                                                                                               "PHSDigit",
                                                                                               "digit-branch-name"},
                                                             BranchDefinition<DTROutputType>{o2::framework::InputSpec{"data", "PHS", "DIGITTRIGREC", 0},
                                                                                             "PHSDigTR",
                                                                                             "digittr-branch-name"},
                                                             BranchDefinition<MCLabelContainer>{o2::framework::InputSpec{"mc", "PHS", "DIGITSMCTR", 0},
                                                                                                "PHSDigitMCTruth",
                                                                                                "digitmc-branch-name"})());
  }

  if (isEnabled(OutputType::Cells)) {
    using CellOutputType = std::vector<o2::phos::Cell>;
    using CTROutputType = std::vector<o2::phos::TriggerRecord>;
    using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::phos::MCLabel>;
    specs.emplace_back(o2::framework::MakeRootTreeWriterSpec("phos-cells-writer", "phoscells.root", "o2sim", -1,
                                                             o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                                             BranchDefinition<CellOutputType>{o2::framework::InputSpec{"data", "PHS", "CELLS", 0},
                                                                                              "PHSCell",
                                                                                              "cell-branch-name"},
                                                             BranchDefinition<CTROutputType>{o2::framework::InputSpec{"data", "PHS", "CELLTRIGREC", 0},
                                                                                             "PHSCellTR",
                                                                                             "celltr-branch-name"},
                                                             BranchDefinition<MCLabelContainer>{o2::framework::InputSpec{"mc", "PHS", "CELLSMCTR", 0},
                                                                                                "PHSCellMCTruth",
                                                                                                "cellmc-branch-name"},
                                                             BranchDefinition<std::vector<uint>>{o2::framework::InputSpec{"mcmap", "PHS", "CELLSMCMAP", 0},
                                                                                                 "PHSCellMCMAP",
                                                                                                 "cellmcmap-branch-name"})());
  }

  if (isEnabled(OutputType::Clusters)) {
    specs.emplace_back(o2::framework::MakeRootTreeWriterSpec("phos-clusters-writer", "phosclusters.root", "o2sim", -1,
                                                             o2::framework::MakeRootTreeWriterSpec::TerminationCondition{checkReady},
                                                             BranchDefinition<std::vector<o2::phos::Cluster>>{o2::framework::InputSpec{"data", "PHS", "CLUSTERS", 0},
                                                                                                              "PHSCluster",
                                                                                                              "cluster-branch-name"},
                                                             BranchDefinition<std::vector<o2::phos::TriggerRecord>>{o2::framework::InputSpec{"datatr", "PHS", "CLUSTERTRIGRECS", 0},
                                                                                                                    "PHSClusTR",
                                                                                                                    "clustertr-branch-name"},
                                                             BranchDefinition<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>>{o2::framework::InputSpec{"mc", "PHS", "CLUSTERTRUEMC", 0},
                                                                                                                                    "PHSClusMC",
                                                                                                                                    "clustermc-branch-name"})());
  }

  return std::move(specs);
}

} // namespace reco_workflow

} // namespace phos

} // namespace o2
