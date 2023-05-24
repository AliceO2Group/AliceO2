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
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/ErrorTypeFEE.h"
#include "EMCALWorkflow/RecoWorkflow.h"
#include "EMCALWorkflow/CellConverterSpec.h"
#include "EMCALWorkflow/ClusterizerSpec.h"
#include "EMCALWorkflow/AnalysisClusterSpec.h"
#include "EMCALWorkflow/DigitsPrinterSpec.h"
#include "EMCALWorkflow/PublisherSpec.h"
#include "EMCALWorkflow/RawToCellConverterSpec.h"
#include "Framework/DataSpecUtils.h"
#include "DataFormatsEMCAL/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

namespace emcal
{

namespace reco_workflow
{

o2::framework::WorkflowSpec getWorkflow(bool propagateMC,
                                        bool askDISTSTF,
                                        bool enableDigitsPrinter,
                                        int subspecificationIn,
                                        int subspecificationOut,
                                        std::string const& cfgInput,
                                        std::string const& cfgOutput,
                                        bool disableRootInput,
                                        bool disableRootOutput,
                                        bool disableDecodingErrors,
                                        bool useccdb)
{

  const std::unordered_map<std::string, InputType> InputMap{
    {"digits", InputType::Digits},
    {"cells", InputType::Cells},
    {"raw", InputType::Raw},
    {"clusters", InputType::Clusters},
  };

  const std::unordered_map<std::string, OutputType> OutputMap{
    {"digits", OutputType::Digits},
    {"cells", OutputType::Cells},
    {"raw", OutputType::Raw},
    {"clusters", OutputType::Clusters},
    {"analysisclusters", OutputType::AnalysisClusters}};

  std::unordered_map<InputType, std::vector<OutputType>> allowedIO;
  allowedIO[InputType::Digits] = std::vector<OutputType>{OutputType::Cells, OutputType::Clusters, OutputType::AnalysisClusters};
  allowedIO[InputType::Cells] = std::vector<OutputType>{OutputType::Cells, OutputType::Clusters, OutputType::AnalysisClusters};
  allowedIO[InputType::Raw] = std::vector<OutputType>{OutputType::Cells};

  InputType inputType;

  try {
    inputType = InputMap.at(cfgInput);
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid input type: ") + cfgInput);
  }
  std::vector<OutputType> outputTypes;
  try {
    outputTypes = RangeTokenizer::tokenize<OutputType>(cfgOutput, [&](std::string const& token) { return OutputMap.at(token); });
  } catch (std::out_of_range&) {
    throw std::invalid_argument(std::string("invalid output type: ") + cfgOutput);
  }
  auto isEnabled = [&outputTypes](OutputType type) {
    return std::find(outputTypes.begin(), outputTypes.end(), type) != outputTypes.end();
  };

  auto isAllowedIOCombination = [&allowedIO](InputType inp, OutputType out) {
    const auto& vout = allowedIO[inp];
    return std::find(vout.begin(), vout.end(), out) != vout.end();
  };

  auto getOutputTypeName = [&OutputMap](OutputType out) {
    std::string str;
    for (const auto& o : OutputMap) {
      if (o.second == out) {
        str = std::string(o.first);
      }
    }
    return str;
  };

  // make sure inputs/outputs combinatios are enabled
  for (const auto outType : outputTypes) {
    if (!isAllowedIOCombination(inputType, outType)) {
      throw std::runtime_error(fmt::format("Input {:s} is not allowed with output {:s}", cfgInput, getOutputTypeName(outType)));
    }
  }

  if (inputType == InputType::Raw) {
    propagateMC = false;
  }

  o2::framework::WorkflowSpec specs;

  if (inputType == InputType::Digits) {
    using digitInputType = std::vector<o2::emcal::Digit>;
    if (!disableRootInput) {
      specs.emplace_back(o2::emcal::getPublisherSpec<digitInputType>(PublisherConf{
                                                                       "emcal-digit-reader",
                                                                       "o2sim",
                                                                       "emcdigits.root",
                                                                       {"digitbranch", "EMCALDigit", "Digit branch"},
                                                                       {"digittriggerbranch", "EMCALDigitTRGR", "Trigger record branch"},
                                                                       {"mcbranch", "EMCALDigitMCTruth", "MC label branch"},
                                                                       o2::framework::OutputSpec{"EMC", "DIGITS"},
                                                                       o2::framework::OutputSpec{"EMC", "DIGITSTRGR"},
                                                                       o2::framework::OutputSpec{"EMC", "DIGITSMCTR"}},
                                                                     0, propagateMC));
    }

    if (enableDigitsPrinter) {
      try {
        specs.emplace_back(o2::emcal::reco_workflow::getEmcalDigitsPrinterSpec("digits"));
      } catch (std::runtime_error& e) {
        LOG(error) << "Cannot create digits printer spec: " << e.what();
      }
    }
  } else if (inputType == InputType::Cells) {
    using cellInputType = std::vector<o2::emcal::Cell>;
    if (!disableRootInput) {
      specs.emplace_back(o2::emcal::getPublisherSpec<cellInputType>(PublisherConf{
                                                                      "emcal-cell-reader",
                                                                      "o2sim",
                                                                      "emccells.root",
                                                                      {"cellbranch", "EMCALCell", "Cell branch"},
                                                                      {"celltriggerbranch", "EMCALCellTRGR", "Trigger record branch"},
                                                                      {"mcbranch", "EMCALCellMCTruth", "MC label branch"},
                                                                      o2::framework::OutputSpec{"EMC", "CELLS"},
                                                                      o2::framework::OutputSpec{"EMC", "CELLSTRGR"},
                                                                      o2::framework::OutputSpec{"EMC", "CELLSMCTR"}},
                                                                    0, propagateMC));
    }
    if (enableDigitsPrinter) {
      try {
        specs.emplace_back(o2::emcal::reco_workflow::getEmcalDigitsPrinterSpec("cells"));
      } catch (std::runtime_error& e) {
        LOG(error) << "Cannot create digits printer spec: " << e.what();
      }
    }
  }

  if (isEnabled(OutputType::Cells)) {
    // add converter for cells
    if (inputType == InputType::Digits) {
      specs.emplace_back(o2::emcal::reco_workflow::getCellConverterSpec(propagateMC, useccdb, subspecificationIn, subspecificationOut));
    } else if (inputType == InputType::Raw) {
      // raw data will come from upstream
      specs.emplace_back(o2::emcal::reco_workflow::getRawToCellConverterSpec(askDISTSTF, disableDecodingErrors, subspecificationOut));
    }
  }

  if (isEnabled(OutputType::Clusters)) {
    // add clusterizer
    specs.emplace_back(o2::emcal::reco_workflow::getClusterizerSpec(inputType == InputType::Digits));
  }

  if (isEnabled(OutputType::AnalysisClusters)) {
    // add clusters from cells
    specs.emplace_back(o2::emcal::reco_workflow::getAnalysisClusterSpec(inputType == InputType::Digits));
  }

  auto makeWriterSpec = [propagateMC](const char* processName, const char* defaultFileName, const char* defaultTreeName,
                                      auto&& databranch, auto&& triggerbranch, auto&& mcbranch) {
    // depending on the MC propagation flag, the RootTreeWriter spec is created with two
    // or one branch definition
    if (propagateMC) {
      return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                             std::move(databranch),
                                                             std::move(triggerbranch),
                                                             std::move(mcbranch)));
    }
    return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                           std::move(databranch),
                                                           std::move(triggerbranch)));
  };

  // TODO: Write comment in push comment @matthiasrichter
  auto makeWriterSpec_Cluster = [](const char* processName, const char* defaultFileName, const char* defaultTreeName,
                                   auto&& clusterbranch, auto&& digitindicesbranch, auto&& clustertriggerbranch, auto&& indicestriggerbranch) {
    // RootTreeWriter spec is created with one branch definition
    return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                           std::move(clusterbranch),
                                                           std::move(digitindicesbranch),
                                                           std::move(clustertriggerbranch),
                                                           std::move(indicestriggerbranch)));
  };

  auto makeWriterSpec_AnalysisCluster = [](const char* processName, const char* defaultFileName, const char* defaultTreeName,
                                           auto&& analysisclusterbranch) {
    // RootTreeWriter spec is created with one branch definition
    return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                           std::move(analysisclusterbranch)));
  };

  auto makeWriterSpec_CellsTR = [disableDecodingErrors](const char* processName, const char* defaultFileName, const char* defaultTreeName,
                                                        auto&& CellsBranch, auto&& TriggerRecordBranch, auto&& DecoderErrorsBranch) {
    return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                           std::move(CellsBranch),
                                                           std::move(TriggerRecordBranch),
                                                           std::move(DecoderErrorsBranch)));
  };

  auto makeWriterSpec_CellsTR_noerrors = [](const char* processName, const char* defaultFileName, const char* defaultTreeName,
                                            auto&& CellsBranch, auto&& TriggerRecordBranch) {
    return std::move(o2::framework::MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                                           std::move(CellsBranch),
                                                           std::move(TriggerRecordBranch)));
  };
  /*
    // RS getting input digits and outputing them under the same outputspec will create dependency loop when piping the workflows
  if (isEnabled(OutputType::Digits) && !disableRootOutput) {
    using DigitOutputType = std::vector<o2::emcal::Digit>;
    using TriggerOutputType = std::vector<o2::emcal::TriggerRecord>;
    specs.push_back(makeWriterSpec("emcal-digits-writer",
                                   inputType == InputType::Digits ? "emc-filtered-digits.root" : "emcdigits.root",
                                   "o2sim",
                                   BranchDefinition<DigitOutputType>{o2::framework::InputSpec{"data", "EMC", "DIGITS", 0},
                                                                     "EMCDigit",
                                                                     "digit-branch-name"},
                                   BranchDefinition<TriggerOutputType>{o2::framework::InputSpec{"trigger", "EMC", "DIGITSTRGR", 0},
                                                                       "EMCALDigitTRGR",
                                                                       "digittrigger-branch-name"},
                                   BranchDefinition<o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>>{o2::framework::InputSpec{"mc", "EMC", "DIGITSMCTR", 0},
                                                                      "EMCDigitMCTruth",
                                                                      "digitmc-branch-name"})());
  }
  */
  if (isEnabled(OutputType::Cells) && !disableRootOutput) {
    if ((inputType == InputType::Digits) || (inputType == InputType::Cells)) {
      using DigitOutputType = std::vector<o2::emcal::Cell>;
      using TriggerOutputType = std::vector<o2::emcal::TriggerRecord>;
      specs.push_back(makeWriterSpec("emcal-cells-writer", "emccells.root", "o2sim",
                                     BranchDefinition<DigitOutputType>{o2::framework::InputSpec{"data", "EMC", "CELLS", 0},
                                                                       "EMCALCell",
                                                                       "cell-branch-name"},
                                     BranchDefinition<TriggerOutputType>{o2::framework::InputSpec{"trigger", "EMC", "CELLSTRGR", 0},
                                                                         "EMCALCellTRGR",
                                                                         "celltrigger-branch-name"},
                                     BranchDefinition<o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>>{o2::framework::InputSpec{"mc", "EMC", "CELLSMCTR", 0},
                                                                                                             "EMCALCellMCTruth",
                                                                                                             "cellmc-branch-name"})());
    } else {
      using CellsDataType = std::vector<o2::emcal::Cell>;
      using TriggerRecordDataType = std::vector<o2::emcal::TriggerRecord>;
      if (disableDecodingErrors) {
        specs.push_back(makeWriterSpec_CellsTR_noerrors("emcal-cells-writer",
                                                        "emccells.root",
                                                        "o2sim",
                                                        BranchDefinition<CellsDataType>{o2::framework::InputSpec{"data", "EMC", "CELLS", 0},
                                                                                        "EMCALCell",
                                                                                        "cell-branch-name"},
                                                        BranchDefinition<TriggerRecordDataType>{o2::framework::InputSpec{"trigger", "EMC", "CELLSTRGR", 0},
                                                                                                "EMCALCellTRGR",
                                                                                                "celltrigger-branch-name"})());

      } else {
        using DecoderErrorsDataType = std::vector<o2::emcal::ErrorTypeFEE>;
        specs.push_back(makeWriterSpec_CellsTR("emcal-cells-writer",
                                               "emccells.root",
                                               "o2sim",
                                               BranchDefinition<CellsDataType>{o2::framework::InputSpec{"data", "EMC", "CELLS", 0},
                                                                               "EMCALCell",
                                                                               "cell-branch-name"},
                                               BranchDefinition<TriggerRecordDataType>{o2::framework::InputSpec{"trigger", "EMC", "CELLSTRGR", 0},
                                                                                       "EMCALCellTRGR",
                                                                                       "celltrigger-branch-name"},
                                               BranchDefinition<DecoderErrorsDataType>{o2::framework::InputSpec{"errors", "EMC", "DECODERERR", 0},
                                                                                       "EMCALDECODERERR",
                                                                                       "decodererror-branch-name"})());
      }
    }
  }

  if (isEnabled(OutputType::Clusters) && !disableRootOutput) {
    using ClusterOutputType = std::vector<o2::emcal::Cluster>;
    using ClusterIndicesOutputType = std::vector<o2::emcal::ClusterIndex>;
    using TriggerOutputType = std::vector<o2::emcal::TriggerRecord>;

    specs.push_back(makeWriterSpec_Cluster("emcal-clusters-writer",
                                           "emcclusters.root",
                                           "o2sim",
                                           BranchDefinition<ClusterOutputType>{o2::framework::InputSpec{"clusters", "EMC", "CLUSTERS", 0},
                                                                               "EMCALCluster",
                                                                               "cluster-branch-name"},
                                           BranchDefinition<ClusterIndicesOutputType>{o2::framework::InputSpec{"clusterindices", "EMC", "INDICES", 0},
                                                                                      "EMCALClusterInputIndex",
                                                                                      "clusterdigitindices-branch-name"},
                                           BranchDefinition<TriggerOutputType>{o2::framework::InputSpec{"clusterTRGR", "EMC", "CLUSTERSTRGR", 0},
                                                                               "EMCALClusterTRGR",
                                                                               "clustertrigger-branch-name"},
                                           BranchDefinition<TriggerOutputType>{o2::framework::InputSpec{"indicesTRGR", "EMC", "INDICESTRGR", 0},
                                                                               "EMCIndicesTRGR",
                                                                               "indicestrigger-branch-name"})());
  }

  if (isEnabled(OutputType::AnalysisClusters) && !disableRootOutput) {
    using AnalysisClusterOutputType = std::vector<o2::emcal::AnalysisCluster>;

    specs.push_back(makeWriterSpec_AnalysisCluster("emcal-analysis-clusters-writer",
                                                   "emcAnalysisClusters.root",
                                                   "o2sim",
                                                   BranchDefinition<AnalysisClusterOutputType>{o2::framework::InputSpec{"analysisclusters", "EMC", "ANALYSISCLUSTERS", 0},
                                                                                               "EMCAnalysisCluster",
                                                                                               "cluster-branch-name"})());
  }

  return std::move(specs);
}

} // namespace reco_workflow

} // namespace emcal

} // namespace o2
