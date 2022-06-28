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

/// @brief  Processor spec for a ROOT file writer for TOF digits

#include "TOFWorkflowIO/TOFDigitWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TOFBase/Digit.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using OutputType = std::vector<o2::tof::Digit>;
using ReadoutWinType = std::vector<o2::tof::ReadoutWindowData>;
using PatternType = std::vector<uint8_t>;
using ErrorType = std::vector<uint64_t>;
using LabelsType = std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>;
using HeaderType = o2::tof::DigitHeader;
using namespace o2::header;

DataProcessorSpec getTOFDigitWriterSpec(bool useMC, bool writeErr)
{
  auto nCalls = std::make_shared<int>();
  *nCalls = 0;
  // the callback to be set as hook at stop of processing for the framework
  auto finishWriting = [nCalls](TFile* outputfile, TTree* outputtree) {
    printf("TOF finish writing with %d entries in the tree\n", *nCalls);
    outputtree->SetEntries(*nCalls);
    outputfile->Write();
    outputfile->Close();
  };
  // preprocessor callback
  // read the trigger data first and store in the trigP2Sect shared pointer
  auto preprocessor = [nCalls](ProcessingContext&) {
    (*nCalls)++;
  };
  auto loggerH = [nCalls](HeaderType const& indata) {
  };
  auto logger = [nCalls](OutputType const& indata) {
    //    LOG(info) << "TOF: RECEIVED DIGITS SIZE " << indata.size();
  };
  auto loggerROW = [nCalls](ReadoutWinType const& row) {
    //    LOG(info) << "TOF: RECEIVED READOUT WINDOWS " << row.size();
  };
  auto loggerPatterns = [nCalls](PatternType const& patterns) {
    //    LOG(info) << "TOF: RECEIVED PATTERNS " << patterns.size();
  };
  auto loggerErrors = [nCalls](ErrorType const& errors) {
    //    LOG(info) << "TOF: Error logger ";
  };
  return MakeRootTreeWriterSpec("TOFDigitWriter",
                                "tofdigits.root",
                                "o2sim",
                                // the preprocessor only increments the call count, we keep this functionality
                                // of the original implementation
                                MakeRootTreeWriterSpec::Preprocessor{preprocessor},
                                BranchDefinition<HeaderType>{InputSpec{"tofdigitheader", gDataOriginTOF, "DIGITHEADER", 0},
                                                             "TOFHeader",
                                                             "tofdigitheader-branch-name",
                                                             1,
                                                             loggerH},
                                BranchDefinition<OutputType>{InputSpec{"tofdigits", gDataOriginTOF, "DIGITS", 0},
                                                             "TOFDigit",
                                                             "tofdigits-branch-name",
                                                             1,
                                                             logger},
                                BranchDefinition<ReadoutWinType>{InputSpec{"tofrowindow", gDataOriginTOF, "READOUTWINDOW", 0},
                                                                 "TOFReadoutWindow",
                                                                 "rowindow-branch-name",
                                                                 1,
                                                                 loggerROW},
                                BranchDefinition<PatternType>{InputSpec{"tofpatterns", gDataOriginTOF, "PATTERNS", 0},
                                                              "TOFPatterns",
                                                              "patterns-branch-name",
                                                              1,
                                                              loggerPatterns},
                                BranchDefinition<ErrorType>{InputSpec{"toferrors", gDataOriginTOF, "ERRORS", 0},
                                                            "TOFErrors",
                                                            "errors-branch-name",
                                                            (writeErr ? 1 : 0), // one branch if mc labels enabled
                                                            loggerErrors},
                                BranchDefinition<LabelsType>{InputSpec{"toflabels", gDataOriginTOF, "DIGITSMCTR", 0},
                                                             "TOFDigitMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "digitlabels-branch-name"})();
}
} // end namespace tof
} // end namespace o2
