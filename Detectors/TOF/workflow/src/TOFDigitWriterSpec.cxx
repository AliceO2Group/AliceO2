// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Processor spec for a ROOT file writer for TOF digits

#include "TOFWorkflow/TOFDigitWriterSpec.h"
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
using LabelsType = std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>;
using namespace o2::header;

DataProcessorSpec getTOFDigitWriterSpec(bool useMC)
{
  auto nCalls = std::make_shared<int>();
  *nCalls = 0;
  // the callback to be set as hook at stop of processing for the framework
  auto finishWriting = [nCalls](TFile* outputfile, TTree* outputtree) {
    printf("finish writing with %d entries in the tree\n", *nCalls);
    outputtree->SetEntries(*nCalls);
    outputtree->Write();
    outputfile->Close();
  };
  // preprocessor callback
  // read the trigger data first and store in the trigP2Sect shared pointer
  auto preprocessor = [nCalls](ProcessingContext&) {
    (*nCalls)++;
  };
  auto logger = [nCalls](OutputType const& indata) {
    LOG(INFO) << "Call " << *nCalls;
    LOG(INFO) << "RECEIVED DIGITS SIZE " << indata.size();
  };
  auto loggerROW = [nCalls](ReadoutWinType const& row) {
    LOG(INFO) << "RECEIVED READOUT WINDOWS " << row.size();
  };
  return MakeRootTreeWriterSpec("TOFDigitWriter",
                                "tofdigits.root",
                                "o2sim",
                                // the preprocessor only increments the call count, we keep this functionality
                                // of the original implementation
                                MakeRootTreeWriterSpec::Preprocessor{preprocessor},
                                BranchDefinition<OutputType>{InputSpec{"digits", gDataOriginTOF, "DIGITS", 0},
                                                             "TOFDigit",
                                                             "tofdigits-branch-name",
                                                             1,
                                                             logger},
                                BranchDefinition<ReadoutWinType>{InputSpec{"rowindow", gDataOriginTOF, "READOUTWINDOW", 0},
                                                                 "TOFReadoutWindow",
                                                                 "rowindow-branch-name",
                                                                 1,
                                                                 loggerROW},
                                BranchDefinition<LabelsType>{InputSpec{"labels", gDataOriginTOF, "DIGITSMCTR", 0},
                                                             "TOFDigitMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "digitlabels-branch-name"})();
}
} // end namespace tof
} // end namespace o2
