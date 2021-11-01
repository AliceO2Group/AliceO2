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

/// @file   FDDDigitWriterSpec.cxx
#include "FDDWorkflow/FDDDigitWriterSpec.h"

namespace o2
{
namespace fdd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getFDDDigitWriterSpec(bool mctruth, bool trigInp)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  // Spectators for logging
  auto logger = [](std::vector<o2::fdd::Digit> const& vecDigits) {
    LOG(debug) << "FDDDigitWriter pulled " << vecDigits.size() << " digits";
  };
  // the callback to be set as hook for custom action when the writer is closed
  auto finishWriting = [](TFile* outputfile, TTree* outputtree) {
    const auto* brArr = outputtree->GetListOfBranches();
    int64_t nent = 0;
    for (const auto* brc : *brArr) {
      int64_t n = ((const TBranch*)brc)->GetEntries();
      if (nent && (nent != n)) {
        LOG(ERROR) << "Branches have different number of entries";
      }
      nent = n;
    }
    outputtree->SetEntries(nent);
    outputtree->Write();
    outputfile->Close();
  };

  auto labelsdef = BranchDefinition<o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>>{InputSpec{"labelinput", "FDD", "DIGITSMCTR"},
                                                                                         "FDDDIGITSMCTR", mctruth ? 1 : 0};
  if (trigInp) {
    return MakeRootTreeWriterSpec("FDDDigitWriter",
                                  "fdddigits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::fdd::Digit>>{InputSpec{"digitBCinput", "FDD", "DIGITSBC"}, "FDDDigitBC", "fdd-digits-branch-name", 1,
                                                                                logger},
                                  BranchDefinition<std::vector<o2::fdd::ChannelData>>{InputSpec{"digitChinput", "FDD", "DIGITSCH"}, "FDDDigitCh", "fdd-chhdata-branch-name"},
                                  BranchDefinition<std::vector<o2::fdd::DetTrigInput>>{InputSpec{"digitTrinput", "FDD", "TRIGGERINPUT"}, "TRIGGERINPUT", "fdd-triggerinput-branch-name"},
                                  std::move(labelsdef))();
  } else {
    return MakeRootTreeWriterSpec("FDDDigitWriterRaw",
                                  "o2_fdddigits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::fdd::Digit>>{InputSpec{"digitBCinput", "FDD", "DIGITSBC"}, "FDDDigitBC", "fdd-digits-branch-name", 1,
                                                                                logger},
                                  BranchDefinition<std::vector<o2::fdd::ChannelData>>{InputSpec{"digitChinput", "FDD", "DIGITSCH"}, "FDDDigitCh", "fdd-chhdata-branch-name"},
                                  std::move(labelsdef))();
  }
}

} // end namespace fdd
} // end namespace o2
