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

/// @file   FT0DigitWriterSpec.cxx
#include "FT0Workflow/FT0DigitWriterSpec.h"

namespace o2
{
namespace ft0
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getFT0DigitWriterSpec(bool mctruth, bool trigInp)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  // Spectators for logging
  auto logger = [](std::vector<o2::ft0::Digit> const& vecDigits) {
    LOG(debug) << "FT0DigitWriter pulled " << vecDigits.size() << " digits";
  };
  // the callback to be set as hook for custom action when the writer is closed
  auto finishWriting = [](TFile* outputfile, TTree* outputtree) {
    const auto* brArr = outputtree->GetListOfBranches();
    int64_t nent = 0;
    for (const auto* brc : *brArr) {
      int64_t n = ((const TBranch*)brc)->GetEntries();
      if (nent && (nent != n)) {
        LOG(error) << "Branches have different number of entries";
      }
      nent = n;
    }
    outputtree->SetEntries(nent);
    outputtree->Write();
    outputfile->Close();
  };

  auto labelsdef = BranchDefinition<o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>>{InputSpec{"labelinput", "FT0", "DIGITSMCTR"},
                                                                                         "FT0DIGITSMCTR", mctruth ? 1 : 0};
  if (trigInp) {
    return MakeRootTreeWriterSpec("FT0DigitWriter",
                                  "ft0digits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::ft0::Digit>>{InputSpec{"ft0digitBCinput", "FT0", "DIGITSBC"}, "FT0DIGITSBC", "ft0-digits-branch-name", 1,
                                                                                logger},
                                  BranchDefinition<std::vector<o2::ft0::ChannelData>>{InputSpec{"ft0digitChinput", "FT0", "DIGITSCH"}, "FT0DIGITSCH", "ft0-chhdata-branch-name"},
                                  BranchDefinition<std::vector<o2::ft0::DetTrigInput>>{InputSpec{"ft0digitTrinput", "FT0", "TRIGGERINPUT"}, "TRIGGERINPUT", "ft0-triggerinput-branch-name"},
                                  std::move(labelsdef))();
  } else {
    return MakeRootTreeWriterSpec("FT0DigitWriterRaw",
                                  "o2_ft0digits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::ft0::Digit>>{InputSpec{"ft0digitBCinput", "FT0", "DIGITSBC"}, "FT0DIGITSBC", "ft0-digits-branch-name", 1,
                                                                                logger},
                                  BranchDefinition<std::vector<o2::ft0::ChannelData>>{InputSpec{"ft0digitChinput", "FT0", "DIGITSCH"}, "FT0DIGITSCH", "ft0-chhdata-branch-name"},
                                  std::move(labelsdef))();
  }
}

} // end namespace ft0
} // end namespace o2
