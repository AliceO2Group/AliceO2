// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CTPWorkflow/CTPDigitWriterSpec.h"

namespace o2
{
namespace ctp
{
template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

framework::DataProcessorSpec getCTPDigitWriterSpec(bool raw)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  // Spectators for logging
  auto logger = [](std::vector<o2::ft0::Digit> const& vecDigits) {
    LOG(INFO) << "FT0DigitWriter pulled " << vecDigits.size() << " digits";
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
  if (raw) {
    return MakeRootTreeWriterSpec("CTPDigitWriter",
                                  "o2_ctpdigits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::ctp::CTPInputDigit>>{InputSpec{"digitBC", "CTP", "DIGITSBC"}, "CTPDIGITSBC", "ctp-digits-branch-name", 1})();
  } else {
    return MakeRootTreeWriterSpec("CTPDigitWriter",
                                  "ctpdigits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::ctp::CTPInputDigit>>{InputSpec{"digitBC", "CTP", "DIGITSBC"}, "CTPDIGITSBC", "ctp-digits-branch-name", 1})();
  }
}

} // namespace ctp
} // namespace o2
