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

#ifndef O2_FDD_DIGITWRITERSPEC_H
#define O2_FDD_DIGITWRITERSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/MCLabel.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace fdd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getFDDDigitWriterSpec(bool mctruth = true, bool trigInp = true)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;

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
    outputfile->Write();
    outputfile->Close();
  };

  // custom handler for labels:
  // essentially transform the input container (as registered in the original branch definition) to the special output format for labels
  auto customlabelhandler = [](TBranch& branch, std::vector<char> const& labeldata, framework::DataRef const& ref) {
    o2::dataformats::ConstMCTruthContainerView<o2::fdd::MCLabel> labels(labeldata);
    // make the actual output object by adopting/casting the buffer
    // into a split format
    o2::dataformats::IOMCTruthContainerView outputcontainer(labeldata);
    auto ptr = &outputcontainer;
    auto br = framework::RootTreeWriter::remapBranch(branch, &ptr);
    br->Fill();
    br->ResetAddress();
  };

  auto labelsdef = BranchDefinition<std::vector<char>>{InputSpec{"FDDlabelinput", "FDD", "DIGITLBL"},
                                                       "FDDDigitLabels", "labels-branch-name",
                                                       // this branch definition is disabled if MC labels are not processed
                                                       (mctruth ? 1 : 0),
                                                       customlabelhandler};
  if (trigInp) {
    return MakeRootTreeWriterSpec("FDDDigitWriter",
                                  "fdddigits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::fdd::Digit>>{InputSpec{"FDDdigitBCinput", "FDD", "DIGITSBC"}, "FDDDigit"},
                                  BranchDefinition<std::vector<o2::fdd::ChannelData>>{InputSpec{"FDDdigitChinput", "FDD", "DIGITSCH"}, "FDDDigitCh"},
                                  BranchDefinition<std::vector<o2::fdd::DetTrigInput>>{InputSpec{"FDDdigitTrinput", "FDD", "TRIGGERINPUT"}, "TRIGGERINPUT"},
                                  std::move(labelsdef))();
  } else {
    return MakeRootTreeWriterSpec("FDDDigitWriterRaw",
                                  "o2_fdddigits.root",
                                  "o2sim",
                                  MakeRootTreeWriterSpec::CustomClose(finishWriting),
                                  BranchDefinition<std::vector<o2::fdd::Digit>>{InputSpec{"FDDdigitBCinput", "FDD", "DIGITSBC"}, "FDDDigit"},
                                  BranchDefinition<std::vector<o2::fdd::ChannelData>>{InputSpec{"FDDdigitChinput", "FDD", "DIGITSCH"}, "FDDDigitCh"},
                                  std::move(labelsdef))();
  }
}

} // end namespace fdd
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_FDDDIGITWRITERSPEC_H_ */
