// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.cxx

#include <vector>

#include "MFTWorkflow/DigitReaderSpec.h"

#include "TTree.h"
#include "Framework/ControlService.h"
#include "ITSMFTBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace mft
{

void DigitReader::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("mft-digit-infile");
  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    mState = 0;
    return;
  }
  mState = 1;
}

void DigitReader::run(ProcessingContext& pc)
{
  if (mState != 1)
    return;

  std::unique_ptr<TTree> treeDig((TTree*)mFile->Get("o2sim"));
  std::unique_ptr<TTree> treeROF((TTree*)mFile->Get("MFTDigitROF"));
  std::unique_ptr<TTree> treeMC2ROF((TTree*)mFile->Get("MFTDigitMC2ROF"));

  if (treeDig && treeROF && treeMC2ROF) {

    std::vector<o2::itsmft::Digit> allDigits;
    std::vector<o2::itsmft::Digit> digits, *pdigits = &digits;
    treeDig->SetBranchAddress("MFTDigit", &pdigits);

    o2::dataformats::MCTruthContainer<o2::MCCompLabel> allLabels;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels, *plabels = &labels;
    treeDig->SetBranchAddress("MFTDigitMCTruth", &plabels);

    std::vector<ROFRecord> rofs, *profs = &rofs;
    treeROF->SetBranchAddress("MFTDigitROF", &profs);
    treeROF->GetEntry(0);

    std::vector<MC2ROFRecord> mc2rofs, *pmc2rofs = &mc2rofs;
    treeMC2ROF->SetBranchAddress("MFTDigitMC2ROF", &pmc2rofs);
    treeMC2ROF->GetEntry(0);

    Int_t ne = treeDig->GetEntries();
    for (Int_t e = 0; e < ne; e++) {
      treeDig->GetEntry(e);
      std::copy(digits.begin(), digits.end(), std::back_inserter(allDigits));
      allLabels.mergeAtBack(labels);
    }
    LOG(INFO) << "MFTDigitReader pushed " << allDigits.size() << " digits, in "
              << profs->size() << " RO frames and "
              << pmc2rofs->size() << " MC events";
    pc.outputs().snapshot(Output{"MFT", "DIGITS", 0, Lifetime::Timeframe}, allDigits);
    pc.outputs().snapshot(Output{"MFT", "DIGITSMCTR", 0, Lifetime::Timeframe}, allLabels);
    pc.outputs().snapshot(Output{"MFT", "MFTDigitROF", 0, Lifetime::Timeframe}, *profs);
    pc.outputs().snapshot(Output{"MFT", "MFTDigitMC2ROF", 0, Lifetime::Timeframe}, *pmc2rofs);
  } else {
    LOG(ERROR) << "Cannot read the MFT digits !";
    return;
  }
  mState = 2;
  //pc.services().get<ControlService>().readyToQuit(true);
}

DataProcessorSpec getDigitReaderSpec()
{
  return DataProcessorSpec{
    "mft-digit-reader",
    Inputs{},
    Outputs{
      OutputSpec{"MFT", "DIGITS", 0, Lifetime::Timeframe},
      OutputSpec{"MFT", "DIGITSMCTR", 0, Lifetime::Timeframe},
      OutputSpec{"MFT", "MFTDigitROF", 0, Lifetime::Timeframe},
      OutputSpec{"MFT", "MFTDigitMC2ROF", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DigitReader>()},
    Options{
      {"mft-digit-infile", VariantType::String, "mftdigits.root", {"Name of the input file"}}}};
}

} // namespace mft
} // namespace o2
