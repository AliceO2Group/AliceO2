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

#include "TTree.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "EC0Workflow/DigitReaderSpec.h"
#include "DataFormatsITSMFT/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

using namespace o2::framework;
using namespace o2::endcaps;

namespace o2
{
namespace ecl
{

void DigitReader::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("ecl-digit-infile");
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
  if (mState != 1) {
    return;
  }

  std::unique_ptr<TTree> treeDig((TTree*)mFile->Get("o2sim"));

  if (treeDig) {

    std::vector<o2::itsmft::Digit> digits, *pdigits = &digits;
    treeDig->SetBranchAddress("EC0Digit", &pdigits);

    std::vector<ROFRecord> rofs, *profs = &rofs;
    treeDig->SetBranchAddress("EC0DigitROF", &profs);

    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels, *plabels = &labels;
    std::vector<MC2ROFRecord> mc2rofs, *pmc2rofs = &mc2rofs;
    if (mUseMC) {
      treeDig->SetBranchAddress("EC0DigitMCTruth", &plabels);
      treeDig->SetBranchAddress("EC0DigitMC2ROF", &pmc2rofs);
    }
    treeDig->GetEntry(0);

    LOG(INFO) << "EC0DigitReader pushed " << digits.size() << " digits, in "
              << profs->size() << " RO frames";

    pc.outputs().snapshot(Output{"EC0", "DIGITS", 0, Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(Output{"EC0", "DIGITSROF", 0, Lifetime::Timeframe}, *profs);
    if (mUseMC) {
      pc.outputs().snapshot(Output{"EC0", "DIGITSMCTR", 0, Lifetime::Timeframe}, labels);
      pc.outputs().snapshot(Output{"EC0", "DIGITSMC2ROF", 0, Lifetime::Timeframe}, *pmc2rofs);
    }
  } else {
    LOG(ERROR) << "Cannot read the EC0 digits !";
    return;
  }
  mState = 2;
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("EC0", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("EC0", "DIGITSROF", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("EC0", "DIGITSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("EC0", "DIGITSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "ecl-digit-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC)},
    Options{
      {"ecl-digit-infile", VariantType::String, "ecldigits.root", {"Name of the input file"}}}};
}

} // namespace ecl
} // namespace o2
