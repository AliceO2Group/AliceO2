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

#include "Framework/ControlService.h"
#include "TOFWorkflow/DigitReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

void DigitReader::init(InitContext& ic)
{
  LOG(INFO) << "Init Digit reader!";
  auto filename = ic.options().get<std::string>("tof-digit-infile");
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
    treeDig->SetBranchAddress("TOFDigit", &mPdigits);

    if (mUseMC) {
      treeDig->SetBranchAddress("TOFDigitMCTruth", &mPlabels);
    }

    treeDig->GetEntry(0);

    // add digits loaded in the output snapshot
    pc.outputs().snapshot(Output{"TOF", "DIGITS", 0, Lifetime::Timeframe}, mDigits);
    if (mUseMC)
      pc.outputs().snapshot(Output{"TOF", "DIGITSMCTR", 0, Lifetime::Timeframe}, mLabels);

    static o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::CONTINUOUS;

    LOG(INFO) << "TOF: Sending ROMode= " << roMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"TOF", "ROMode", 0, Lifetime::Timeframe}, roMode);
  } else {
    LOG(ERROR) << "Cannot read the TOF digits !";
    return;
  }

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TOF", "DIGITS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TOF", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("TOF", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tof-digit-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC)},
    Options{
      {"tof-digit-infile", VariantType::String, "tofdigits.root", {"Name of the input file"}}}};
}

} // namespace tof
} // namespace o2
