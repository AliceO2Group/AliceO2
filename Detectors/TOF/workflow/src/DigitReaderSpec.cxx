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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "TOFWorkflowUtils/DigitReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

void DigitReader::init(InitContext& ic)
{
  LOG(INFO) << "Init Digit reader!";
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("tof-digit-infile"));
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
    treeDig->SetBranchAddress("TOFReadoutWindow", &mProw);
    treeDig->SetBranchAddress("TOFPatterns", &mPpatterns);

    if (mUseMC) {
      treeDig->SetBranchAddress("TOFDigitMCTruth", &mPlabels);
    }

    treeDig->GetEntry(mCurrentEntry);

    // add digits loaded in the output snapshot
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe}, mDigits);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe}, mRow);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe}, mPatterns);
    if (mUseMC) {
      pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe}, mLabels);
    }

    static o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::CONTINUOUS;

    LOG(DEBUG) << "TOF: Sending ROMode= " << roMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe}, roMode);
  } else {
    LOG(ERROR) << "Cannot read the TOF digits !";
    return;
  }

  mCurrentEntry++;

  if (mCurrentEntry >= treeDig->GetEntries()) {
    mState = 2;
    //pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    pc.services().get<ControlService>().endOfStream();
  }
}

DataProcessorSpec getDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back(o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tof-digit-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC)},
    Options{
      {"tof-digit-infile", VariantType::String, "tofdigits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace tof
} // namespace o2
