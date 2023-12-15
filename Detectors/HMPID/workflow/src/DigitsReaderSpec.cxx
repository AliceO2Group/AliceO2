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

/// @file   DigitReaderSpec.cxx

#include <vector>
#include <TTree.h>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonUtils/NameConf.h"
#include "HMPIDWorkflow/DigitsReaderSpec.h"

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>

#include "CommonUtils/StringUtils.h" // o2::utils::Str

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "HMPIDBase/Geo.h"

using namespace o2::framework;
using namespace o2;
using namespace o2::header;

namespace o2
{
namespace hmpid
{

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("hmpid-digit-infile"));
  mFile.reset(TFile::Open(filename.c_str()));

  LOG(info) << "HMPID DigitWriterSpec::init() : Trying to read File : " << filename.c_str();

  mDigitsReceived = 0;
  if (!mFile->IsOpen()) {
    LOG(error) << "HMPID DigitWriterSpec::init() : Did not find any digits file " << filename.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }

  if ((TTree*)mFile->Get("o2sim") != nullptr) {
    mTree.reset((TTree*)mFile->Get("o2sim"));
  } else if ((TTree*)mFile->Get("o2hmp") != nullptr) {
    mTree.reset((TTree*)mFile->Get("o2hmp"));
  } else {
    LOG(error) << "Did not find o2hmp tree in " << filename.c_str();
    throw std::runtime_error("Did Not find Any correct Tree in HMPID Digits File");
  }

  if (!mTree) {
    LOG(error) << "Did not find o2hmp tree in " << filename.c_str();
    throw std::runtime_error("Did Not find Any correct Tree in HMPID Digits File");
  } /*else {
    LOG(info) << "HMPID DigitWriterSpec::init() : Reading From Branch  o2hmp" << File : " << filename.c_str()";
  } */
}

void DigitReader::run(ProcessingContext& pc)
{
  std::vector<o2::hmpid::Digit> mDigitsFromFile, *mDigitsFromFilePtr = &mDigitsFromFile;
  std::vector<o2::hmpid::Trigger> mTriggersFromFile, *mTriggersFromFilePtr = &mTriggersFromFile;

  /*  */
  if (mTree->GetBranchStatus("HMPDigit")) {
    mTree->SetBranchAddress("HMPDigit", &mDigitsFromFilePtr);
  } else if (mTree->GetBranchStatus("HMPIDDigits")) {
    mTree->SetBranchAddress("HMPIDDigits", &mDigitsFromFilePtr);
  } else {
    LOG(error)
      << "HMPID DigitWriterSpec::init() : Did not find any branch for Digits";
    throw std::runtime_error("Did Not find Any correct Branch for Digits in HMPID Digits File");
  }

  if (mTree->GetBranchStatus("InteractionRecords")) {
    mTree->SetBranchAddress("InteractionRecords", &mTriggersFromFilePtr);
  } else {
    LOG(error)
      << "HMPID DigitWriterSpec::init() : Did not find  branch for Triggers";
    throw std::runtime_error("Did Not find Branch For triggers in HMPID Digits File");
  }
  // mTree->Print("toponly");

  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);

  pc.outputs().snapshot(Output{"HMP", "DIGITS", 0}, mDigitsFromFile);
  pc.outputs().snapshot(Output{"HMP", "INTRECORDS", 0}, mTriggersFromFile);
  mDigitsReceived += mDigitsFromFile.size();
  LOG(info) << "[HMPID DigitsReader - run() ] digits  = " << mDigitsFromFile.size();

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mExTimer.stop();
    mExTimer.logMes("End DigitsReader !  digits = " +
                    std::to_string(mDigitsReceived));
  }
}

DataProcessorSpec getDigitsReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("HMP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("HMP", "INTRECORDS", 0, o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "HMP-DigitReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>()},
    Options{{"hmpid-digit-infile" /*"/qc-hmpid-digits"*/, VariantType::String, "hmpiddigits.root", {"Name of the input file with digits"}},
            {"input-dir", VariantType::String, "./", {"Input directory"}}}};
}

} // namespace hmpid
} // namespace o2
