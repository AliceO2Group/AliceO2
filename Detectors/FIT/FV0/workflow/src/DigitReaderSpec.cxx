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
#include "FV0Workflow/DigitReaderSpec.h"
#include "DataFormatsFV0/Digit.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("fv0-digit-infile"));
  mFile.reset(TFile::Open(filename.c_str()));
  if (!mFile->IsOpen()) {
    LOG(error) << "Cannot open the " << filename.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }
  mTree.reset((TTree*)mFile->Get("o2sim"));
  if (!mTree) {
    LOG(error) << "Did not find o2sim tree in " << filename.c_str();
    throw std::runtime_error("Did not fine o2sim file in FV0 digits tree");
  }
}

void DigitReader::run(ProcessingContext& pc)
{

  std::vector<o2::fv0::Digit> digits, *pdigits = &digits;
  std::vector<o2::fv0::DetTrigInput> trgInput, *ptrTrgInput = &trgInput;
  std::vector<o2::fv0::ChannelData> channels, *pchannels = &channels;
  mTree->SetBranchAddress("FV0DigitBC", &pdigits);
  mTree->SetBranchAddress("FV0DigitCh", &pchannels);
  if (mUseTrgInput) {
    mTree->SetBranchAddress("TRIGGERINPUT", &ptrTrgInput);
  }
  o2::dataformats::MCTruthContainer<o2::fv0::MCLabel> labels, *plabels = &labels;
  if (mUseMC) {
    mTree->SetBranchAddress("FV0DigitLabels", &plabels);
  }
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(debug) << "FV0DigitReader pushed " << channels.size() << " channels in " << digits.size() << " digits";
  pc.outputs().snapshot(Output{"FV0", "DIGITSBC", 0, Lifetime::Timeframe}, digits);
  pc.outputs().snapshot(Output{"FV0", "DIGITSCH", 0, Lifetime::Timeframe}, channels);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"FV0", "DIGITSMCTR", 0, Lifetime::Timeframe}, labels);
  }
  if (mUseTrgInput) {
    pc.outputs().snapshot(Output{"FV0", "TRIGGERINPUT", 0, Lifetime::Timeframe}, trgInput);
  }
  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getDigitReaderSpec(bool useMC, bool useTrgInput)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("FV0", "DIGITSBC", 0, Lifetime::Timeframe);
  outputs.emplace_back("FV0", "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("FV0", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  if (useTrgInput) {
    outputs.emplace_back("FV0", "TRIGGERINPUT", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "fv0-digit-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC, useTrgInput)},
    Options{
      {"fv0-digit-infile", VariantType::String, "fv0digits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace fv0
} // namespace o2
