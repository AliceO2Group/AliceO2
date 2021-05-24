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
#include "Framework/Logger.h"
#include "FT0Workflow/DigitReaderSpec.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("ft0-digit-infile"));
  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }
  mTree.reset((TTree*)mFile->Get("o2sim"));
  if (!mTree) {
    LOG(ERROR) << "Did not find o2sim tree in " << filename.c_str();
    throw std::runtime_error("Did not fine o2sim file in FT0 digits tree");
  }
}

void DigitReader::run(ProcessingContext& pc)
{

  std::vector<o2::ft0::Digit> digits, *pdigits = &digits;
  std::vector<o2::ft0::DetTrigInput> trgInput, *ptrTrgInput = &trgInput;
  std::vector<o2::ft0::ChannelData> channels, *pchannels = &channels;
  mTree->SetBranchAddress("FT0DIGITSBC", &pdigits);
  mTree->SetBranchAddress("FT0DIGITSCH", &pchannels);
  if (mUseTrgInput) {
    mTree->SetBranchAddress("TRIGGERINPUT", &ptrTrgInput);
  }
  o2::dataformats::MCTruthContainer<o2::ft0::MCLabel> labels, *plabels = &labels;
  if (mUseMC) {
    mTree->SetBranchAddress("FT0DIGITSMCTR", &plabels);
  }
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "FT0DigitReader pushed " << channels.size() << " channels in " << digits.size() << " digits";
  pc.outputs().snapshot(Output{"FT0", "DIGITSBC", 0, Lifetime::Timeframe}, digits);
  pc.outputs().snapshot(Output{"FT0", "DIGITSCH", 0, Lifetime::Timeframe}, channels);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"FT0", "DIGITSMCTR", 0, Lifetime::Timeframe}, labels);
  }
  if (mUseTrgInput) {
    pc.outputs().snapshot(Output{"FT0", "TRIGGERINPUT", 0, Lifetime::Timeframe}, trgInput);
  }
  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getDigitReaderSpec(bool useMC, bool useTrgInput)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("FT0", "DIGITSBC", 0, Lifetime::Timeframe);
  outputs.emplace_back("FT0", "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("FT0", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  if (useTrgInput) {
    outputs.emplace_back("FT0", "TRIGGERINPUT", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "ft0-digit-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC, useTrgInput)},
    Options{
      {"ft0-digit-infile", VariantType::String, "ft0digits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace ft0
} // namespace o2
