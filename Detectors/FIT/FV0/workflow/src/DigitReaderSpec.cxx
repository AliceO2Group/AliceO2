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
#include "FV0Workflow/DigitReaderSpec.h"
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("fv0-digit-infile"));
  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }
  mTree.reset((TTree*)mFile->Get("o2sim"));
  if (!mTree) {
    LOG(ERROR) << "Did not find o2sim tree in " << filename.c_str();
    throw std::runtime_error("Did not fine o2sim file in FV0 digits tree");
  }
}

void DigitReader::run(ProcessingContext& pc)
{

  std::vector<o2::fv0::BCData> digits, *pdigits = &digits;
  std::vector<o2::fv0::ChannelData> channels, *pchannels = &channels;
  mTree->SetBranchAddress("FV0DigitBC", &pdigits);
  mTree->SetBranchAddress("FV0DigitCh", &pchannels);

  o2::dataformats::MCTruthContainer<o2::fv0::MCLabel> labels, *plabels = &labels;
  if (mUseMC) {
    mTree->SetBranchAddress("FV0DigitLabels", &plabels);
  }
  mTree->GetEntry(0);

  LOG(INFO) << "FV0DigitReader pushed " << channels.size() << " channels in " << digits.size() << " digits";

  pc.outputs().snapshot(Output{"FV0", "DIGITSBC", 0, Lifetime::Timeframe}, digits);
  pc.outputs().snapshot(Output{"FV0", "DIGITSCH", 0, Lifetime::Timeframe}, channels);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"FV0", "DIGITSMCTR", 0, Lifetime::Timeframe}, labels);
  }
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("FV0", "DIGITSBC", 0, Lifetime::Timeframe);
  outputs.emplace_back("FV0", "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("FV0", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "fv0-digit-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC)},
    Options{
      {"fv0-digit-infile", VariantType::String, "fv0digits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace fv0
} // namespace o2
