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
#include "ZDCWorkflow/DigitReaderSpec.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("zdc-digit-infile"));
  mFile = std::make_unique<TFile>(filename.c_str());
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }
  mTree.reset((TTree*)mFile->Get("o2sim"));
  if (!mTree) {
    LOG(ERROR) << "Did not find o2sim tree in " << filename.c_str();
    throw std::runtime_error("Did not fine o2sim file in ZDC digits tree");
  }
}

void DigitReader::run(ProcessingContext& pc)
{
  std::vector<o2::zdc::ChannelData> digitsCh, *digitsChPtr = &digitsCh;
  std::vector<o2::zdc::BCData> digitsBC, *digitsBCPtr = &digitsBC;
  std::vector<o2::zdc::OrbitData> orbitData, *orbitDataPtr = &orbitData;
  mTree->SetBranchAddress("ZDCDigitBC", &digitsBCPtr);
  mTree->SetBranchAddress("ZDCDigitCh", &digitsChPtr);
  mTree->SetBranchAddress("ZDCDigitOrbit", &orbitDataPtr);

  o2::dataformats::MCTruthContainer<o2::zdc::MCLabel> labels, *plabels = &labels;
  if (mUseMC) {
    mTree->SetBranchAddress("ZDCDigitLabels", &plabels);
  }
  mTree->GetEntry(0);

  LOG(INFO) << "ZDCDigitReader pushed " << digitsCh.size() << " channels in " << digitsBC.size() << " digits";

  pc.outputs().snapshot(Output{"ZDC", "DIGITSBC", 0, Lifetime::Timeframe}, digitsBC);
  pc.outputs().snapshot(Output{"ZDC", "DIGITSCH", 0, Lifetime::Timeframe}, digitsCh);
  pc.outputs().snapshot(Output{"ZDC", "DIGITSPD", 0, Lifetime::Timeframe}, orbitData);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"ZDC", "DIGITSLBL", 0, Lifetime::Timeframe}, labels);
  }
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "DIGITSPD", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("ZDC", "DIGITSLBL", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "zdc-digit-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC)},
    Options{
      {"zdc-digit-infile", VariantType::String, "zdcdigits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace zdc
} // namespace o2
