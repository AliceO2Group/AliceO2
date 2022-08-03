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

#include "TTree.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "ZDCWorkflow/DigitReaderSpec.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("zdc-digit-infile"));
  mFirstEntry = ic.options().get<int>("first-entry");
  mLastEntry = ic.options().get<int>("last-entry");
  mFile = std::make_unique<TFile>(filename.c_str());
  if (!mFile->IsOpen()) {
    LOG(error) << "Cannot open the " << filename.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }
  mTree.reset((TTree*)mFile->Get("o2sim"));
  if (!mTree) {
    LOG(error) << "Did not find o2sim tree in " << filename.c_str();
    throw std::runtime_error("Did not fine o2sim file in ZDC digits tree");
  }
}

void DigitReader::run(ProcessingContext& pc)
{

  std::vector<o2::zdc::OrbitData> zdcOrbitData, *zdcOrbitDataPtr = &zdcOrbitData;
  std::vector<o2::zdc::BCData> zdcBCData, *zdcBCDataPtr = &zdcBCData;
  std::vector<o2::zdc::ChannelData> zdcChData, *zdcChDataPtr = &zdcChData;

  mTree->SetBranchAddress("ZDCDigitOrbit", &zdcOrbitDataPtr);
  mTree->SetBranchAddress("ZDCDigitBC", &zdcBCDataPtr);
  mTree->SetBranchAddress("ZDCDigitCh", &zdcChDataPtr);
  o2::dataformats::MCTruthContainer<o2::zdc::MCLabel> labels, *plabels = &labels;
  if (mUseMC) {
    mTree->SetBranchAddress("ZDCDigitLabels", &plabels);
  }

  auto ent = mTree->GetReadEntry() < 0 ? mTree->GetReadEntry() + mFirstEntry + 1 : mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "ZDCDigitReader pushed " << zdcOrbitData.size() << " orbits with " << zdcBCData.size() << " bcs and " << zdcChData.size() << " digits";
  pc.outputs().snapshot(Output{"ZDC", "DIGITSPD", 0, Lifetime::Timeframe}, zdcOrbitData);
  pc.outputs().snapshot(Output{"ZDC", "DIGITSBC", 0, Lifetime::Timeframe}, zdcBCData);
  pc.outputs().snapshot(Output{"ZDC", "DIGITSCH", 0, Lifetime::Timeframe}, zdcChData);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"ZDC", "DIGITSLBL", 0, Lifetime::Timeframe}, labels);
  }
  uint64_t nextEntry = mTree->GetReadEntry() + 1;
  if (nextEntry >= mTree->GetEntries() || (mLastEntry >= 0 && nextEntry > mLastEntry)) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
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
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"first-entry", o2::framework::VariantType::Int, 0, {"First digit entry"}},
      {"last-entry", o2::framework::VariantType::Int, -1, {"Last digit entry"}}}};
}

} // namespace zdc
} // namespace o2
