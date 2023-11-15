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

/// @file   RecoReaderSpec.cxx

#include <vector>

#include <TTree.h>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "ZDCWorkflow/RecoReaderSpec.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

void RecoReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("zdc-reco-infile"));
  mFile.reset(TFile::Open(filename.c_str()));
  if (!mFile->IsOpen()) {
    LOG(error) << "Cannot open the " << filename.c_str() << " file !";
    throw std::runtime_error("cannot open input digits file");
  }
  mTree.reset((TTree*)mFile->Get("o2rec"));
  if (!mTree) {
    LOG(error) << "Did not find o2sim tree in " << filename.c_str();
    throw std::runtime_error("Did not find o2rec tree in ZDC reco file");
  }
}

void RecoReader::run(ProcessingContext& pc)
{

  std::vector<o2::zdc::BCRecData> RecBC, *RecBCPtr = &RecBC;
  std::vector<o2::zdc::ZDCEnergy> Energy, *EnergyPtr = &Energy;
  std::vector<o2::zdc::ZDCTDCData> TDCData, *TDCDataPtr = &TDCData;
  std::vector<uint16_t> Info, *InfoPtr = &Info;

  mTree->SetBranchAddress("ZDCRecBC", &RecBCPtr);
  mTree->SetBranchAddress("ZDCRecE", &EnergyPtr);
  mTree->SetBranchAddress("ZDCRecTDC", &TDCDataPtr);
  mTree->SetBranchAddress("ZDCRecInfo", &InfoPtr);

  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "ZDCRecoReader pushed " << RecBC.size() << " b.c. " << Energy.size() << " Energies " << TDCData.size() << " TDCs " << Info.size() << " Infos";
  pc.outputs().snapshot(Output{"ZDC", "BCREC", 0, Lifetime::Timeframe}, RecBC);
  pc.outputs().snapshot(Output{"ZDC", "ENERGY", 0, Lifetime::Timeframe}, Energy);
  pc.outputs().snapshot(Output{"ZDC", "TDCDATA", 0, Lifetime::Timeframe}, TDCData);
  pc.outputs().snapshot(Output{"ZDC", "INFO", 0, Lifetime::Timeframe}, Info);
  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getRecoReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "BCREC", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "ENERGY", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "INFO", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "zdc-reco-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<RecoReader>()},
    Options{
      {"zdc-reco-infile", VariantType::String, "zdcreco.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace zdc
} // namespace o2
