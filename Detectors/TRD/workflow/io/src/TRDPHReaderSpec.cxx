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

/// @file   TRDPHReaderSpec.cxx

#include "TRDWorkflowIO/TRDPHReaderSpec.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "fairlogger/Logger.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDPHReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(info) << "Init TRD PH reader!";
  mInFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("trd-ph-infile"));
  mInTreeName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("treename"));
  connectTree();
}

void TRDPHReader::connectTree()
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mInFileName.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mInTreeName.c_str()));
  assert(mTree);
  mTree->SetBranchAddress("values", &mPHValuesPtr);
  LOG(info) << "Loaded tree from " << mInFileName << " with " << mTree->GetEntries() << " entries";
}

void TRDPHReader::run(ProcessingContext& pc)
{
  auto currEntry = mTree->GetReadEntry() + 1;
  assert(currEntry < mTree->GetEntries());
  mTree->GetEntry(currEntry);

  LOG(info) << "Pushing vector of PH values filled with " << mPHValues.size() << " entries at tree entry " << currEntry;
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "PULSEHEIGHT", 0}, mPHValues);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getTRDPHReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "PULSEHEIGHT", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TRDPHReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TRDPHReader>()},
    Options{
      {"trd-ph-infile", VariantType::String, "trd_PH.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "ph", {"Name of top-level TTree"}},
    }};
}

} // namespace trd
} // namespace o2
