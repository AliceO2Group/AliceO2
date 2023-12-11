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

/// @file   RecPointReaderSpec.cxx

#include <vector>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "FT0Workflow/RecPointReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::ft0;

namespace o2
{
namespace ft0
{

RecPointReader::RecPointReader(bool useMC)
{
  mUseMC = useMC;
  if (useMC) {
    LOG(warning) << "FT0 RecPoint reader at the moment does not process MC";
  }
}

void RecPointReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("infile"));
  connectTree(mInputFileName);
}

void RecPointReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);

  LOG(debug) << "FT0 RecPointReader pushes " << mRecPoints->size() << " recpoints with " << mChannelData->size() << " channels at entry " << ent;
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0}, *mRecPoints);
  pc.outputs().snapshot(Output{mOrigin, "RECCHDATA", 0}, *mChannelData);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void RecPointReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mRecPointTreeName.c_str()));
  assert(mTree);

  mTree->SetBranchAddress(mRecPointBranchName.c_str(), &mRecPoints);
  mTree->SetBranchAddress(mChannelDataBranchName.c_str(), &mChannelData);
  if (mUseMC) {
    LOG(warning) << "MC-truth is not supported for FT0 recpoints currently";
    mUseMC = false;
  }

  LOG(info) << "Loaded FT0 RecPoints tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getRecPointReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "RECCHDATA", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(warning) << "MC-truth is not supported for FT0 recpoints currently";
  }

  return DataProcessorSpec{
    "ft0-recpoints-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<RecPointReader>()},
    Options{
      {"infile", VariantType::String, "o2reco_ft0.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace ft0
} // namespace o2
