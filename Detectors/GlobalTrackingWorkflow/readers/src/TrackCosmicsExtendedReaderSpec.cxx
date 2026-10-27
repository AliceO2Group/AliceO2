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

/// @file   TrackCosmicsExtendedReaderSpec.cxx

#include <vector>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "GlobalTrackingWorkflowReaders/TrackCosmicsExtendedReaderSpec.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;
using namespace o2::globaltracking;

namespace o2
{
namespace globaltracking
{
void TrackCosmicsExtendedReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("cosmics-full-infile"));
  connectTree(mFileName);
}

void TrackCosmicsExtendedReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  // Same as in TrackCosmicsReader: publish empty containers if the tree has no entry left
  const bool noEntry = ent >= mTree->GetEntries();
  if (noEntry) {
    LOG(info) << "no entry to read, publishing empty output";
    mCosmics.clear();
    mTFInfo = {};
    mTFID = {};
  } else {
    mTree->GetEntry(ent);
  }
  LOG(info) << "Pushing " << mCosmics.size() << " cosmics with clusters at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "COSMFULL", 0}, mCosmics);
  pc.outputs().snapshot(Output{"GLO", "COSMFULLTF", 0}, mTFInfo);
  pc.outputs().snapshot(Output{"GLO", "COSMFULLTFID", 0}, mTFID);

  if (noEntry || mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TrackCosmicsExtendedReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  if (!mFile || mFile->IsZombie()) {
    LOGP(fatal, "Failed to open {}", filename);
  }
  mTree.reset((TTree*)mFile->Get("cosmicsFull"));
  if (!mTree) {
    LOGP(fatal, "Tree cosmicsFull is not found in {}", filename);
  }
  mTree->SetBranchAddress("cosmics", &mCosmicsPtr);
  mTree->SetBranchAddress("tfInfo", &mTFInfoPtr);
  mTree->SetBranchAddress("tfID", &mTFIDPtr);
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getTrackCosmicsExtendedReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "COSMFULL", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "COSMFULLTF", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "COSMFULLTFID", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "cosmic-ext-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TrackCosmicsExtendedReader>()},
    Options{
      {"cosmics-full-infile", VariantType::String, "o2_cosmics_full.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace globaltracking
} // namespace o2
