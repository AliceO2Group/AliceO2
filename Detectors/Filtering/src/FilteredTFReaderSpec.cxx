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

/// @file   FilteredTFReaderSpec.cxx

#include <cassert>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "FilteredTFReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::dataformats;

namespace o2::filtering
{

FilteredTFReader::FilteredTFReader(bool useMC)
{
  mUseMC = useMC;
}

void FilteredTFReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("filtered-tf-infile"));
  connectTree(mInputFileName);
}

void FilteredTFReader::run(ProcessingContext& pc)
{
  // FIXME: fill all output headers by TF specific info (extend findMessageHeaderStack)

  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);

  LOG(info) << "Pushing filtered TF: " << mFiltTF.header.asString();
  // ITS
  pc.outputs().snapshot(Output{"ITS", "ITSTrackROF", 0}, mFiltTF.ITSTrackROFs);
  pc.outputs().snapshot(Output{"ITS", "TRACKS", 0}, mFiltTF.ITSTracks);
  pc.outputs().snapshot(Output{"ITS", "TRACKCLSID", 0}, mFiltTF.ITSClusterIndices);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0}, mFiltTF.ITSTrackMCTruth);
  }
  pc.outputs().snapshot(Output{"ITS", "CLUSTERSROF", 0}, mFiltTF.ITSClusterROFs);
  pc.outputs().snapshot(Output{"ITS", "COMPCLUSTERS", 0}, mFiltTF.ITSClusters);
  pc.outputs().snapshot(Output{"ITS", "PATTERNS", 0}, mFiltTF.ITSClusterPatterns);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void FilteredTFReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mInputTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mFTFBranchName.c_str()));
  mTree->SetBranchAddress(mFTFBranchName.c_str(), &mFiltTFPtr);
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getFilteredTFReaderSpec(bool useMC)
{

  std::vector<OutputSpec> outputSpec;
  // same as ITSWorkflow/TrackReaderSpec
  outputSpec.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
  }
  // same as ITSMFTWorkflow/ClusterReaderSpec
  outputSpec.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "PATTERNS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "filtered-reco-tf-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<FilteredTFReader>(useMC)},
    Options{
      {"filtered-tf-infile", VariantType::String, "o2_filtered_tf.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace o2::filtering
