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

/// @file   TrackTPCITSReaderSpec.cxx

#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Framework/SerializationMethods.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsITSMFT/TrkClusRef.h"

using namespace o2::framework;
using namespace o2::globaltracking;

namespace o2
{
namespace globaltracking
{

class TrackTPCITSReader : public Task
{
 public:
  TrackTPCITSReader(bool useMC) : mUseMC(useMC) {}
  ~TrackTPCITSReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  bool mUseMC = true;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::vector<o2::dataformats::TrackTPCITS> mTracks, *mTracksPtr = &mTracks;
  std::vector<o2::itsmft::TrkClusRef> mABTrkClusRefs, *mABTrkClusRefsPtr = &mABTrkClusRefs;
  std::vector<int> mABTrkClIDs, *mABTrkClIDsPtr = &mABTrkClIDs;
  std::vector<o2::MCCompLabel> mLabels, *mLabelsPtr = &mLabels;
  std::vector<o2::MCCompLabel> mLabelsAB, *mLabelsABPtr = &mLabelsAB;
};

void TrackTPCITSReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("itstpc-track-infile"));
  connectTree(mFileName);
}

void TrackTPCITSReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mTracks.size() << " TPC-ITS matches at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "TPCITS", 0}, mTracks);
  pc.outputs().snapshot(Output{"GLO", "TPCITSAB_REFS", 0}, mABTrkClusRefs);
  pc.outputs().snapshot(Output{"GLO", "TPCITSAB_CLID", 0}, mABTrkClIDs);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "TPCITS_MC", 0}, mLabels);
    pc.outputs().snapshot(Output{"GLO", "TPCITSAB_MC", 0}, mLabelsAB);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TrackTPCITSReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("matchTPCITS"));
  assert(mTree);
  mTree->SetBranchAddress("TPCITS", &mTracksPtr);
  mTree->SetBranchAddress("TPCITSABRefs", &mABTrkClusRefsPtr);
  mTree->SetBranchAddress("TPCITSABCLID", &mABTrkClIDsPtr);
  if (mUseMC) {
    mTree->SetBranchAddress("MatchMCTruth", &mLabelsPtr);
    mTree->SetBranchAddress("MatchABMCTruth", &mLabelsABPtr);
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getTrackTPCITSReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "TPCITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TPCITSAB_REFS", 0, Lifetime::Timeframe); // AftetBurner ITS tracklet references (referred by GlobalTrackID::ITSAB) on cluster indices
  outputs.emplace_back("GLO", "TPCITSAB_CLID", 0, Lifetime::Timeframe); // cluster indices of ITS tracklets attached by the AfterBurner
  if (useMC) {
    outputs.emplace_back("GLO", "TPCITS_MC", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "TPCITSAB_MC", 0, Lifetime::Timeframe); // AfterBurner ITS tracklet MC
  }

  return DataProcessorSpec{
    "itstpc-track-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TrackTPCITSReader>(useMC)},
    Options{
      {"itstpc-track-infile", VariantType::String, "o2match_itstpc.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace globaltracking
} // namespace o2
