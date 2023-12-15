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

/// @file   MatchedMFTMCHReaderSpec.cxx

#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/MatchInfoFwd.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "GlobalTrackingWorkflowReaders/MatchedMFTMCHReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Framework/SerializationMethods.h"
#include "CommonUtils/NameConf.h"
#include <fairlogger/Logger.h>

using namespace o2::framework;
using namespace o2::globaltracking;

namespace o2
{
namespace globaltracking
{

class MatchMFTMCHReader : public Task
{
 public:
  MatchMFTMCHReader(bool useMC) : mUseMC(useMC) {}
  ~MatchMFTMCHReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  bool mUseMC = true;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::vector<o2::dataformats::MatchInfoFwd> mTracks, *mTracksPtr = &mTracks;
  std::vector<o2::MCCompLabel> mLabels, *mLabelsPtr = &mLabels;
};

void MatchMFTMCHReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("mftmch-match-infile"));
  connectTree(mFileName);
}

void MatchMFTMCHReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mTracks.size() << " MFTMCH matches at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "MTC_MFTMCH", 0}, mTracks);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void MatchMFTMCHReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get("o2sim"));
  assert(mTree);
  mTree->SetBranchAddress("matches", &mTracksPtr);
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getMFTMCHMatchedReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "MTC_MFTMCH", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "mftmch-matches-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<MatchMFTMCHReader>(useMC)},
    Options{
      {"mftmch-match-infile", VariantType::String, "mftmchmatches.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace globaltracking
} // namespace o2
