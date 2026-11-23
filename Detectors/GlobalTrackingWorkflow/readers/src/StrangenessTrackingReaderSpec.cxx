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

/// @file   StrangenessTrackingReaderSpec.cxx

#include "Framework/DeviceSpec.h"
#include "Framework/TimingInfo.h"
#include "GlobalTrackingWorkflowReaders/StrangenessTrackingReaderSpec.h"

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/StrangenessTrackingStats.h"
#include "ReconstructionDataFormats/StrangeTrack.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TFile.h"
#include "TTree.h"

using namespace o2::framework;

namespace o2
{
namespace strangeness_tracking
{
// read strangeness tracking candidates
class StrangenessTrackingReader : public o2::framework::Task
{
  using StrangeTrack = dataformats::StrangeTrack;

 public:
  StrangenessTrackingReader(bool useMC, bool extensions, bool shortTracks) : mUseMC(useMC), mExtensions(extensions), mShortTracks(shortTracks) {}
  ~StrangenessTrackingReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree();

  bool mUseMC = false;
  bool mExtensions = false, mShortTracks = false;

  std::vector<StrangeTrack> mStrangeTrack, *mStrangeTrackPtr = &mStrangeTrack;
  std::vector<o2::MCCompLabel> mStrangeTrackMC, *mStrangeTrackMCPtr = &mStrangeTrackMC;
  std::vector<StrangeTrack> mExtended, *mExtendedPtr = &mExtended;
  std::vector<o2::MCCompLabel> mExtendedLabels, *mExtendedLabelsPtr = &mExtendedLabels, mShortLabels, *mShortLabelsPtr = &mShortLabels;
  std::vector<o2::its::TimeStamp> mTimes, *mTimesPtr = &mTimes;
  std::vector<o2::its::TrackITS> mShort, *mShortPtr = &mShort;
  std::vector<int> mShortPV, *mShortPVPtr = &mShortPV, mShortClus, *mShortClusPtr = &mShortClus;
  o2::dataformats::StrangenessTrackingStats mStats, *mStatsPtr = &mStats;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::string mSTrackingTreeName = "o2sim";
  std::string mStrackBranchName = "StrangeTracks";
  std::string mStrackMCBranchName = "StrangeTrackMCLab";
};

void StrangenessTrackingReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("strange-tracks-infile"));
  connectTree();
}

void StrangenessTrackingReader::run(ProcessingContext& pc)
{
  long ent = pc.services().get<o2::framework::TimingInfo>().timeslice;
  // A timeframe holds no collision at all whenever the interaction rate is low enough, and
  // the tree then has no entry to read. Publish empty containers instead of reading past the
  // end and pushing branch addresses that GetEntry has not filled, so that the consumers
  // downstream still see the timeframe. (This used to be an assert, which is compiled out of
  // every production build since ENABLE_CASSERT defaults to OFF.)
  const bool noEntry = ent >= mTree->GetEntries();
  if (noEntry) {
    LOGP(warning, "No entry {} to read, tree contains {} entries, publishing empty output", ent, mTree->GetEntries());
  } else {
    const auto bytes = mTree->GetEntry(ent);
    if ((mExtensions || mShortTracks) && bytes <= 0) {
      LOG(fatal) << "Cannot read strange-track entry " << ent << " in " << mFileName;
    }
  }
  if (mExtensions && (mTimes.size() != mExtended.size() || (mUseMC && mExtendedLabels.size() != mExtended.size()))) {
    LOG(fatal) << "Extended strange-track/time/MC cardinality mismatch at entry " << ent;
  }
  if (mShortTracks) {
    if (mShortPV.size() != mShort.size() || mShortClus.size() != 3 * mShort.size() || (mUseMC && mShortLabels.size() != mShort.size())) {
      LOG(fatal) << "ITS short-track/PV/cluster/MC cardinality mismatch at entry " << ent;
    }
    for (size_t i = 0; i < mShort.size(); ++i) {
      if (mShort[i].getNClusters() != 3 || mShort[i].getFirstClusterEntry() != 3 * i) {
        LOG(fatal) << "Invalid ITS short-track cluster reference range at entry " << ent;
      }
    }
  }
  LOG(info) << "Pushing " << mStrangeTrack.size() << " strange tracks at entry " << ent;
  pc.outputs().snapshot(Output{"GLO", "STRANGETRACKS", 0}, mStrangeTrack);

  if (mUseMC) {
    LOG(info) << "Pushing " << mStrangeTrackMC.size() << " strange tracks MC labels at entry " << ent;
    pc.outputs().snapshot(Output{"GLO", "STRANGETRACKS_MC", 0}, mStrangeTrackMC);
  }

  if (mExtensions) {
    pc.outputs().snapshot(Output{"GLO", "EXTSTRTRACKS", 0}, mExtended);
    pc.outputs().snapshot(Output{"GLO", "EXTSTRTIMES", 0}, mTimes);
    if (mUseMC) {
      pc.outputs().snapshot(Output{"GLO", "EXTSTRTRACKS_MC", 0}, mExtendedLabels);
    }
  }
  if (mShortTracks) {
    pc.outputs().snapshot(Output{"GLO", "ITSSHORTTRACKS", 0}, mShort);
    pc.outputs().snapshot(Output{"GLO", "ITSSHORTPV", 0}, mShortPV);
    pc.outputs().snapshot(Output{"GLO", "ITSSHORTCLREF", 0}, mShortClus);
    if (mUseMC) {
      pc.outputs().snapshot(Output{"GLO", "ITSSHORTMC", 0}, mShortLabels);
    }
  }
  if (mExtensions || mShortTracks) {
    pc.outputs().snapshot(Output{"GLO", "STRTRACKSTATS", 0}, mStats);
  }

  if (noEntry || mTree->GetReadEntry() + pc.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void StrangenessTrackingReader::connectTree()
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mFileName.c_str()));
  if (!mFile || mFile->IsZombie()) {
    LOG(fatal) << "Cannot open strange-track input " << mFileName;
  }
  mTree.reset((TTree*)mFile->Get(mSTrackingTreeName.c_str()));
  if (!mTree) {
    LOG(fatal) << "Missing " << mSTrackingTreeName << " tree in " << mFileName;
  }

  mTree->SetBranchAddress(mStrackBranchName.c_str(), &mStrangeTrackPtr);
  if (mUseMC) {
    mTree->SetBranchAddress(mStrackMCBranchName.c_str(), &mStrangeTrackMCPtr);
  }

  auto bind = [&](const char* name, auto address) {
    if (!mTree->GetBranch(name) || mTree->SetBranchAddress(name, address) < 0) {
      LOG(fatal) << "Missing or incompatible enabled strangeness-tracking branch " << name << " in " << mFileName;
    }
  };
  if (mExtensions) {
    bind("ExtendedStrangeTracks", &mExtendedPtr);
    bind("ExtendedStrangeTrackTimes", &mTimesPtr);
    if (mUseMC) {
      bind("ExtendedStrangeTrackMCLab", &mExtendedLabelsPtr);
    }
  }
  if (mShortTracks) {
    bind("ITSShortTracks", &mShortPtr);
    bind("ITSShortTrackPVRefs", &mShortPVPtr);
    bind("ITSShortTrackClusRefs", &mShortClusPtr);
    if (mUseMC) {
      bind("ITSShortTrackMCLab", &mShortLabelsPtr);
    }
  }
  if (mExtensions || mShortTracks) {
    bind("StrangenessTrackingStats", &mStatsPtr);
  }

  LOG(info) << "Loaded " << mSTrackingTreeName << " tree from " << mFileName << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getStrangenessTrackingReaderSpec(bool useMC, bool enableExtensions, bool enableShortTracks)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "STRANGETRACKS", 0, Lifetime::Timeframe); // found strange tracks
  if (useMC) {
    outputs.emplace_back("GLO", "STRANGETRACKS_MC", 0, Lifetime::Timeframe); // MC labels
  }

  if (enableExtensions) {
    outputs.emplace_back("GLO", "EXTSTRTRACKS", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "EXTSTRTIMES", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back("GLO", "EXTSTRTRACKS_MC", 0, Lifetime::Timeframe);
    }
  }
  if (enableShortTracks) {
    outputs.emplace_back("GLO", "ITSSHORTTRACKS", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "ITSSHORTPV", 0, Lifetime::Timeframe);
    outputs.emplace_back("GLO", "ITSSHORTCLREF", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back("GLO", "ITSSHORTMC", 0, Lifetime::Timeframe);
    }
  }
  if (enableExtensions || enableShortTracks) {
    outputs.emplace_back("GLO", "STRTRACKSTATS", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "strangeness-tracking-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<StrangenessTrackingReader>(useMC, enableExtensions, enableShortTracks)},
    Options{
      {"strange-tracks-infile", VariantType::String, "o2_strange_tracks.root", {"Name of the input strange tracks file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace strangeness_tracking
} // namespace o2
