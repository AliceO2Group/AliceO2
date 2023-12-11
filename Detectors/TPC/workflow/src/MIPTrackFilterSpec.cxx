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

/// \file MIPTrackFilterSpec.h
/// \brief Workflow to filter MIP tracks and streams them to other devices.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#include "TPCWorkflow/MIPTrackFilterSpec.h"

#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include <random>

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "Headers/DataHeader.h"

using namespace o2::framework;

namespace o2::tpc
{

class MIPTrackFilterDevice : public Task
{
 public:
  void init(framework::InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& eos) final;

 private:
  void sendOutput(DataAllocator& output);

  TrackCuts mCuts{};                  ///< Tracks cuts object
  std::vector<TrackTPC> mMIPTracks;   ///< Filtered MIP tracks
  unsigned int mProcessEveryNthTF{1}; ///< process every Nth TF only
  int mMaxTracksPerTF{-1};            ///< max number of MIP tracks processed per TF
  uint32_t mTFCounter{0};             ///< counter to keep track of the TFs
  int mProcessNFirstTFs{0};           ///< number of first TFs which are not sampled
  bool mSendDummy{false};             ///< send empty data in case TF is skipped
};

void MIPTrackFilterDevice::init(framework::InitContext& ic)
{
  const double minP = ic.options().get<double>("min-momentum");
  const double maxP = ic.options().get<double>("max-momentum");
  const double mindEdx = ic.options().get<double>("min-dedx");
  const double maxdEdx = ic.options().get<double>("max-dedx");
  const int minClusters = std::max(10, ic.options().get<int>("min-clusters"));
  const double mSendDummy = ic.options().get<bool>("send-dummy-data");
  mMaxTracksPerTF = ic.options().get<int>("maxTracksPerTF");
  if (mMaxTracksPerTF > 0) {
    mMIPTracks.reserve(mMaxTracksPerTF);
  }

  mProcessEveryNthTF = ic.options().get<int>("processEveryNthTF");
  if (mProcessEveryNthTF <= 0) {
    mProcessEveryNthTF = 1;
  }
  mProcessNFirstTFs = ic.options().get<int>("process-first-n-TFs");

  if (mProcessEveryNthTF > 1) {
    std::mt19937 rng(std::time(nullptr));
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, mProcessEveryNthTF);
    mTFCounter = dist(rng);
    LOGP(info, "Skipping first {} TFs", mProcessEveryNthTF - mTFCounter);
  }

  mCuts.setPMin(minP);
  mCuts.setPMax(maxP);
  mCuts.setNClusMin(minClusters);
  mCuts.setdEdxMin(mindEdx);
  mCuts.setdEdxMax(maxdEdx);
}

void MIPTrackFilterDevice::run(ProcessingContext& pc)
{
  const auto currentTF = processing_helpers::getCurrentTF(pc);
  if ((mTFCounter++ % mProcessEveryNthTF) && (currentTF >= mProcessNFirstTFs)) {
    LOGP(info, "Skipping TF {}", currentTF);
    mMIPTracks.clear();
    if (mSendDummy) {
      sendOutput(pc.outputs());
    }
    return;
  }

  const auto tracks = pc.inputs().get<gsl::span<TrackTPC>>("tracks");
  const auto nTracks = tracks.size();

  if ((mMaxTracksPerTF != -1) && (nTracks > mMaxTracksPerTF)) {
    // indices to good tracks
    std::vector<size_t> indices;
    indices.reserve(nTracks);
    for (size_t i = 0; i < nTracks; ++i) {
      if (mCuts.goodTrack(tracks[i])) {
        indices.emplace_back(i);
      }
    }

    // in case no good tracks have been found
    if (indices.empty()) {
      mMIPTracks.clear();
      if (mSendDummy) {
        sendOutput(pc.outputs());
      }
      return;
    }

    // shuffle indices to good tracks
    std::minstd_rand rng(std::time(nullptr));
    std::shuffle(indices.begin(), indices.end(), rng);

    // copy good tracks
    const int loopEnd = (mMaxTracksPerTF > indices.size()) ? indices.size() : mMaxTracksPerTF;
    for (int i = 0; i < loopEnd; ++i) {
      mMIPTracks.emplace_back(tracks[indices[i]]);
    }
  } else {
    std::copy_if(tracks.begin(), tracks.end(), std::back_inserter(mMIPTracks), [this](const auto& track) { return mCuts.goodTrack(track); });
  }

  LOGP(info, "Filtered {} MIP tracks out of {} total tpc tracks", mMIPTracks.size(), tracks.size());
  sendOutput(pc.outputs());
  mMIPTracks.clear();
}

void MIPTrackFilterDevice::sendOutput(DataAllocator& output) { output.snapshot(Output{header::gDataOriginTPC, "MIPS", 0}, mMIPTracks); }

void MIPTrackFilterDevice::endOfStream(EndOfStreamContext& eos)
{
  LOG(info) << "Finalizig MIP Tracks filter";
}

DataProcessorSpec getMIPTrackFilterSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(header::gDataOriginTPC, "MIPS", 0, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-miptrack-filter",
    Inputs{
      InputSpec{"tracks", "TPC", "TRACKS"},
    },
    outputs,
    adaptFromTask<MIPTrackFilterDevice>(),
    Options{
      {"min-momentum", VariantType::Double, 0.3, {"minimum momentum cut"}},
      {"max-momentum", VariantType::Double, 0.7, {"maximum momentum cut"}},
      {"min-dedx", VariantType::Double, 20., {"minimum dEdx cut"}},
      {"max-dedx", VariantType::Double, 200., {"maximum dEdx cut"}},
      {"min-clusters", VariantType::Int, 60, {"minimum number of clusters in a track"}},
      {"processEveryNthTF", VariantType::Int, 1, {"Using only a fraction of the data: 1: Use every TF, 10: Process only every tenth TF."}},
      {"maxTracksPerTF", VariantType::Int, -1, {"Maximum number of processed tracks per TF (-1 for processing all tracks)"}},
      {"process-first-n-TFs", VariantType::Int, 1, {"Number of first TFs which are not sampled"}},
      {"send-dummy-data", VariantType::Bool, false, {"Send empty data in case TF is skipped"}}}};
}

} // namespace o2::tpc
