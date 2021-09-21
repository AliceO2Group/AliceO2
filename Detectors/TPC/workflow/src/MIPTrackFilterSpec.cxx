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

//o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"

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

  TrackCuts mCuts{};                ///< Tracks cuts object
  std::vector<TrackTPC> mMIPTracks; ///< Filtered MIP tracks
};

void MIPTrackFilterDevice::init(framework::InitContext& ic)
{
  const double minP = ic.options().get<double>("min-momentum");
  const double maxP = ic.options().get<double>("max-momentum");
  assert(minP < maxP);
  const int minClusters = std::max(10, ic.options().get<int>("min-clusters"));

  mCuts.setPMin(minP);
  mCuts.setPMax(maxP);
  mCuts.setNClusMin(minClusters);
}

void MIPTrackFilterDevice::run(ProcessingContext& pc)
{
  const auto tracks = pc.inputs().get<gsl::span<TrackTPC>>("tracks");

  std::copy_if(tracks.begin(), tracks.end(), std::back_inserter(mMIPTracks),
               [this](const auto& track) { return this->mCuts.goodTrack(track); });

  LOG(info) << mMIPTracks.size() << " MIP tracks in a total of " << tracks.size() << " tracks";

  pc.outputs().snapshot(Output{"TPC", "MIPS", 0, Lifetime::Timeframe}, mMIPTracks);
  mMIPTracks.clear();
}

void MIPTrackFilterDevice::endOfStream(EndOfStreamContext& eos)
{
  LOG(info) << "Finalizig MIP Tracks filter";
}

DataProcessorSpec getMIPTrackFilterSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TPC", "MIPS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-miptrack-filter",
    Inputs{
      InputSpec{"tracks", "TPC", "TRACKS"},
    },
    outputs,
    adaptFromTask<MIPTrackFilterDevice>(),
    Options{
      {"min-momentum", VariantType::Double, 0.4, {"minimum momentum cut"}},
      {"max-momentum", VariantType::Double, 0.6, {"maximum momentum cut"}},
      {"min-clusters", VariantType::Int, 60, {"minimum number of clusters in a track"}}}};
}

} // namespace o2::tpc
