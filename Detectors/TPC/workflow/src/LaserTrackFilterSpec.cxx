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

/// @file   LaserTrackFilterSpec.cxx
/// @brief  Device to filter out laser tracks

#include <algorithm>
#include <iterator>

#include "DataFormatsTPC/TrackTPC.h"
#include "TPCCalibration/CalibLaserTracks.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"

using namespace o2::framework;

namespace o2::tpc
{

class LaserTrackFilterDevice : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto tracks = pc.inputs().get<gsl::span<TrackTPC>>("tracks");
    std::copy_if(tracks.begin(), tracks.end(), std::back_inserter(mLaserTracks),
                 [this](const auto& track) { return isLaserTrackCandidate(track); });

    LOGP(info, "Filtered {} laser track candidates out of {} total tpc tracks", mLaserTracks.size(), tracks.size());

    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
  }

 private:
  std::vector<TrackTPC> mLaserTracks;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    output.snapshot(Output{"TPC", "LASERTRACKS", 0}, mLaserTracks);
    mLaserTracks.clear();
  }

  bool isLaserTrackCandidate(const TrackTPC& track)
  {
    if (track.getP() < 1) {
      return false;
    }

    if (track.getNClusters() < 80) {
      //return false;
    }

    if (track.hasBothSidesClusters()) {
      return false;
    }

    const auto& parOutLtr = track.getOuterParam();
    if (parOutLtr.getX() < 220) {
      return false;
    }

    const int side = track.hasCSideClusters();
    if (!CalibLaserTracks::hasNearbyLaserRod(parOutLtr, side)) {
      return false;
    }

    return true;
  }
};

DataProcessorSpec getLaserTrackFilter()
{
  using device = o2::tpc::LaserTrackFilterDevice;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TPC", "LASERTRACKS", 0, o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-laser-track-filter",
    Inputs{{"tracks", "TPC", "TRACKS", 0}},
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{}};
}

} // namespace o2::tpc
