// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitterSpec.cxx
/// \brief Implementation of a data processor to read, refit and send tracks with attached clusters
///
/// \author Philippe Pillot, Subatech

#include "MFTWorkflow/TrackFitterSpec.h"
#include "DataFormatsITSMFT/Cluster.h"

#include <stdexcept>
#include <list>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MFTTracking/TrackParam.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/FitterTrackMFT.h"
#include "MFTTracking/TrackFitter.h"

using namespace std;
using namespace o2::framework;

namespace o2
{
namespace mft
{

void TrackFitterTask::init(InitContext& ic)
{
  /// Prepare the track extrapolation tools
  LOG(INFO) << "initializing track fitter";
  mTrackFitter = std::make_unique<o2::mft::TrackFitter>();
  mState = 1;
}

//_________________________________________________________________________________________________
void TrackFitterTask::run(ProcessingContext& pc)
{

  if (mState != 1)
    return;

  auto tracksLTF = pc.inputs().get<const std::vector<o2::mft::TrackLTF>>("tracksltf");
  auto tracksCA = pc.inputs().get<const std::vector<o2::mft::TrackCA>>("tracksca");
  int nTracksCA = 0;
  int nTracksLTF = 0;
  std::vector<o2::mft::FitterTrackMFT> fittertracks;
  std::vector<o2::mft::TrackMFT> finalMFTtracks;

  // Fit LTF tracks
  for (auto track : tracksLTF) {
    o2::mft::FitterTrackMFT& temptrack = fittertracks.emplace_back();
    mTrackFitter->convertTrack(track, temptrack);
    mTrackFitter->fit(temptrack, false);
    LOG(INFO) << "tracksLTF: nTracksLTF  = " << nTracksLTF << " tracks.size() = " << fittertracks.size() << std::endl;
    nTracksLTF++;
  }
  // Fit CA tracks
  for (auto track : tracksCA) {
    o2::mft::FitterTrackMFT& temptrack = fittertracks.emplace_back();
    mTrackFitter->convertTrack(track, temptrack);
    mTrackFitter->fit(temptrack, false);
    LOG(INFO) << "tracksCA: nTracksCA  = " << nTracksCA << " tracks.size() = " << fittertracks.size() << std::endl;
    nTracksCA++;
  }

  // Convert fitter tracks to the final Standalone MFT Track
  for (auto track : fittertracks) {
    o2::mft::TrackMFT& temptrack = finalMFTtracks.emplace_back();
    // TODO: Convert FitterTrackMFT to TrackMFTExt or TrackMFT
    // Straight tracks considering only the first and last clusters
    auto dz = track.last().getZ() - track.first().getZ();
    auto slopeX = (track.last().getX() - track.first().getX()) / dz;
    auto slopeY = (track.last().getY() - track.first().getY()) / dz;
    auto tanl = -std::sqrt(1.f / (slopeX * slopeX + slopeY * +slopeY));
    auto tanp = slopeY / slopeX;
    auto sinp = tanp / std::sqrt(1.f + tanp * tanp);
    //LOG(INFO) << "   TrackPars: Tgl = " << tanl << "  Snp = " << sinp << std::endl;

    temptrack.setX(track.first().getX());
    temptrack.setY(track.first().getY());
    temptrack.setZ(track.first().getZ());
    temptrack.setTgl(tanl);
    temptrack.setSnp(sinp);
  }

  LOG(INFO) << "MFTFitter loaded " << tracksLTF.size() << " LTF tracks";
  LOG(INFO) << "MFTFitter loaded " << tracksCA.size() << " CA tracks";
  LOG(INFO) << "MFTFitter pushed " << fittertracks.size() << " tracks";
  pc.outputs().snapshot(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe}, finalMFTtracks);

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFitterSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tracksltf", "MFT", "TRACKSLTF", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksca", "MFT", "TRACKSCA", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "TRACKS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "mft-track-fitter",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackFitterTask>()},
    Options{}};
}

} // namespace mft
} // namespace o2
