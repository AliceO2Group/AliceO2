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

#include <stdexcept>
#include <list>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MFTTracking/TrackParam.h"
#include "MFTTracking/Cluster.h"
#include "MFTTracking/TrackCA.h"
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
    std::vector<Cluster> clusters;
    int nTracksCA = 0;
    int nTracksLTF = 0;
    std::vector<o2::mft::TrackMFTExt> tracks;

    for (auto track : tracksLTF)
      {
      LOG(INFO) << "tracksLTF: nTracksLTF  = " << nTracksLTF << " tracks.size() = " << tracks.size() << std::endl;
      tracks.push_back(mTrackFitter->fit(track, clusters));
      nTracksLTF++;
      }

    for (auto track : tracksCA)
      {
      tracks.push_back(mTrackFitter->fit(track, clusters));
      LOG(INFO) << "tracksCA: nTracksCA  = " << nTracksCA << " tracks.size() = " << tracks.size() << std::endl;
      nTracksCA++;
      }


    LOG(INFO) << "MFTFitter loaded " << tracksLTF.size() << " LTF tracks";
    LOG(INFO) << "MFTFitter loaded " << tracksCA.size() << " CA tracks";

    LOG(INFO) << "MFTFitter pushed " << nTracksLTF << " LTF tracks";
    LOG(INFO) << "MFTFitter pushed " << nTracksCA << " CA tracks";
    LOG(INFO) << "MFTFitter pushed " << tracks.size() << " tracks";
    pc.outputs().snapshot(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe}, tracks);



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
