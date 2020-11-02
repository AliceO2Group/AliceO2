// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSinkSpec.cxx
/// \brief Implementation of a data processor to print the tracks
///
/// \author Philippe Pillot, Subatech

#include "TrackSinkSpec.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/ClusterBlock.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackSinkTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the output file from the context
    LOG(INFO) << "initializing track sink";

    auto outputFileName = ic.options().get<std::string>("outfile");
    mOutputFile.open(outputFileName, ios::out | ios::binary);
    if (!mOutputFile.is_open()) {
      throw invalid_argument("Cannot open output file" + outputFileName);
    }

    auto stop = [this]() {
      /// close the output file
      LOG(INFO) << "stop track sink";
      this->mOutputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// dump the tracks with attached clusters of the current event

    // get the input messages
    gsl::span<const TrackMCH> tracks{};
    if (pc.inputs().getPos("tracks") >= 0) {
      tracks = pc.inputs().get<gsl::span<TrackMCH>>("tracks");
    }
    gsl::span<const ClusterStruct> clusters{};
    if (pc.inputs().getPos("clusters") >= 0) {
      clusters = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");
    }
    gsl::span<const char> tracksAtVtx{};
    if (pc.inputs().getPos("tracksAtVtx") >= 0) {
      tracksAtVtx = pc.inputs().get<gsl::span<char>>("tracksAtVtx");
    }

    // write the number of tracks at vertex, MCH tracks and attached clusters
    int nTracksAtVtx = tracksAtVtx.empty() ? 0 : *reinterpret_cast<const int*>(tracksAtVtx.data());
    mOutputFile.write(reinterpret_cast<char*>(&nTracksAtVtx), sizeof(int));
    int nTracks = tracks.size();
    mOutputFile.write(reinterpret_cast<char*>(&nTracks), sizeof(int));
    int nClusters = clusters.size();
    mOutputFile.write(reinterpret_cast<char*>(&nClusters), sizeof(int));

    // write the tracks at vertex, MCH tracks and attached clusters
    if (tracksAtVtx.size() > sizeof(int)) {
      mOutputFile.write(&tracksAtVtx[sizeof(int)], tracksAtVtx.size() - sizeof(int));
    }
    mOutputFile.write(reinterpret_cast<const char*>(tracks.data()), tracks.size_bytes());
    mOutputFile.write(reinterpret_cast<const char*>(clusters.data()), clusters.size_bytes());
  }

 private:
  std::ofstream mOutputFile{}; ///< output file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackSinkSpec(bool mchTracks, bool tracksAtVtx)
{
  Inputs inputs{};
  if (mchTracks) {
    inputs.emplace_back("tracks", "MCH", "TRACKS", 0, Lifetime::Timeframe);
    inputs.emplace_back("clusters", "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe);
  }
  if (tracksAtVtx) {
    inputs.emplace_back("tracksAtVtx", "MCH", "TRACKSATVERTEX", 0, Lifetime::Timeframe);
  }
  if (inputs.empty()) {
    throw invalid_argument("nothing to write");
  }

  return DataProcessorSpec{
    "TrackSink",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<TrackSinkTask>()},
    Options{{"outfile", VariantType::String, "AliESDs.out.dat", {"output filename"}}}};
}

} // end namespace mch
} // end namespace o2
