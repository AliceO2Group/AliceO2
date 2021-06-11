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
#include <vector>

#include <gsl/span>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHBase/TrackBlock.h"

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
    /// dump the tracks and attached clusters for each event in the TF

    // get the input messages
    gsl::span<const ROFRecord> rofs{};
    gsl::span<const TrackMCH> tracks{};
    gsl::span<const ClusterStruct> clusters{};
    if (pc.inputs().getPos("rofs") >= 0) {
      rofs = pc.inputs().get<gsl::span<ROFRecord>>("rofs");
      tracks = pc.inputs().get<gsl::span<TrackMCH>>("tracks");
      clusters = pc.inputs().get<gsl::span<ClusterStruct>>("clusters");
    }
    gsl::span<const char> tracksAtVtx{};
    if (pc.inputs().getPos("tracksAtVtx") >= 0) {
      tracksAtVtx = pc.inputs().get<gsl::span<char>>("tracksAtVtx");
    }

    // loop over events based on ROF records, if any, or based on the tracks-at-vertex message otherwise
    if (!rofs.empty()) {

      int tracksAtVtxOffset(0);
      std::vector<TrackMCH> eventTracks{};
      for (const auto& rof : rofs) {

        // get the MCH tracks, attached clusters and corresponding tracks at vertex (if any)
        auto eventClusters = getEventTracksAndClusters(rof, tracks, clusters, eventTracks);
        auto eventTracksAtVtx = getEventTracksAtVtx(tracksAtVtx, tracksAtVtxOffset);

        // write the number of tracks at vertex, MCH tracks and attached clusters
        int nEventTracksAtVtx = eventTracksAtVtx.size() / sizeof(TrackAtVtxStruct);
        mOutputFile.write(reinterpret_cast<char*>(&nEventTracksAtVtx), sizeof(int));
        int nEventTracks = eventTracks.size();
        mOutputFile.write(reinterpret_cast<char*>(&nEventTracks), sizeof(int));
        int nEventClusters = eventClusters.size();
        mOutputFile.write(reinterpret_cast<char*>(&nEventClusters), sizeof(int));

        // write the tracks at vertex, MCH tracks and attached clusters
        mOutputFile.write(eventTracksAtVtx.data(), eventTracksAtVtx.size());
        mOutputFile.write(reinterpret_cast<const char*>(eventTracks.data()), eventTracks.size() * sizeof(TrackMCH));
        mOutputFile.write(reinterpret_cast<const char*>(eventClusters.data()), eventClusters.size_bytes());
      }

      // at this point we should have dumped all the tracks at vertex, if any
      if (tracksAtVtxOffset != tracksAtVtx.size()) {
        throw length_error("inconsistent payload");
      }

    } else if (!tracksAtVtx.empty()) {

      int tracksAtVtxOffset(0);
      int zero(0);
      while (tracksAtVtxOffset != tracksAtVtx.size()) {

        // get the tracks at vertex
        auto eventTracksAtVtx = getEventTracksAtVtx(tracksAtVtx, tracksAtVtxOffset);

        // write the number of tracks at vertex (number of MCH tracks and attached clusters = 0)
        int nEventTracksAtVtx = eventTracksAtVtx.size() / sizeof(TrackAtVtxStruct);
        mOutputFile.write(reinterpret_cast<char*>(&nEventTracksAtVtx), sizeof(int));
        mOutputFile.write(reinterpret_cast<char*>(&zero), sizeof(int));
        mOutputFile.write(reinterpret_cast<char*>(&zero), sizeof(int));

        // write the tracks at vertex
        mOutputFile.write(eventTracksAtVtx.data(), eventTracksAtVtx.size());
      }

    } else {
      throw length_error("empty time frame");
    }
  }

 private:
  struct TrackAtVtxStruct {
    TrackParamStruct paramAtVertex{};
    double dca = 0.;
    double rAbs = 0.;
    int mchTrackIdx = 0;
  };

  //_________________________________________________________________________________________________
  gsl::span<const ClusterStruct> getEventTracksAndClusters(const ROFRecord& rof, gsl::span<const TrackMCH> tracks,
                                                           gsl::span<const ClusterStruct> clusters,
                                                           std::vector<TrackMCH>& eventTracks) const
  {
    /// copy the MCH tracks of the current event (needed to edit the tracks)
    /// modify the references to the attached clusters to start the indexing from 0
    /// return a sub-span with the attached clusters

    eventTracks.clear();

    if (rof.getNEntries() < 1) {
      return {};
    }

    if (rof.getLastIdx() >= tracks.size()) {
      throw length_error("missing tracks");
    }

    eventTracks.insert(eventTracks.end(), tracks.begin() + rof.getFirstIdx(), tracks.begin() + rof.getLastIdx() + 1);

    int clusterIdxOffset = eventTracks.front().getFirstClusterIdx();
    for (auto& track : eventTracks) {
      track.setClusterRef(track.getFirstClusterIdx() - clusterIdxOffset, track.getNClusters());
    }

    if (eventTracks.back().getLastClusterIdx() + clusterIdxOffset >= clusters.size()) {
      throw length_error("missing clusters");
    }

    return clusters.subspan(clusterIdxOffset, eventTracks.back().getLastClusterIdx() + 1);
  }

  //_________________________________________________________________________________________________
  gsl::span<const char> getEventTracksAtVtx(gsl::span<const char> tracksAtVtx, int& tracksAtVtxOffset) const
  {
    /// return a sub-span with the tracks at vertex of the current event, if any,
    /// and move forward the tracksAtVtxOffset to point to the next event

    if (tracksAtVtx.empty()) {
      return {};
    }

    if (tracksAtVtx.size() - tracksAtVtxOffset < sizeof(int)) {
      throw length_error("inconsistent payload");
    }

    int nEventTracksAtVtx = *reinterpret_cast<const int*>(&tracksAtVtx[tracksAtVtxOffset]);
    tracksAtVtxOffset += sizeof(int);

    int payloadSize = nEventTracksAtVtx * sizeof(TrackAtVtxStruct);
    if (tracksAtVtx.size() - tracksAtVtxOffset < payloadSize) {
      throw length_error("inconsistent payload");
    }

    tracksAtVtxOffset += payloadSize;
    return tracksAtVtx.subspan(tracksAtVtxOffset - payloadSize, payloadSize);
  }

  std::ofstream mOutputFile{}; ///< output file
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackSinkSpec(bool mchTracks, bool tracksAtVtx)
{
  Inputs inputs{};
  if (mchTracks) {
    inputs.emplace_back("rofs", "MCH", "TRACKROFS", 0, Lifetime::Timeframe);
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
