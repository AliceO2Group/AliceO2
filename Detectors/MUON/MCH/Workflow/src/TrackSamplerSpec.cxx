// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackSamplerSpec.cxx
/// \brief Implementation of a data processor to read and send tracks
///
/// \author Philippe Pillot, Subatech

#include "TrackSamplerSpec.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/OutputRef.h"
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

class TrackSamplerTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Get the input file from the context
    LOG(INFO) << "initializing track sampler";

    auto inputFileName = ic.options().get<std::string>("infile");
    mInputFile.open(inputFileName, ios::binary);
    if (!mInputFile.is_open()) {
      throw invalid_argument("cannot open input file" + inputFileName);
    }
    if (mInputFile.peek() == EOF) {
      throw length_error("input file is empty");
    }

    mNEventsPerTF = ic.options().get<int>("nEventsPerTF");
    if (mNEventsPerTF < 1) {
      throw invalid_argument("number of events per time frame must be >= 1");
    }

    auto stop = [this]() {
      /// close the input file
      LOG(INFO) << "stop track sampler";
      this->mInputFile.close();
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// send the tracks with attached clusters of the next events in the current TF

    static uint32_t event(0);

    // reached eof
    if (mInputFile.peek() == EOF) {
      pc.services().get<ControlService>().endOfStream();
      //pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      return;
    }

    // create the output messages
    auto& rofs = pc.outputs().make<std::vector<ROFRecord>>(OutputRef{"rofs"});
    auto& tracks = pc.outputs().make<std::vector<TrackMCH>>(OutputRef{"tracks"});
    auto& clusters = pc.outputs().make<std::vector<ClusterStruct>>(OutputRef{"clusters"});

    // loop over the requested number of events (or until eof) and fill the messages
    for (int iEvt = 0; iEvt < mNEventsPerTF && mInputFile.peek() != EOF; ++iEvt) {
      int nTracks = readOneEvent(tracks, clusters);
      rofs.emplace_back(o2::InteractionRecord{0, event++}, tracks.size() - nTracks, nTracks);
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
  int readOneEvent(std::vector<TrackMCH, o2::pmr::polymorphic_allocator<TrackMCH>>& tracks,
                   std::vector<ClusterStruct, o2::pmr::polymorphic_allocator<ClusterStruct>>& clusters)
  {
    /// fill the output messages with the tracks and attached clusters of the current event
    /// modify the references to the attached clusters according to their position in the global vector

    // read the number of tracks at vertex, MCH tracks and attached clusters
    int nTracksAtVtx(-1);
    mInputFile.read(reinterpret_cast<char*>(&nTracksAtVtx), sizeof(int));
    if (mInputFile.fail()) {
      throw length_error("invalid input");
    }
    int nMCHTracks(-1);
    mInputFile.read(reinterpret_cast<char*>(&nMCHTracks), sizeof(int));
    if (mInputFile.fail()) {
      throw length_error("invalid input");
    }
    int nClusters(-1);
    mInputFile.read(reinterpret_cast<char*>(&nClusters), sizeof(int));
    if (mInputFile.fail()) {
      throw length_error("invalid input");
    }

    if (nTracksAtVtx < 0 || nMCHTracks < 0 || nClusters < 0) {
      throw length_error("invalid input");
    }
    if (nMCHTracks > 0 && nClusters == 0) {
      throw length_error("clusters are missing");
    }

    // skip the tracks at vertex if any
    if (nTracksAtVtx > 0) {
      mInputFile.seekg(nTracksAtVtx * sizeof(TrackAtVtxStruct), std::ios::cur);
      if (mInputFile.fail()) {
        throw length_error("invalid input");
      }
    }

    if (nMCHTracks > 0) {

      // read the MCH tracks
      int trackOffset = tracks.size();
      tracks.resize(trackOffset + nMCHTracks);
      mInputFile.read(reinterpret_cast<char*>(&tracks[trackOffset]), nMCHTracks * sizeof(TrackMCH));
      if (mInputFile.fail()) {
        throw length_error("invalid input");
      }

      // read the attached clusters
      int clusterOffset = clusters.size();
      clusters.resize(clusterOffset + nClusters);
      mInputFile.read(reinterpret_cast<char*>(&clusters[clusterOffset]), nClusters * sizeof(ClusterStruct));
      if (mInputFile.fail()) {
        throw length_error("invalid input");
      }

      // modify the cluster references
      for (auto itTrack = tracks.begin() + trackOffset; itTrack < tracks.end(); ++itTrack) {
        itTrack->setClusterRef(clusterOffset, itTrack->getNClusters());
        clusterOffset += itTrack->getNClusters();
      }
      if (clusterOffset != clusters.size()) {
        throw length_error("inconsistent cluster references");
      }
    }

    return nMCHTracks;
  }

  std::ifstream mInputFile{}; ///< input file
  int mNEventsPerTF = 1;      ///< number of events per time frame
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackSamplerSpec(bool forTrackFitter)
{
  Outputs outputs{};
  if (forTrackFitter) {
    outputs.emplace_back(OutputLabel{"rofs"}, "MCH", "TRACKROFSIN", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"tracks"}, "MCH", "TRACKSIN", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"clusters"}, "MCH", "TRACKCLUSTERSIN", 0, Lifetime::Timeframe);
  } else {
    outputs.emplace_back(OutputLabel{"rofs"}, "MCH", "TRACKROFS", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"tracks"}, "MCH", "TRACKS", 0, Lifetime::Timeframe);
    outputs.emplace_back(OutputLabel{"clusters"}, "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TrackSampler",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TrackSamplerTask>()},
    Options{{"infile", VariantType::String, "", {"input filename"}},
            {"nEventsPerTF", VariantType::Int, 1, {"number of events per time frame"}}}};
}

} // end namespace mch
} // end namespace o2
