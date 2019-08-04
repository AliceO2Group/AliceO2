// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFinderOriginalSpec.cxx
/// \brief Implementation of a data processor to read clusters, reconstruct tracks and send them
///
/// \author Philippe Pillot, Subatech

#include "TrackFinderOriginalSpec.h"

#include <chrono>
#include <list>
#include <stdexcept>

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MCHBase/ClusterBlock.h"
#include "MCHBase/TrackBlock.h"
#include "TrackParam.h"
#include "Cluster.h"
#include "Track.h"
#include "TrackFinderOriginal.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackFinderTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools

    LOG(INFO) << "initializing track finder";

    auto l3Current = ic.options().get<float>("l3Current");
    auto dipoleCurrent = ic.options().get<float>("dipoleCurrent");
    mTrackFinder.init(l3Current, dipoleCurrent);

    auto moreCandidates = ic.options().get<bool>("moreCandidates");
    mTrackFinder.findMoreTrackCandidates(moreCandidates);

    auto debugLevel = ic.options().get<int>("debug");
    mTrackFinder.debug(debugLevel);

    auto stop = [this]() {
      LOG(INFO) << "tracking duration = " << mElapsedTime.count() << " s";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the clusters of the current event, find tracks and send them

    // get the input buffer
    auto msgIn = pc.inputs().get<gsl::span<char>>("clusters");
    auto bufferPtr = msgIn.data();
    int sizeLeft = msgIn.size();

    // get the event number
    if (sizeLeft < SSizeOfInt) {
      throw out_of_range("missing event header");
    }
    const int& event = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;
    sizeLeft -= SSizeOfInt;

    // get the input clusters
    std::array<std::list<Cluster>, 10> clusters{};
    try {
      readClusters(bufferPtr, sizeLeft, clusters);
    } catch (exception const& e) {
      throw;
    }

    // run the track finder
    auto tStart = std::chrono::high_resolution_clock::now();
    const auto& tracks = mTrackFinder.findTracks(&clusters);
    auto tEnd = std::chrono::high_resolution_clock::now();
    mElapsedTime += tEnd - tStart;

    // calculate the size of the payload for the output message, excluding the event header
    int trackSize = getSize(tracks);

    // create the output message
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "TRACKS", 0, Lifetime::Timeframe}, SHeaderSize + trackSize);
    auto bufferPtrOut = msgOut.data();

    // write the event header
    writeHeader(event, trackSize, bufferPtrOut);

    // write the tracks
    if (trackSize > 0) {
      writeTracks(tracks, bufferPtrOut);
    }
  }

 private:
  //_________________________________________________________________________________________________
  void readClusters(const char*& bufferPtr, int& sizeLeft, std::array<std::list<Cluster>, 10>& clusters)
  {
    /// read the cluster informations from the buffer
    /// move the buffer ptr and decrease the size left
    /// throw an exception in case of error

    // read the number of clusters
    if (sizeLeft < SSizeOfInt) {
      throw out_of_range("missing number of clusters");
    }
    const int& nClusters = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;
    sizeLeft -= SSizeOfInt;

    for (int iCl = 0; iCl < nClusters; ++iCl) {

      // read cluster info
      if (sizeLeft < SSizeOfClusterStruct) {
        throw out_of_range("missing cluster");
      }
      const auto cluster = reinterpret_cast<const ClusterStruct*>(bufferPtr);
      clusters[cluster->getChamberId()].emplace_back(*cluster);
      bufferPtr += SSizeOfClusterStruct;
      sizeLeft -= SSizeOfClusterStruct;
    }

    if (sizeLeft != 0) {
      throw length_error("incorrect payload");
    }
  }

  //_________________________________________________________________________________________________
  int getSize(const std::list<Track>& tracks)
  {
    /// calculate the total number of bytes requested to store the tracks

    int size(0);
    for (const auto& track : tracks) {
      size += SSizeOfTrackParamStruct + SSizeOfInt + track.getNClusters() * SSizeOfClusterStruct;
    }
    if (size > 0) {
      size += SSizeOfInt;
    }

    return size;
  }

  //_________________________________________________________________________________________________
  void writeHeader(int event, int trackSize, char*& bufferPtr) const
  {
    /// write header informations in the output buffer and move the buffer ptr

    // write the event number
    memcpy(bufferPtr, &event, SSizeOfInt);
    bufferPtr += SSizeOfInt;

    // write the size of the payload
    memcpy(bufferPtr, &trackSize, SSizeOfInt);
    bufferPtr += SSizeOfInt;
  }

  //_________________________________________________________________________________________________
  void writeTracks(const std::list<Track>& tracks, char*& bufferPtr) const
  {
    /// write the track informations in the buffer and move the buffer ptr

    // write the number of tracks
    int nTracks = tracks.size();
    memcpy(bufferPtr, &nTracks, SSizeOfInt);
    bufferPtr += SSizeOfInt;

    for (const auto& track : tracks) {

      // write track parameters
      TrackParamStruct paramStruct = track.first().getTrackParamStruct();
      memcpy(bufferPtr, &paramStruct, SSizeOfTrackParamStruct);
      bufferPtr += SSizeOfTrackParamStruct;

      // write the number of clusters
      int nClusters = track.getNClusters();
      memcpy(bufferPtr, &nClusters, SSizeOfInt);
      bufferPtr += SSizeOfInt;

      for (const auto& param : track) {

        // write cluster info
        ClusterStruct clusterStruct = param.getClusterPtr()->getClusterStruct();
        memcpy(bufferPtr, &clusterStruct, SSizeOfClusterStruct);
        bufferPtr += SSizeOfClusterStruct;
      }
    }
  }

  static constexpr int SSizeOfInt = sizeof(int);
  static constexpr int SHeaderSize = 2 * SSizeOfInt;
  static constexpr int SSizeOfClusterStruct = sizeof(ClusterStruct);
  static constexpr int SSizeOfTrackParamStruct = sizeof(TrackParamStruct);

  TrackFinderOriginal mTrackFinder{};           ///< track finder
  std::chrono::duration<double> mElapsedTime{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFinderOriginalSpec()
{
  return DataProcessorSpec{
    "TrackFinderOriginal",
    Inputs{InputSpec{"clusters", "MCH", "CLUSTERS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "TRACKS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackFinderTask>()},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}},
            {"moreCandidates", VariantType::Bool, false, {"Find more track candidates"}},
            {"debug", VariantType::Int, 0, {"debug level"}}}};
}

} // namespace mch
} // namespace o2
