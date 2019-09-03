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

#include "TrackFitterSpec.h"

#include <stdexcept>
#include <list>

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
#include "TrackFitter.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackFitterTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools
    LOG(INFO) << "initializing track fitter";
    auto l3Current = ic.options().get<float>("l3Current");
    auto dipoleCurrent = ic.options().get<float>("dipoleCurrent");
    mTrackFitter.initField(l3Current, dipoleCurrent);
    mTrackFitter.smoothTracks(true);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the tracks with attached clusters of the current event,
    /// refit them and send the new version

    // get the input buffer
    auto msgIn = pc.inputs().get<gsl::span<char>>("tracks");
    auto bufferPtr = msgIn.data();
    int sizeLeft = msgIn.size();

    // create the output message
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "REFITTRACKS", 0, Lifetime::Timeframe}, sizeLeft);
    auto bufferPtrOut = msgOut.data();

    // copy header info
    try {
      copyHeader(bufferPtr, sizeLeft, bufferPtrOut);
    } catch (exception const& e) {
      throw;
    }

    // get the number of tracks and copy it to the output message
    if (sizeLeft < SSizeOfInt) {
      throw out_of_range("missing number of tracks");
    }
    const int& nTracks = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;
    memcpy(bufferPtrOut, &nTracks, SSizeOfInt);
    bufferPtrOut += SSizeOfInt;
    sizeLeft -= SSizeOfInt;

    std::list<Cluster> clusters{};
    for (int iTrack = 0; iTrack < nTracks; ++iTrack) {

      // get the input track
      Track track{};
      try {
        readTrack(bufferPtr, sizeLeft, track, clusters);
      } catch (exception const& e) {
        throw;
      }

      // refit the track
      try {
        mTrackFitter.fit(track);
      } catch (exception const& e) {
        throw runtime_error(std::string("Track fit failed: ") + e.what());
      }

      // write the refitted track to the ouput message
      writeTrack(track, bufferPtrOut);
    }

    if (sizeLeft != 0) {
      throw length_error("incorrect payload");
    }
  }

 private:
  //_________________________________________________________________________________________________
  void copyHeader(const char*& bufferPtr, int& sizeLeft, char*& bufferPtrOut) const
  {
    /// copy header informations from the input buffer to the output message
    /// move the buffer ptr and decrease the size left
    /// throw an exception in case of error

    if (sizeLeft < SHeaderSize) {
      throw out_of_range("missing event header");
    }

    const int& event = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;
    memcpy(bufferPtrOut, &event, SSizeOfInt);
    bufferPtrOut += SSizeOfInt;

    const int& size = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;
    memcpy(bufferPtrOut, &size, SSizeOfInt);
    bufferPtrOut += SSizeOfInt;

    sizeLeft -= SHeaderSize;
    if (sizeLeft != size) {
      throw length_error("incorrect payload size");
    }
  }

  //_________________________________________________________________________________________________
  void readTrack(const char*& bufferPtr, int& sizeLeft, Track& track, std::list<Cluster>& clusters) const
  {
    /// read the track informations from the buffer
    /// move the buffer ptr and decrease the size left
    /// throw an exception in case of error

    // skip the track parameters
    if (sizeLeft < SSizeOfTrackParamStruct) {
      throw out_of_range("missing track parameters");
    }
    bufferPtr += SSizeOfTrackParamStruct;
    sizeLeft -= SSizeOfTrackParamStruct;

    // read number of clusters
    if (sizeLeft < SSizeOfInt) {
      throw out_of_range("missing number of clusters");
    }
    const int& nClusters = *reinterpret_cast<const int*>(bufferPtr);
    if (nClusters > 20) {
      throw length_error("too many (>20) clusters attached to the track");
    }
    bufferPtr += SSizeOfInt;
    sizeLeft -= SSizeOfInt;

    for (int iCl = 0; iCl < nClusters; ++iCl) {

      // read cluster info
      if (sizeLeft < SSizeOfClusterStruct) {
        throw out_of_range("missing cluster");
      }
      clusters.emplace_back(*reinterpret_cast<const ClusterStruct*>(bufferPtr));
      track.createParamAtCluster(clusters.back());
      bufferPtr += SSizeOfClusterStruct;
      sizeLeft -= SSizeOfClusterStruct;
    }
  }

  //_________________________________________________________________________________________________
  void writeTrack(Track& track, char*& bufferPtrOut) const
  {
    /// write the track informations to the buffer and move the buffer ptr

    // fill track parameters
    TrackParamStruct paramStruct = track.first().getTrackParamStruct();
    memcpy(bufferPtrOut, &paramStruct, SSizeOfTrackParamStruct);
    bufferPtrOut += SSizeOfTrackParamStruct;

    // fill number of clusters
    int nClusters = track.getNClusters();
    memcpy(bufferPtrOut, &nClusters, SSizeOfInt);
    bufferPtrOut += SSizeOfInt;

    for (const auto& param : track) {

      // fill cluster info
      ClusterStruct clusterStruct = param.getClusterPtr()->getClusterStruct();
      memcpy(bufferPtrOut, &clusterStruct, SSizeOfClusterStruct);
      bufferPtrOut += SSizeOfClusterStruct;
    }
  }

  static constexpr int SSizeOfInt = sizeof(int);
  static constexpr int SHeaderSize = 2 * SSizeOfInt;
  static constexpr int SSizeOfTrackParamStruct = sizeof(TrackParamStruct);
  static constexpr int SSizeOfClusterStruct = sizeof(ClusterStruct);

  TrackFitter mTrackFitter{}; ///< track fitter
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFitterSpec()
{
  return DataProcessorSpec{
    "TrackFitter",
    Inputs{InputSpec{"tracks", "MCH", "TRACKS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "REFITTRACKS", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackFitterTask>()},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}}}};
}

} // namespace mch
} // namespace o2
