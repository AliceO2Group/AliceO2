// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackAtVertexSpec.cxx
/// \brief Implementation of a data processor to extrapolate the tracks to the vertex
///
/// \author Philippe Pillot, Subatech

#include "TrackAtVertexSpec.h"

#include <chrono>
#include <stdexcept>
#include <list>

#include <TMath.h>
#include <TGeoManager.h>
#include <TGeoGlobalMagField.h>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "DetectorsBase/GeometryManager.h"
#include "MathUtils/Cartesian3D.h"
#include "Field/MagneticField.h"
#include "MCHBase/ClusterBlock.h"
#include "MCHBase/TrackBlock.h"
#include "TrackParam.h"
#include "TrackExtrap.h"

namespace o2
{
namespace mch
{

using namespace std;
using namespace o2::framework;

class TrackAtVertexTask
{
 public:
  //_________________________________________________________________________________________________
  void init(framework::InitContext& ic)
  {
    /// Prepare the track extrapolation tools

    LOG(INFO) << "initializing track extrapolation to vertex";

    if (!gGeoManager) {
      o2::base::GeometryManager::loadGeometry();
      if (!gGeoManager) {
        throw runtime_error("cannot load the geometry");
      }
    }

    if (!TGeoGlobalMagField::Instance()->GetField()) {
      auto l3Current = ic.options().get<float>("l3Current");
      auto dipoleCurrent = ic.options().get<float>("dipoleCurrent");
      auto field = o2::field::MagneticField::createFieldMap(l3Current, dipoleCurrent, o2::field::MagneticField::kConvLHC, false,
                                                            3500., "A-A", "$(O2_ROOT)/share/Common/maps/mfchebKGI_sym.root");
      TGeoGlobalMagField::Instance()->SetField(field);
      TGeoGlobalMagField::Instance()->Lock();
      TrackExtrap::setField();
    }

    auto stop = [this]() {
      LOG(INFO) << "track propagation to vertex duration = " << mElapsedTime.count() << " s";
    };
    ic.services().get<CallbackService>().set(CallbackService::Id::Stop, stop);
  }

  //_________________________________________________________________________________________________
  void run(framework::ProcessingContext& pc)
  {
    /// read the tracks with attached clusters of the current event,
    /// propagate them to the corresponding vertex and send the new version

    // get the vertex
    Point3D<double> vertex(0., 0., 0.);
    int eventVtx = readVertex(pc.inputs().get<gsl::span<char>>("vertex"), vertex);

    // get the tracks
    std::list<TrackStruct> tracks{};
    int eventTracks = readTracks(pc.inputs().get<gsl::span<char>>("tracks"), tracks);

    if (eventVtx > -1 && eventVtx != eventTracks) {
      throw runtime_error("vertex and tracks are from different events");
    }

    // propagate the tracks to the vertex
    auto tStart = std::chrono::high_resolution_clock::now();
    for (auto itTrack = tracks.begin(); itTrack != tracks.end();) {
      if (extrapTrackToVertex(*itTrack, vertex)) {
        ++itTrack;
      } else {
        itTrack = tracks.erase(itTrack);
      }
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    mElapsedTime += tEnd - tStart;

    // calculate the size of the payload for the output message, excluding the event header
    int trackSize = getSize(tracks);

    // create the output message
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "TRACKSATVERTEX", 0, Lifetime::Timeframe}, 2 * SSizeOfInt + trackSize);
    auto bufferPtrOut = msgOut.data();

    // write the event header
    writeHeader(eventTracks, trackSize, bufferPtrOut);

    // write the tracks
    if (trackSize > 0) {
      writeTracks(tracks, bufferPtrOut);
    }
  }

 private:
  struct TrackStruct {
    TrackParamStruct paramAtVertex{};
    double dca = 0.;
    double rAbs = 0.;
    TrackParamStruct paramAt1stCluster{};
    double chi2 = 0.;
    std::vector<ClusterStruct> clusters{};
  };

  //_________________________________________________________________________________________________
  int readVertex(const gsl::span<const char>& msgIn, Point3D<double>& vertex) const
  {
    /// get the vertex and return the event number
    /// throw an exception in case of error

    auto bufferPtr = msgIn.data();
    int size = msgIn.size();

    if (size != SSizeOfInt + SSizeOfPoint3D) {
      throw length_error("incorrect payload size");
    }

    const int& event = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;

    vertex = *reinterpret_cast<const Point3D<double>*>(bufferPtr);

    return event;
  }

  //_________________________________________________________________________________________________
  int readTracks(const gsl::span<const char>& msgIn, std::list<TrackStruct>& tracks) const
  {
    /// get the tracks and return the event number
    /// throw an exception in case of error

    auto bufferPtr = msgIn.data();
    int size = msgIn.size();

    if (size < 2 * SSizeOfInt) {
      throw out_of_range("missing event header");
    }
    const int& event = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;
    int sizeLeft = *reinterpret_cast<const int*>(bufferPtr);
    bufferPtr += SSizeOfInt;

    if (sizeLeft != size - 2 * SSizeOfInt) {
      throw length_error("incorrect payload size");
    }

    if (sizeLeft > 0) {

      if (sizeLeft < SSizeOfInt) {
        throw out_of_range("missing number of tracks");
      }
      const int& nTracks = *reinterpret_cast<const int*>(bufferPtr);
      bufferPtr += SSizeOfInt;
      sizeLeft -= SSizeOfInt;

      for (int iTrack = 0; iTrack < nTracks; ++iTrack) {

        tracks.emplace_back();
        auto& track = tracks.back();

        if (sizeLeft < SSizeOfTrackParamStruct) {
          throw out_of_range("missing track parameters");
        }
        track.paramAt1stCluster = *reinterpret_cast<const TrackParamStruct*>(bufferPtr);
        bufferPtr += SSizeOfTrackParamStruct;
        sizeLeft -= SSizeOfTrackParamStruct;

        if (sizeLeft < SSizeOfDouble) {
          throw out_of_range("missing chi2");
        }
        track.chi2 = *reinterpret_cast<const double*>(bufferPtr);
        bufferPtr += SSizeOfDouble;
        sizeLeft -= SSizeOfDouble;

        if (sizeLeft < SSizeOfInt) {
          throw out_of_range("missing number of clusters");
        }
        const int& nClusters = *reinterpret_cast<const int*>(bufferPtr);
        bufferPtr += SSizeOfInt;
        sizeLeft -= SSizeOfInt;

        for (int iCl = 0; iCl < nClusters; ++iCl) {

          if (sizeLeft < SSizeOfClusterStruct) {
            throw out_of_range("missing cluster");
          }
          track.clusters.emplace_back(*reinterpret_cast<const ClusterStruct*>(bufferPtr));
          bufferPtr += SSizeOfClusterStruct;
          sizeLeft -= SSizeOfClusterStruct;
        }
      }

      if (sizeLeft != 0) {
        throw length_error("incorrect payload size");
      }
    }

    return event;
  }

  //_________________________________________________________________________________________________
  bool extrapTrackToVertex(TrackStruct& track, Point3D<double>& vertex)
  {
    /// compute the track parameters at vertex, at DCA and at the end of the absorber

    // convert parameters at first cluster in internal format
    TrackParam trackParam;
    trackParam.setNonBendingCoor(track.paramAt1stCluster.x);
    trackParam.setBendingCoor(track.paramAt1stCluster.y);
    trackParam.setZ(track.paramAt1stCluster.z);
    trackParam.setNonBendingSlope(track.paramAt1stCluster.px / track.paramAt1stCluster.pz);
    trackParam.setBendingSlope(track.paramAt1stCluster.py / track.paramAt1stCluster.pz);
    trackParam.setInverseBendingMomentum(track.paramAt1stCluster.sign / TMath::Sqrt(track.paramAt1stCluster.py * track.paramAt1stCluster.py + track.paramAt1stCluster.pz * track.paramAt1stCluster.pz));

    // extrapolate to vertex
    TrackParam trackParamAtVertex(trackParam);
    if (!TrackExtrap::extrapToVertex(&trackParamAtVertex, vertex.x(), vertex.y(), vertex.z(), 0., 0.)) {
      return false;
    }
    track.paramAtVertex.x = trackParamAtVertex.getNonBendingCoor();
    track.paramAtVertex.y = trackParamAtVertex.getBendingCoor();
    track.paramAtVertex.z = trackParamAtVertex.getZ();
    track.paramAtVertex.px = trackParamAtVertex.px();
    track.paramAtVertex.py = trackParamAtVertex.py();
    track.paramAtVertex.pz = trackParamAtVertex.pz();
    track.paramAtVertex.sign = trackParamAtVertex.getCharge();

    // extrapolate to DCA
    TrackParam trackParamAtDCA(trackParam);
    if (!TrackExtrap::extrapToVertexWithoutBranson(&trackParamAtDCA, vertex.z())) {
      return false;
    }
    double dcaX = trackParamAtDCA.getNonBendingCoor() - vertex.x();
    double dcaY = trackParamAtDCA.getBendingCoor() - vertex.y();
    track.dca = TMath::Sqrt(dcaX * dcaX + dcaY * dcaY);

    // extrapolate to the end of the absorber
    if (!TrackExtrap::extrapToZ(&trackParam, -505.)) {
      return false;
    }
    double xAbs = trackParam.getNonBendingCoor();
    double yAbs = trackParam.getBendingCoor();
    track.rAbs = TMath::Sqrt(xAbs * xAbs + yAbs * yAbs);

    return true;
  }

  //_________________________________________________________________________________________________
  int getSize(const std::list<TrackStruct>& tracks)
  {
    /// calculate the total number of bytes requested to store the tracks

    int size(0);
    for (const auto& track : tracks) {
      size += 2 * SSizeOfTrackParamStruct + 3 * SSizeOfDouble + SSizeOfInt + track.clusters.size() * SSizeOfClusterStruct;
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
  void writeTracks(const std::list<TrackStruct>& tracks, char*& bufferPtr) const
  {
    /// write the track informations in the buffer and move the buffer ptr

    // write the number of tracks
    int nTracks = tracks.size();
    memcpy(bufferPtr, &nTracks, SSizeOfInt);
    bufferPtr += SSizeOfInt;

    for (const auto& track : tracks) {

      // write track parameters at vertex
      memcpy(bufferPtr, &(track.paramAtVertex), SSizeOfTrackParamStruct);
      bufferPtr += SSizeOfTrackParamStruct;

      // write dca
      memcpy(bufferPtr, &(track.dca), SSizeOfDouble);
      bufferPtr += SSizeOfDouble;

      // write rAbs
      memcpy(bufferPtr, &(track.rAbs), SSizeOfDouble);
      bufferPtr += SSizeOfDouble;

      // write track parameters at first cluster
      memcpy(bufferPtr, &(track.paramAt1stCluster), SSizeOfTrackParamStruct);
      bufferPtr += SSizeOfTrackParamStruct;

      // write track chi2
      memcpy(bufferPtr, &(track.chi2), SSizeOfDouble);
      bufferPtr += SSizeOfDouble;

      // write the number of clusters
      int nClusters = track.clusters.size();
      memcpy(bufferPtr, &nClusters, SSizeOfInt);
      bufferPtr += SSizeOfInt;

      // write clusters
      for (const auto& cluster : track.clusters) {
        memcpy(bufferPtr, &cluster, SSizeOfClusterStruct);
        bufferPtr += SSizeOfClusterStruct;
      }
    }
  }

  static constexpr int SSizeOfInt = sizeof(int);
  static constexpr int SSizeOfDouble = sizeof(double);
  static constexpr int SSizeOfTrackParamStruct = sizeof(TrackParamStruct);
  static constexpr int SSizeOfClusterStruct = sizeof(ClusterStruct);
  static constexpr int SSizeOfPoint3D = sizeof(Point3D<double>);

  std::chrono::duration<double> mElapsedTime{}; ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackAtVertexSpec()
{
  return DataProcessorSpec{
    "TrackAtVertex",
    Inputs{InputSpec{"vertex", "MCH", "VERTEX", 0, Lifetime::Timeframe},
           InputSpec{"tracks", "MCH", "TRACKS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{"MCH", "TRACKSATVERTEX", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackAtVertexTask>()},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}}}};
}

} // namespace mch
} // namespace o2
