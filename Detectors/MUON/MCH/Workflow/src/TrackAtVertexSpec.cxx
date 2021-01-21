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

#include <gsl/span>

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
#include "MathUtils/Cartesian.h"
#include "Field/MagneticField.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/TrackBlock.h"
#include "MCHTracking/TrackParam.h"
#include "MCHTracking/TrackExtrap.h"

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
      o2::base::GeometryManager::loadGeometry("O2geometry.root");
      if (!gGeoManager) {
        throw std::runtime_error("cannot load the geometry");
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
    /// propagate the MCH tracks to the vertex for each event in the TF and send the results

    // get the ROFs, tracks and vertices
    auto rofs = pc.inputs().get<gsl::span<ROFRecord>>("rofs");
    auto tracks = pc.inputs().get<gsl::span<TrackMCH>>("tracks");
    auto vertices = pc.inputs().get<gsl::span<math_utils::Point3D<double>>>("vertices");

    if (vertices.size() != rofs.size()) {
      throw length_error("number of vertices different from number of events");
    }

    // for each event, propagate the tracks to the vertex
    mTracksAtVtx.clear();
    int iVertex(-1);
    int nTracksTot(0);
    for (const auto& rof : rofs) {
      auto tStart = std::chrono::high_resolution_clock::now();
      nTracksTot += extrapTracksToVertex(tracks.subspan(rof.getFirstIdx(), rof.getNEntries()), vertices[++iVertex]);
      auto tEnd = std::chrono::high_resolution_clock::now();
      mElapsedTime += tEnd - tStart;
    }

    // create the output message
    auto msgOut = pc.outputs().make<char>(Output{"MCH", "TRACKSATVERTEX", 0, Lifetime::Timeframe},
                                          mTracksAtVtx.size() * sizeof(int) + nTracksTot * sizeof(TrackAtVtxStruct));

    // write the tracks
    writeTracks(msgOut.data());
  }

 private:
  struct TrackAtVtxStruct {
    TrackParamStruct paramAtVertex{};
    double dca = 0.;
    double rAbs = 0.;
    int mchTrackIdx = 0;
  };

  //_________________________________________________________________________________________________
  int extrapTracksToVertex(gsl::span<const TrackMCH> tracks, const math_utils::Point3D<double>& vertex)
  {
    /// compute the tracks parameters at vertex, at DCA and at the end of the absorber
    /// return the number of tracks successfully propagated to the vertex

    auto& tracksAtVtx = mTracksAtVtx.emplace_back();
    int trackIdx(-1);

    for (const auto& track : tracks) {

      // create a new track at vertex pointing to the current track (index within the current event)
      auto& trackAtVtx = tracksAtVtx.emplace_back();
      trackAtVtx.mchTrackIdx = ++trackIdx;

      // extrapolate to vertex
      TrackParam trackParamAtVertex(track.getZ(), track.getParameters());
      if (!TrackExtrap::extrapToVertex(&trackParamAtVertex, vertex.x(), vertex.y(), vertex.z(), 0., 0.)) {
        tracksAtVtx.pop_back();
        continue;
      }
      trackAtVtx.paramAtVertex.x = trackParamAtVertex.getNonBendingCoor();
      trackAtVtx.paramAtVertex.y = trackParamAtVertex.getBendingCoor();
      trackAtVtx.paramAtVertex.z = trackParamAtVertex.getZ();
      trackAtVtx.paramAtVertex.px = trackParamAtVertex.px();
      trackAtVtx.paramAtVertex.py = trackParamAtVertex.py();
      trackAtVtx.paramAtVertex.pz = trackParamAtVertex.pz();
      trackAtVtx.paramAtVertex.sign = trackParamAtVertex.getCharge();

      // extrapolate to DCA
      TrackParam trackParamAtDCA(track.getZ(), track.getParameters());
      if (!TrackExtrap::extrapToVertexWithoutBranson(&trackParamAtDCA, vertex.z())) {
        tracksAtVtx.pop_back();
        continue;
      }
      double dcaX = trackParamAtDCA.getNonBendingCoor() - vertex.x();
      double dcaY = trackParamAtDCA.getBendingCoor() - vertex.y();
      trackAtVtx.dca = TMath::Sqrt(dcaX * dcaX + dcaY * dcaY);

      // extrapolate to the end of the absorber
      TrackParam trackParamAtRAbs(track.getZ(), track.getParameters());
      if (!TrackExtrap::extrapToZ(&trackParamAtRAbs, -505.)) {
        tracksAtVtx.pop_back();
        continue;
      }
      double xAbs = trackParamAtRAbs.getNonBendingCoor();
      double yAbs = trackParamAtRAbs.getBendingCoor();
      trackAtVtx.rAbs = TMath::Sqrt(xAbs * xAbs + yAbs * yAbs);
    }

    return tracksAtVtx.size();
  }

  //_________________________________________________________________________________________________
  void writeTracks(char* bufferPtr) const
  {
    /// write the track informations for each event in the message payload

    for (const auto& tracksAtVtx : mTracksAtVtx) {

      // write the number of tracks
      int nTracks = tracksAtVtx.size();
      memcpy(bufferPtr, &nTracks, sizeof(int));
      bufferPtr += sizeof(int);

      // write the tracks
      if (nTracks > 0) {
        memcpy(bufferPtr, tracksAtVtx.data(), nTracks * sizeof(TrackAtVtxStruct));
        bufferPtr += nTracks * sizeof(TrackAtVtxStruct);
      }
    }
  }

  std::vector<std::vector<TrackAtVtxStruct>> mTracksAtVtx{}; ///< list of tracks extrapolated to vertex for each event
  std::chrono::duration<double> mElapsedTime{};              ///< timer
};

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackAtVertexSpec()
{
  return DataProcessorSpec{
    "TrackAtVertex",
    Inputs{InputSpec{"vertices", "MCH", "VERTICES", 0, Lifetime::Timeframe},
           InputSpec{"rofs", "MCH", "TRACKROFS", 0, Lifetime::Timeframe},
           InputSpec{"tracks", "MCH", "TRACKS", 0, Lifetime::Timeframe},
           InputSpec{"clusters", "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe}},
    Outputs{OutputSpec{{"tracksatvertex"}, "MCH", "TRACKSATVERTEX", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TrackAtVertexTask>()},
    Options{{"l3Current", VariantType::Float, -30000.0f, {"L3 current"}},
            {"dipoleCurrent", VariantType::Float, -6000.0f, {"Dipole current"}}}};
}

} // namespace mch
} // namespace o2
