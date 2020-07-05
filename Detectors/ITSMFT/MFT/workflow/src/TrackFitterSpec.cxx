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
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#include "MFTWorkflow/TrackFitterSpec.h"
#include "Field/MagneticField.h"
#include "TGeoGlobalMagField.h"
#include "DetectorsBase/Propagator.h"

#include <stdexcept>
#include <list>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"

#include "MFTTracking/TrackParamMFT.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/FitterTrackMFT.h"
#include "MFTTracking/TrackFitter.h"
#include "MFTTracking/TrackExtrap.h"
#include "MFTTracking/Cluster.h"

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

  auto filename = ic.options().get<std::string>("grp-file");
  const auto grp = o2::parameters::GRPObject::loadFrom(filename.c_str());
  if (grp) {
    mGRP.reset(grp);
    o2::base::Propagator::initFieldFromGRP(grp);
    auto field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());

    double centerMFT[3] = {0, 0, -61.4}; // Field at center of MFT
    mTrackFitter->setBz(field->getBz(centerMFT));

  } else {
    LOG(ERROR) << "Cannot retrieve GRP from the " << filename.c_str() << " file !";
    mState = 0;
  }
  mState = 1;
}

//_________________________________________________________________________________________________
void TrackFitterTask::run(ProcessingContext& pc)
{

  if (mState != 1)
    return;

  auto tracksLTF = pc.inputs().get<gsl::span<o2::mft::TrackLTF>>("tracksltf");
  auto tracksCA = pc.inputs().get<gsl::span<o2::mft::TrackCA>>("tracksca");

  int nTracksCA = 0;
  int nFailedTracksCA = 0;
  int nTracksLTF = 0;
  int nFailedTracksLTF = 0;
  std::vector<o2::mft::FitterTrackMFT> fittertracks(tracksLTF.size() + tracksCA.size());
  std::list<Cluster> clusters;

  // Fit LTF tracks
  for (const auto& track : tracksLTF) {
    auto& temptrack = fittertracks.at(nTracksLTF + nFailedTracksLTF);
    convertTrack(track, temptrack, clusters);
    mTrackFitter->fit(temptrack, false) ? nTracksLTF++ : nFailedTracksLTF++;
  } // end fit LTF tracks

  // Fit CA tracks
  for (const auto& track : tracksCA) {
    auto& temptrack = fittertracks.at(nTracksLTF + nFailedTracksLTF + nTracksCA + nFailedTracksCA);
    convertTrack(track, temptrack, clusters);
    mTrackFitter->fit(temptrack, false) ? nTracksCA++ : nFailedTracksCA++;
  } // end fit CA tracks

  auto& finalMFTtracks = pc.outputs().make<std::vector<o2::mft::TrackMFT>>(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe});
  finalMFTtracks.resize(nTracksLTF + nTracksCA);

  auto nTotalTracks = 0;
  // Convert fitter tracks to the final Standalone MFT Track
  for (const auto& track : fittertracks) {
    if (!track.isRemovable()) {
      auto& temptrack = finalMFTtracks.at(nTotalTracks);
      temptrack.setZ(track.first().getZ());
      temptrack.setParameters(TtoSMatrix5(track.first().getParameters()));
      temptrack.setCovariances(TtoSMatrixSym55(track.first().getCovariances()));
      temptrack.setTrackChi2(track.first().getTrackChi2());
      temptrack.setMCCompLabels(track.getMCCompLabels(), track.getNPoints());
      temptrack.setInvQPtQuadtratic(track.getInvQPtQuadtratic());
      temptrack.setChi2QPtQuadtratic(track.getChi2QPtQuadtratic());
      //finalMFTtracks.back().printMCCompLabels();
      nTotalTracks++;
    }
  }

  LOG(INFO) << "MFTFitter loaded " << tracksLTF.size() << " LTF tracks";
  LOG(INFO) << "MFTFitter loaded " << tracksCA.size() << " CA tracks";
  LOG(INFO) << "MFTFitter pushed " << fittertracks.size() << " tracks";
  LOG(INFO) << "MFTFitter dropped " << nFailedTracksLTF << " LTF tracks";
  LOG(INFO) << "MFTFitter dropped " << nFailedTracksCA << " CA tracks";

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getTrackFitterSpec(bool useMC)
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
    AlgorithmSpec{adaptFromTask<TrackFitterTask>(useMC)},
    Options{
      {"grp-file", VariantType::String, "o2sim_grp.root", {"Name of the output file"}},
    }};
}

//_________________________________________________________________________________________________
template <typename T, typename O, typename C>
void convertTrack(const T& inTrack, O& outTrack, C& clusters)
{
  //auto fittedTrack = FitterTrackMFT();
  //TMatrixD covariances(5,5);
  auto xpos = inTrack.getXCoordinates();
  auto ypos = inTrack.getYCoordinates();
  auto zpos = inTrack.getZCoordinates();
  auto clusterIDs = inTrack.getClustersId();
  auto nClusters = inTrack.getNPoints();
  static int ntrack = 0;

  // Add clusters to Tracker's cluster vector & set fittedTrack cluster range.
  // TODO: get rid of this cluster vector
  for (auto cls = 0; cls < nClusters; cls++) {
    Cluster& tempcluster = clusters.emplace_back(xpos[cls], ypos[cls], zpos[cls], clusterIDs[cls]);
    tempcluster.sigmaY2 = 5.43e-4; // FIXME: Use clusters errors once available
    tempcluster.sigmaX2 = 5.0e-4;
    outTrack.createParamAtCluster(tempcluster);
  }
  outTrack.setMCCompLabels(inTrack.getMCCompLabels(), nClusters);
}

//_________________________________________________________________________________________________
SMatrix55 TtoSMatrixSym55(TMatrixD inMatrix)
{
  // TMatrix to sym SMatrix
  SMatrix55 outMatrix;
  for (auto i = 5; i--;) {
    outMatrix(i, i) = inMatrix(i, i);
    for (auto j = i; j--;) {
      outMatrix(i, j) = inMatrix(i, j);
    }
  }
  return outMatrix;
}

//_________________________________________________________________________________________________
SMatrix5 TtoSMatrix5(TMatrixD inMatrix)
{
  SMatrix5 outMatrix;
  for (auto i = 0; i < 5; i++)
    outMatrix(i) = inMatrix(i, 0);
  return outMatrix;
}

} // namespace mft
} // namespace o2
