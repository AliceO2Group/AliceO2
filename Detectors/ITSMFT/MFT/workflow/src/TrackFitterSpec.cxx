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
#include "DataFormatsITSMFT/Cluster.h"
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

  auto tracksLTF = pc.inputs().get<const std::vector<o2::mft::TrackLTF>>("tracksltf");
  auto tracksCA = pc.inputs().get<const std::vector<o2::mft::TrackCA>>("tracksca");
  int nTracksCA = 0;
  int nTracksLTF = 0;
  std::vector<o2::mft::FitterTrackMFT> fittertracks;
  auto& finalMFTtracks = pc.outputs().make<std::vector<o2::mft::TrackMFT>>(Output{"MFT", "TRACKS", 0, Lifetime::Timeframe});
  std::list<o2::itsmft::Cluster> clusters;

  // Fit LTF tracks
  for (auto track : tracksLTF) {
    o2::mft::FitterTrackMFT& temptrack = fittertracks.emplace_back();
    convertTrack(track, temptrack, clusters);
    mTrackFitter->fit(temptrack, false);
    //LOG(INFO) << "tracksLTF: nTracksLTF  = " << nTracksLTF << " tracks.size() = " << fittertracks.size() << std::endl;
    nTracksLTF++;
  }
  // Fit CA tracks
  for (auto track : tracksCA) {
    o2::mft::FitterTrackMFT& temptrack = fittertracks.emplace_back();
    //o2::itsmft::Cluster& tempcluster = clusters.emplace_back();
    convertTrack(track, temptrack, clusters);
    mTrackFitter->fit(temptrack, false);
    //LOG(INFO) << "tracksCA: nTracksCA  = " << nTracksCA << " tracks.size() = " << fittertracks.size() << std::endl;
    nTracksCA++;
  }

  // Convert fitter tracks to the final Standalone MFT Track
  for (auto track : fittertracks) {
    o2::mft::TrackMFT& temptrack = finalMFTtracks.emplace_back();
    temptrack.setZ(track.first().getZ());
    temptrack.setParameters(track.first().getParameters());
    temptrack.setCovariances(track.first().getCovariances());
    temptrack.setTrackChi2(track.first().getTrackChi2());

    auto tanl = track.first().getTanl();
    auto phi = TMath::Sin(track.first().getPhi());
    LOG(INFO) << "TrackPars: p = " << track.first().getP() << " Tgl = " << tanl << " phi = " << phi << " pz = " << track.first().getPz() << " pt = " << track.first().getPt() << " charge = " << track.first().getCharge() << " Chi2 = " << track.first().getTrackChi2() << std::endl;
  }

  LOG(INFO) << "MFTFitter loaded " << tracksLTF.size() << " LTF tracks";
  LOG(INFO) << "MFTFitter loaded " << tracksCA.size() << " CA tracks";
  LOG(INFO) << "MFTFitter pushed " << fittertracks.size() << " tracks";

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

  //std::cout << "** Converting MFT track with nClusters = " << nClusters << std::endl;
  // Add clusters to Tracker's cluster vector & set fittedTrack cluster range
  for (auto cls = 0; cls < nClusters; cls++) {
    o2::itsmft::Cluster& tempcluster = clusters.emplace_back(clusterIDs[cls], xpos[cls], ypos[cls], zpos[cls]);
    //auto cluster = o2::itsmft::Cluster();
    tempcluster.setSigmaY2(0.0001); // FIXME:
    tempcluster.setSigmaZ2(0.0001); // FIXME: Use clusters errors once available
    outTrack.createParamAtCluster(tempcluster);
    //std::cout << "Adding cluster " << cls << " to track " << ntrack << " with clusterID " << tempcluster.getSensorID() << " at z = " << tempcluster.getZ() << std::endl;
  }

  //std::cout << "  ** outTrack has getNClusters = " << outTrack.getNClusters() << std::endl;
  //for (auto par = outTrack.rbegin(); par != outTrack.rend(); par++)
  //  std::cout << "     getZ() =  " << par->getZ() << std::endl;

  //fit(outTrack, false);
}

} // namespace mft
} // namespace o2
