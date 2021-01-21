// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

// O2 includes
#include "ReconstructionDataFormats/Track.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "MathUtils/Utils.h"
#include "DataModel/LFDerived.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::math_utils::detail;

#include "Framework/runDataProcessing.h"

struct TPCSpectraProviderTask {

  //Produces<aod::LFCollisions> outputCollisions; //currently it seems in the spectraTPC task no loop over the collision is made. Leave this here in case it will be added
  Produces<aod::LFTracks> outputTracks;

  Configurable<float> vertexZCut{"vertexZCut", 10.0f, "Accepted z-vertex range"};
  Configurable<float> trackEtaCut{"trackEtaCut", 0.9f, "Eta range for tracks"};
  Configurable<float> trackPtCut{"trackPtCut", 0.0f, "Pt range for tracks"};

  Filter collisionFilter = nabs(aod::collision::posZ) < vertexZCut;
  Filter trackFilter = (nabs(aod::track::eta) < trackEtaCut) && (aod::track::isGlobalTrack == (uint8_t) true) && (aod::track::pt > trackPtCut);

  Configurable<float> nsigmacut{"nsigmacut", 3, "Value of the Nsigma cut"}; //can we add an upper limit?

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::pidRespTPC, aod::TrackSelection>>;
  void process(soa::Filtered<aod::Collisions>::iterator const& collision, TrackCandidates const& tracks)
  {
    uint32_t pNsigma = 0xFFFFFF00; //15 bit precision for Nsigma
    outputTracks.reserve(tracks.size());
    for (auto track : tracks) {
      float nsigma[9] = {truncateFloatFraction(track.tpcNSigmaEl(), pNsigma), truncateFloatFraction(track.tpcNSigmaMu(), pNsigma),
                         truncateFloatFraction(track.tpcNSigmaPi(), pNsigma), truncateFloatFraction(track.tpcNSigmaKa(), pNsigma),
                         truncateFloatFraction(track.tpcNSigmaPr(), pNsigma), truncateFloatFraction(track.tpcNSigmaDe(), pNsigma),
                         truncateFloatFraction(track.tpcNSigmaTr(), pNsigma), truncateFloatFraction(track.tpcNSigmaHe(), pNsigma),
                         truncateFloatFraction(track.tpcNSigmaAl(), pNsigma)}; //the significance needs to be discussed

      //outputTracks(outputCollisions.lastIndex(), track.pt(), track.p(), track.eta(), nsigma);
      outputTracks(track.pt(), track.p(), track.eta(), nsigma);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec workflow{adaptAnalysisTask<TPCSpectraProviderTask>("tpcspectra-task-skim-provider")};
  return workflow;
}
