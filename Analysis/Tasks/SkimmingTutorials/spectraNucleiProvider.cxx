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

#include "AnalysisDataModel/EventSelection.h"

#include <TLorentzVector.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::math_utils::detail;

#include "Framework/runDataProcessing.h"

struct NucleiSpectraProviderTask {

  Produces<aod::LFCollisions> outputCollisions;
  Produces<aod::LFNucleiTracks> outputTracks;

  Configurable<float> vertexZCut{"vertexZCut", 10.0f, "Accepted z-vertex range"};
  Configurable<float> trackEtaCut{"trackEtaCut", 0.8f, "Eta range for tracks"};
  Configurable<float> trackPtCut{"trackPtCut", 0.0f, "Pt range for tracks"};

  Configurable<float> yMin{"yMin", -0.8, "Maximum rapidity"};
  Configurable<float> yMax{"yMax", 0.8, "Minimum rapidity"};
  Configurable<float> yBeam{"yBeam", 0., "Beam rapidity"};

  Configurable<float> nSigmaCutLow{"nSigmaCutLow", -30.0, "Value of the Nsigma cut"};
  Configurable<float> nSigmaCutHigh{"nSigmaCutHigh", +3., "Value of the Nsigma cut"};

  Filter collisionFilter = nabs(aod::collision::posZ) < vertexZCut;
  Filter trackFilter = (nabs(aod::track::eta) < trackEtaCut) && (aod::track::isGlobalTrack == (uint8_t) true) && (aod::track::pt > trackPtCut);

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra,
                                                  aod::pidTPCFullPi, aod::pidTPCFullKa, aod::pidTPCFullPr, aod::pidTPCFullHe,
                                                  aod::pidTOFFullPi, aod::pidTOFFullKa, aod::pidTOFFullPr, aod::pidTOFFullHe,
                                                  aod::TrackSelection>>;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision, TrackCandidates const& tracks)
  {
    bool keepEvent = false;
    if (!collision.alias()[kINT7]) {
      return; //is this correct?
    }
    if (!collision.sel7()) {
      return;
    }
    for (auto& track : tracks) {
      TLorentzVector cutVector{};
      cutVector.SetPtEtaPhiM(track.pt() * 2.0, track.eta(), track.phi(), constants::physics::MassHelium3);
      if (cutVector.Rapidity() < yMin + yBeam || cutVector.Rapidity() > yMax + yBeam) {
        continue;
      }
      if (track.tpcNSigmaHe() > nSigmaCutLow && track.tpcNSigmaHe() < nSigmaCutHigh) {
        keepEvent = true;
        break;
      }
    }
    if (keepEvent) {
      outputCollisions(collision.posZ());
      uint32_t pNsigma = 0xFFFFFF00; //15 bit precision for Nsigma - does this respect the sign?
      outputTracks.reserve(tracks.size());
      for (auto track : tracks) {
        outputTracks(outputCollisions.lastIndex(), track.pt(), track.eta(), track.phi(),
                     //truncateFloatFraction(track.tpcNSigmaEl(), pNsigma), truncateFloatFraction(track.tpcNSigmaMu(), pNsigma),
                     truncateFloatFraction(track.tpcNSigmaPi(), pNsigma), truncateFloatFraction(track.tpcNSigmaKa(), pNsigma),
                     truncateFloatFraction(track.tpcNSigmaPr(), pNsigma), //truncateFloatFraction(track.tpcNSigmaDe(), pNsigma),
                     //truncateFloatFraction(track.tpcNSigmaTr(), pNsigma),
                     truncateFloatFraction(track.tpcNSigmaHe(), pNsigma),
                     //truncateFloatFraction(track.tpcNSigmaAl(), pNsigma),
                     //truncateFloatFraction(track.tofNSigmaEl(), pNsigma), truncateFloatFraction(track.tofNSigmaMu(), pNsigma),
                     truncateFloatFraction(track.tofNSigmaPi(), pNsigma), truncateFloatFraction(track.tofNSigmaKa(), pNsigma),
                     truncateFloatFraction(track.tofNSigmaPr(), pNsigma), //truncateFloatFraction(track.tofNSigmaDe(), pNsigma),
                     //truncateFloatFraction(track.tofNSigmaTr(), pNsigma),
                     truncateFloatFraction(track.tofNSigmaHe(), pNsigma));
        //truncateFloatFraction(track.tofNSigmaAl(), pNsigma));
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<NucleiSpectraProviderTask>(cfgc, TaskName{"nucleispectra-task-skim-provider"})};
  return workflow;
}
