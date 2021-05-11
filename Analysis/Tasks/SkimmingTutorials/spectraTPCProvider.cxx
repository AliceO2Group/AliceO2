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
#include "AnalysisDataModel/EventSelection.h"
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
  Produces<aod::LFSmallTracks> outputSmallTracks;
  Produces<aod::LFSingleTracks> outputSingleTracks;

  Configurable<float> vertexZCut{"vertexZCut", 10.0f, "Accepted z-vertex range"};
  Configurable<float> trackEtaCut{"trackEtaCut", 0.8f, "Eta range for tracks"};
  Configurable<float> trackPtCut{"trackPtCut", 0.5f, "Pt range for tracks"};

  Configurable<bool> saveTracks{"saveTracks", false, "Save large LF tracks table"};
  Configurable<bool> saveSmallTracks{"saveSmallTracks", false, "Save small LF tracks table"};
  Configurable<bool> saveSingleTracks{"saveSingleTracks", false, "Save single species LF tracks table"};
  Configurable<int> species1{"species1", 0, "First particle species to be kept"};
  Configurable<bool> species1NsigmaSelection{"species1NsigmaSelection", false, "select on species 1 Nsigma"};
  Configurable<float> species1NsigmaLow{"species1NsigmaLow", -3., "species 1 Nsigma Lower Bound"};
  Configurable<float> species1NsigmaHigh{"species1NsigmaHigh", 3., "species 1 Nsigma Higher Bound"};
  Configurable<int> species2{"species2", 0, "Second particle species to be kept"};
  Configurable<bool> species2NsigmaSelection{"species2NsigmaSelection", false, "select on species 2 Nsigma"};
  Configurable<float> species2NsigmaLow{"species2NsigmaLow", -3., "species 2 Nsigma Lower Bound"};
  Configurable<float> species2NsigmaHigh{"species2NsigmaHigh", 3., "species 2 Nsigma Higher Bound"};
  Configurable<int> species3{"species3", 0, "Third particle species to be kept"};
  Configurable<bool> species3NsigmaSelection{"species3NsigmaSelection", false, "select on species 3 Nsigma"};
  Configurable<float> species3NsigmaLow{"species3NsigmaLow", -3., "species 3 Nsigma Lower Bound"};
  Configurable<float> species3NsigmaHigh{"species3NsigmaHigh", 3., "species 3 Nsigma Higher Bound"};
  Configurable<int> species4{"species4", 0, "Fourth particle species to be kept"};
  Configurable<bool> species4NsigmaSelection{"species4NsigmaSelection", false, "select on species 4 Nsigma"};
  Configurable<float> species4NsigmaLow{"species4NsigmaLow", -3., "species 4 Nsigma Lower Bound"};
  Configurable<float> species4NsigmaHigh{"species4NsigmaHigh", 3., "species 4 Nsigma Higher Bound"};

  template <typename T>
  float SpeciesNSigma(const T& track, int specie)
  {
    uint32_t pNsigma = 0xFFFFFF00; //15 bit precision for Nsigma
    if (specie == 0) {
      return track.tpcNSigmaEl();
    } else if (specie == 1) {
      return track.tpcNSigmaMu();
    } else if (specie == 2) {
      return track.tpcNSigmaPi();
    } else if (specie == 3) {
      return track.tpcNSigmaKa();
    } else if (specie == 4) {
      return track.tpcNSigmaPr();
    } else if (specie == 5) {
      return track.tpcNSigmaDe();
    } else if (specie == 6) {
      return truncateFloatFraction(track.tpcNSigmaTr(), pNsigma);
    } else if (specie == 7) {
      return truncateFloatFraction(track.tpcNSigmaHe(), pNsigma);
    } else if (specie == 8) {
      return truncateFloatFraction(track.tpcNSigmaAl(), pNsigma);
    } else {
      return -99.;
    }
  }

  Filter collisionFilter = nabs(aod::collision::posZ) < vertexZCut;
  Filter trackFilter = (nabs(aod::track::eta) < trackEtaCut) && (aod::track::pt > trackPtCut) && (aod::track::isGlobalTrack == (uint8_t) true);

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra,
                                                  aod::pidTPCFullEl, aod::pidTPCFullMu, aod::pidTPCFullPi,
                                                  aod::pidTPCFullKa, aod::pidTPCFullPr, aod::pidTPCFullDe,
                                                  aod::pidTPCFullTr, aod::pidTPCFullHe, aod::pidTPCFullAl,
                                                  aod::TrackSelection>>;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision, TrackCandidates const& tracks)
  {
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    bool acceptedPID = true;
    float species1NSigma = -99.;
    float species2NSigma = -99.;
    float species3NSigma = -99.;
    float species4NSigma = -99.;

    for (auto& track : tracks) {

      if (saveSmallTracks || saveSingleTracks) {
        species1NSigma = SpeciesNSigma(track, species1);
      }
      if (saveSmallTracks) {
        species2NSigma = SpeciesNSigma(track, species2);
        species3NSigma = SpeciesNSigma(track, species3);
        species4NSigma = SpeciesNSigma(track, species4);
      }
      acceptedPID = false;
      if (!species1NsigmaSelection && !species2NsigmaSelection && !species3NsigmaSelection && !species4NsigmaSelection) {
        acceptedPID = true;
      } else if (species1NsigmaSelection && species1NSigma > species1NsigmaLow && species1NSigma < species1NsigmaHigh) {
        acceptedPID = true;
      } else if (species2NsigmaSelection && species2NSigma > species2NsigmaLow && species2NSigma < species2NsigmaHigh) {
        acceptedPID = true;
      } else if (species3NsigmaSelection && species3NSigma > species3NsigmaLow && species3NSigma < species3NsigmaHigh) {
        acceptedPID = true;
      } else if (species4NsigmaSelection && species4NSigma > species4NsigmaLow && species4NSigma < species4NsigmaHigh) {
        acceptedPID = true;
      }

      if (!acceptedPID) {
        continue;
      }

      if (saveTracks) {
        outputTracks(track.pt(), track.eta(), track.phi(),
                     SpeciesNSigma(track, 0), SpeciesNSigma(track, 1),
                     SpeciesNSigma(track, 2), SpeciesNSigma(track, 3),
                     SpeciesNSigma(track, 4), SpeciesNSigma(track, 5),
                     SpeciesNSigma(track, 6), SpeciesNSigma(track, 7),
                     SpeciesNSigma(track, 8));
      }
      if (saveSmallTracks) {
        outputSmallTracks(track.pt(), track.eta(), track.phi(),
                          species1NSigma, species2NSigma,
                          species3NSigma, species4NSigma);
      }
      if (saveSingleTracks) {
        outputSingleTracks(track.pt(), track.eta(), track.phi(),
                           species1NSigma);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<TPCSpectraProviderTask>(cfgc, TaskName{"tpcspectra-task-skim-provider"})};
  return workflow;
}
