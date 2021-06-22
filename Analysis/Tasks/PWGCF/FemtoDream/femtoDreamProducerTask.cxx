// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file femtoDreamProducerTask.cxx
/// \brief Tasks that produces the track tables used for the pairing
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "include/FemtoDream/FemtoDreamCollisionSelection.h"
#include "include/FemtoDream/FemtoDreamTrackSelection.h"
#include "include/FemtoDream/FemtoDerived.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "Framework/HistogramRegistry.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "ReconstructionDataFormats/Track.h"

using namespace o2;
using namespace o2::analysis::femtoDream;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
using FilteredFullCollision = soa::Filtered<soa::Join<aod::Collisions,
                                                      aod::EvSels,
                                                      aod::Mults>>::iterator;
using FilteredFullTracks = soa::Filtered<soa::Join<aod::FullTracks,
                                                   aod::TracksExtended,
                                                   aod::pidTPCEl, aod::pidTPCMu, aod::pidTPCPi,
                                                   aod::pidTPCKa, aod::pidTPCPr, aod::pidTPCDe,
                                                   aod::pidTOFEl, aod::pidTOFMu, aod::pidTOFPi,
                                                   aod::pidTOFKa, aod::pidTOFPr, aod::pidTOFDe>>;
} // namespace o2::aod

struct femtoDreamProducerTask {

  Produces<aod::FemtoDreamCollisions> outputCollision;
  Produces<aod::FemtoDreamParticles> outputTracks;
  Produces<aod::FemtoDreamDebugParticles> outputDebugTracks;

  Configurable<bool> ConfDebugOutput{"ConfDebugOutput", true, "Debug output"};

  /// Event cuts
  FemtoDreamCollisionSelection colCuts;
  Configurable<float> ConfEvtZvtx{"ConfEvtZvtx", 10.f, "Evt sel: Max. z-Vertex (cm)"};
  Configurable<bool> ConfEvtTriggerCheck{"ConfEvtTriggerCheck", false, "Evt sel: check for trigger"};
  Configurable<int> ConfEvtTriggerSel{"ConfEvtTriggerSel", kINT7, "Evt sel: trigger"};
  Configurable<bool> ConfEvtOfflineCheck{"ConfEvtOfflineCheck", false, "Evt sel: check for offline selection"};

  Filter colFilter = nabs(aod::collision::posZ) < ConfEvtZvtx;

  FemtoDreamTrackSelection trackCuts;
  Configurable<std::vector<float>> ConfTrkCharge{"ConfTrkCharge", std::vector<float>{-1, 1}, "Trk sel: Charge"};
  Configurable<std::vector<float>> ConfTrkPtmin{"ConfTrkPtmin", std::vector<float>{0.4f, 0.6f, 0.5f}, "Trk sel: Min. pT (GeV/c)"};
  Configurable<std::vector<float>> ConfTrkPtmax{"ConfTrkPtmax", std::vector<float>{4.05f, 999.f}, "Trk sel: Max. pT (GeV/c)"};
  Configurable<std::vector<float>> ConfTrkEta{"ConfTrkEta", std::vector<float>{0.8f, 0.7f, 0.9f}, "Trk sel: Max. eta"};
  Configurable<std::vector<float>> ConfTrkTPCnclsMin{"ConfTrkTPCnclsMin", std::vector<float>{80.f, 70.f, 60.f}, "Trk sel: Min. nCls TPC"};
  Configurable<std::vector<float>> ConfTrkTPCfCls{"ConfTrkTPCfCls", std::vector<float>{0.7f, 0.83f, 0.9f}, "Trk sel: Min. ratio crossed rows/findable"};
  Configurable<std::vector<float>> ConfTrkTPCsCls{"ConfTrkTPCsCls", std::vector<float>{0.1f, 160.f}, "Trk sel: Max. TPC shared cluster"};
  Configurable<std::vector<float>> ConfTrkDCAxyMax{"ConfTrkDCAxyMax", std::vector<float>{0.1f, 3.5f}, "Trk sel: Max. DCA_xy (cm)"}; /// here we need an open cut to do the DCA fits later on!
  Configurable<std::vector<float>> ConfTrkDCAzMax{"ConfTrkDCAzMax", std::vector<float>{0.2f, 3.5f}, "Trk sel: Max. DCA_z (cm)"};
  /// \todo maybe we need to remove the PID from the general cut container and have a separate one, these are lots and lots of bits we need
  Configurable<std::vector<float>> ConfTrkPIDnSigmaMax{"ConfTrkPIDnSigmaMax", std::vector<float>{3.5f, 3.f, 2.5f}, "Trk sel: Max. PID nSigma"};
  Configurable<std::vector<int>> ConfTrkTPIDspecies{"ConfTrkTPIDspecies", std::vector<int>{o2::track::PID::Electron, o2::track::PID::Pion, o2::track::PID::Kaon, o2::track::PID::Proton, o2::track::PID::Deuteron}, "Trk sel: Particles species for PID"};

  // for now this selection does not work yet, however will very soon
  // \todo once this is the case, set the limits less strict!
  MutableConfigurable<float> TrackMinSelPtMin{"TrackMinSelPtMin", 0.4f, "(automatic) Minimal pT selection for tracks"};
  MutableConfigurable<float> TrackMinSelPtMax{"TrackMinSelPtMax", 10.f, "(automatic) Maximal pT selection for tracks"};
  MutableConfigurable<float> TrackMinSelEtaMax{"TrackMinSelEtaMax", 1.f, "(automatic) Maximal eta selection for tracks"};

  Filter trackFilter = (aod::track::pt > TrackMinSelPtMin.value) &&
                       (aod::track::pt < TrackMinSelPtMax.value) &&
                       (nabs(aod::track::eta) < TrackMinSelEtaMax.value);

  HistogramRegistry qaRegistry{"QAHistos", {}, OutputObjHandlingPolicy::QAObject};

  void init(InitContext&)
  {
    colCuts.setCuts(ConfEvtZvtx, ConfEvtTriggerCheck, ConfEvtTriggerSel, ConfEvtOfflineCheck);
    colCuts.init(&qaRegistry);

    trackCuts.setSelection(ConfTrkCharge, femtoDreamTrackSelection::kSign, femtoDreamSelection::kEqual);
    trackCuts.setSelection(ConfTrkPtmin, femtoDreamTrackSelection::kpTMin, femtoDreamSelection::kLowerLimit);
    trackCuts.setSelection(ConfTrkPtmax, femtoDreamTrackSelection::kpTMax, femtoDreamSelection::kUpperLimit);
    trackCuts.setSelection(ConfTrkEta, femtoDreamTrackSelection::kEtaMax, femtoDreamSelection::kAbsUpperLimit);
    trackCuts.setSelection(ConfTrkTPCnclsMin, femtoDreamTrackSelection::kTPCnClsMin, femtoDreamSelection::kLowerLimit);
    trackCuts.setSelection(ConfTrkTPCfCls, femtoDreamTrackSelection::kTPCfClsMin, femtoDreamSelection::kLowerLimit);
    trackCuts.setSelection(ConfTrkTPCsCls, femtoDreamTrackSelection::kTPCsClsMax, femtoDreamSelection::kUpperLimit);
    trackCuts.setSelection(ConfTrkDCAxyMax, femtoDreamTrackSelection::kDCAxyMax, femtoDreamSelection::kAbsUpperLimit);
    trackCuts.setSelection(ConfTrkDCAzMax, femtoDreamTrackSelection::kDCAzMax, femtoDreamSelection::kAbsUpperLimit);
    trackCuts.setSelection(ConfTrkPIDnSigmaMax, femtoDreamTrackSelection::kPIDnSigmaMax, femtoDreamSelection::kAbsUpperLimit);
    trackCuts.setPIDSpecies(ConfTrkTPIDspecies);
    trackCuts.init(&qaRegistry);

    if (trackCuts.getNSelections(femtoDreamTrackSelection::kpTMin) > 0) {
      TrackMinSelPtMin.value = trackCuts.getMinimalSelection(femtoDreamTrackSelection::kpTMin, femtoDreamSelection::kLowerLimit);
    }
    if (trackCuts.getNSelections(femtoDreamTrackSelection::kpTMax) > 0) {
      TrackMinSelPtMax.value = trackCuts.getMinimalSelection(femtoDreamTrackSelection::kpTMax, femtoDreamSelection::kUpperLimit);
    }
    if (trackCuts.getNSelections(femtoDreamTrackSelection::kEtaMax) > 0) {
      TrackMinSelEtaMax.value = trackCuts.getMinimalSelection(femtoDreamTrackSelection::kEtaMax, femtoDreamSelection::kAbsUpperLimit);
    }
  }

  void process(aod::FilteredFullCollision const& col,
               aod::FilteredFullTracks const& tracks)
  {

    if (!colCuts.isSelected(col)) {
      return;
    }
    const auto vtxZ = col.posZ();
    const auto mult = col.multV0M();
    const auto spher = colCuts.computeSphericity(col, tracks);
    colCuts.fillQA(col);
    outputCollision(vtxZ, mult, spher);

    for (auto& track : tracks) {
      if (!trackCuts.isSelectedMinimal(track)) {
        continue;
      }
      trackCuts.fillQA(track);
      auto cutContainer = trackCuts.getCutContainer(track);
      if (cutContainer > 0) {
        trackCuts.fillCutQA(track, cutContainer);
        outputTracks(outputCollision.lastIndex(), track.pt(), track.eta(), track.phi(), cutContainer, track.dcaXY());
        if (ConfDebugOutput) {
          outputDebugTracks(outputCollision.lastIndex(),
                            track.sign(), track.tpcNClsFound(),
                            track.tpcNClsFindable(),
                            track.tpcNClsCrossedRows(), track.tpcNClsShared(), track.dcaXY(), track.dcaZ(),
                            track.tpcNSigmaEl(), track.tpcNSigmaPi(), track.tpcNSigmaKa(), track.tpcNSigmaPr(), track.tpcNSigmaDe(),
                            track.tofNSigmaEl(), track.tofNSigmaPi(), track.tofNSigmaKa(), track.tofNSigmaPr(), track.tofNSigmaDe());
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<femtoDreamProducerTask>(cfgc)};
  return workflow;
}
