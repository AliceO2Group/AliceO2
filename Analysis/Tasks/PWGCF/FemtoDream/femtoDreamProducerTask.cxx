// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file femtoDreamProducerTask.cxx
/// \brief Tasks that produces the track tables used for the pairing
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "include/FemtoDream/FemtoDreamCollisionSelection.h"
#include "include/FemtoDream/FemtoDreamTrackSelection.h"
#include "include/FemtoDream/FemtoDreamV0Selection.h"
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
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisDataModel/StrangenessTables.h"
#include "Math/Vector4D.h"
#include "TMath.h"

using namespace o2;
using namespace o2::analysis::femtoDream;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
using FilteredFullCollision = soa::Filtered<soa::Join<aod::Collisions,
                                                      aod::EvSels,
                                                      aod::Mults>>::iterator;
using FilteredFullTracks = soa::Join<aod::FullTracks,
                                     aod::TracksExtended,
                                     aod::pidTPCEl, aod::pidTPCMu, aod::pidTPCPi,
                                     aod::pidTPCKa, aod::pidTPCPr, aod::pidTPCDe,
                                     aod::pidTOFEl, aod::pidTOFMu, aod::pidTOFPi,
                                     aod::pidTOFKa, aod::pidTOFPr, aod::pidTOFDe>;
using FilteredFullV0s = soa::Filtered<aod::V0Datas>; /// predefined Join table for o2::aod::V0s = soa::Join<o2::aod::TransientV0s, o2::aod::StoredV0s> to be used when we add v0Filter
} // namespace o2::aod

/// \todo fix how to pass array to setSelection, getRow() passing a different type!
// static constexpr float arrayV0Sel[3][3] = {{100.f, 100.f, 100.f}, {0.2f, 0.2f, 0.2f}, {100.f, 100.f, 100.f}};
// unsigned int rows = sizeof(arrayV0Sel) / sizeof(arrayV0Sel[0]);
// unsigned int columns = sizeof(arrayV0Sel[0]) / sizeof(arrayV0Sel[0][0]);

template <typename T>
int getRowDaughters(int daughID, T const& vecID)
{
  int rowInPrimaryTrackTableDaugh = -1;
  for (size_t i = 0; i < vecID.size(); i++) {
    if (vecID.at(i) == daughID) {
      rowInPrimaryTrackTableDaugh = i;
      break;
    }
  }
  return rowInPrimaryTrackTableDaugh;
}

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

  FemtoDreamV0Selection v0Cuts;
  /// \todo fix how to pass array to setSelection, getRow() passing a different type!
  // Configurable<LabeledArray<float>> ConfV0Selection{"ConfV0Selection", {arrayV0Sel[0], 3, 3,
  // {"V0 sel: Max. distance from Vtx (cm)",
  // "V0 sel: Min. transverse radius (cm)",
  // "V0 sel: Max. transverse radius (cm)"},
  // {"lower", "default", "upper"}}, "Labeled array for V0 selection"};

  Configurable<std::vector<float>> ConfDCAV0DaughMax{"ConfDCAV0DaughMax", std::vector<float>{1.2f, 1.5f}, "V0 sel: Max. DCA daugh from SV (cm)"};
  Configurable<std::vector<float>> ConfCPAV0Min{"ConfCPAV0Min", std::vector<float>{0.9f, 0.995f}, "V0 sel: Min. CPA"};
  MutableConfigurable<float> V0DecVtxMax{"V0DecVtxMax", 100.f, "V0 sel: Max. distance from Vtx (cm)"};
  MutableConfigurable<float> V0TranRadV0Min{"V0TranRadV0Min", 0.2f, "V0 sel: Min. transverse radius (cm)"};
  MutableConfigurable<float> V0TranRadV0Max{"V0TranRadV0Max", 100.f, "V0 sel: Max. transverse radius (cm)"};

  Configurable<std::vector<float>> ConfV0DaughTPCnclsMin{"ConfV0DaughTPCnclsMin", std::vector<float>{80.f, 70.f, 60.f}, "V0 Daugh sel: Min. nCls TPC"};
  Configurable<std::vector<float>> ConfV0DaughDCAMax{"ConfV0DaughDCAMax", std::vector<float>{0.05f, 0.06f}, "V0 Daugh sel:  Max. DCA Daugh to PV (cm)"};
  Configurable<std::vector<float>> ConfV0DaughPIDnSigmaMax{"ConfV0DaughPIDnSigmaMax", std::vector<float>{5.f, 4.f}, "V0 Daugh sel: Max. PID nSigma TPC"};

  /// \todo should we add filter on min value pT/eta of V0 and daughters?
  Filter v0Filter = (nabs(aod::v0data::x) < V0DecVtxMax.value) &&
                    (nabs(aod::v0data::y) < V0DecVtxMax.value) &&
                    (nabs(aod::v0data::z) < V0DecVtxMax.value);
  // (aod::v0data::v0radius > V0TranRadV0Min.value); to be added, not working for now do not know why

  HistogramRegistry qaRegistry{"QAHistos", {}, OutputObjHandlingPolicy::QAObject};

  void init(InitContext&)
  {
    colCuts.setCuts(ConfEvtZvtx, ConfEvtTriggerCheck, ConfEvtTriggerSel, ConfEvtOfflineCheck);
    colCuts.init(&qaRegistry);

    trackCuts.setSelection(ConfTrkCharge, femtoDreamTrackSelection::kSign, femtoDreamSelection::kEqual);
    trackCuts.setSelection(ConfTrkTPCnclsMin, femtoDreamTrackSelection::kTPCnClsMin, femtoDreamSelection::kLowerLimit);
    trackCuts.setSelection(ConfTrkTPCfCls, femtoDreamTrackSelection::kTPCfClsMin, femtoDreamSelection::kLowerLimit);
    trackCuts.setSelection(ConfTrkTPCsCls, femtoDreamTrackSelection::kTPCsClsMax, femtoDreamSelection::kUpperLimit);
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

    /// \todo fix how to pass array to setSelection, getRow() passing a different type!
    // v0Cuts.setSelection(ConfV0Selection->getRow(0), femtoDreamV0Selection::kDecVtxMax, femtoDreamSelection::kAbsUpperLimit);

    v0Cuts.setSelection(ConfDCAV0DaughMax, femtoDreamV0Selection::kDCAV0DaughMax, femtoDreamSelection::kUpperLimit);
    v0Cuts.setSelection(ConfCPAV0Min, femtoDreamV0Selection::kCPAV0Min, femtoDreamSelection::kLowerLimit);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kPosTrack, ConfTrkCharge, femtoDreamTrackSelection::kSign, femtoDreamSelection::kEqual);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kPosTrack, ConfV0DaughTPCnclsMin, femtoDreamTrackSelection::kTPCnClsMin, femtoDreamSelection::kLowerLimit);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kPosTrack, ConfV0DaughDCAMax, femtoDreamTrackSelection::kDCAzMax, femtoDreamSelection::kAbsUpperLimit);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kPosTrack, ConfV0DaughPIDnSigmaMax, femtoDreamTrackSelection::kPIDnSigmaMax, femtoDreamSelection::kAbsUpperLimit);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kNegTrack, ConfTrkCharge, femtoDreamTrackSelection::kSign, femtoDreamSelection::kEqual);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kNegTrack, ConfV0DaughTPCnclsMin, femtoDreamTrackSelection::kTPCnClsMin, femtoDreamSelection::kLowerLimit);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kNegTrack, ConfV0DaughDCAMax, femtoDreamTrackSelection::kDCAzMax, femtoDreamSelection::kAbsUpperLimit);
    v0Cuts.setChildCuts(femtoDreamV0Selection::kNegTrack, ConfV0DaughPIDnSigmaMax, femtoDreamTrackSelection::kPIDnSigmaMax, femtoDreamSelection::kAbsUpperLimit);
    v0Cuts.init(&qaRegistry);
  }

  void process(aod::FilteredFullCollision const& col,
               aod::FilteredFullTracks const& tracks,
               o2::aod::V0Datas const& fullV0s) /// \todo with FilteredFullV0s
  {
    if (!colCuts.isSelected(col)) {
      return;
    }
    const auto vtxZ = col.posZ();
    const auto mult = col.multV0M();
    const auto spher = colCuts.computeSphericity(col, tracks);
    colCuts.fillQA(col);
    outputCollision(vtxZ, mult, spher);

    int childIDs[2] = {0, 0};
    std::vector<int> tmpIDtrack;
    float temptrack[2];
    std::vector<float> temptrackPt;
    std::vector<float> tempPostrackPt;

    for (auto& track : tracks) {
      if (!trackCuts.isSelectedMinimal(track)) {
        continue;
      }
      trackCuts.fillQA(track);
      auto cutContainer = trackCuts.getCutContainer(track);
      if (cutContainer > 0) {
        trackCuts.fillCutQA(track, cutContainer);
        outputTracks(outputCollision.lastIndex(), track.pt(), track.eta(), track.phi(), aod::femtodreamparticle::ParticleType::kTrack, cutContainer, track.dcaXY(), childIDs);
        tmpIDtrack.push_back(track.globalIndex());
        temptrackPt.push_back(track.pt());
        temptrack[0] = outputTracks.lastIndex();
        temptrack[1] = track.pt();
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

    for (auto& v0 : fullV0s) {
      auto postrack = v0.posTrack_as<aod::FilteredFullTracks>();
      auto negtrack = v0.negTrack_as<aod::FilteredFullTracks>(); ///\tocheck funnily enough if we apply the filter the sign of Pos and Neg track is always negative
      if (!v0Cuts.isSelectedMinimal(col, v0, postrack, negtrack)) {
        continue;
      }
      v0Cuts.fillQA(col, v0); ///\todo fill QA also for daughters
      auto cutContainerV0 = v0Cuts.getCutContainer(col, v0, postrack, negtrack);
      if ((cutContainerV0.at(0) > 0) && (cutContainerV0.at(1) > 0) && (cutContainerV0.at(2) > 0)) {
        int postrackID = v0.posTrackId();
        int rowInPrimaryTrackTablePos = -1;
        rowInPrimaryTrackTablePos = getRowDaughters(postrackID, tmpIDtrack);
        childIDs[0] = rowInPrimaryTrackTablePos;
        childIDs[1] = 0;
        ROOT::Math::PxPyPzMVector postrackVec(v0.pxpos(), v0.pypos(), v0.pzpos(), 0.);
        ROOT::Math::PxPyPzMVector negtrackVec(v0.pxneg(), v0.pyneg(), v0.pzneg(), 0.);
        outputTracks(outputCollision.lastIndex(), postrackVec.Pt(), postrackVec.Eta(), postrackVec.Phi(), aod::femtodreamparticle::ParticleType::kV0Child, cutContainerV0.at(1), 0., childIDs);
        const int rowOfPosTrack = outputTracks.lastIndex();
        int negtrackID = v0.negTrackId();
        int rowInPrimaryTrackTableNeg = -1;
        rowInPrimaryTrackTableNeg = getRowDaughters(negtrackID, tmpIDtrack);
        childIDs[0] = 0;
        childIDs[1] = rowInPrimaryTrackTableNeg;
        outputTracks(outputCollision.lastIndex(), negtrackVec.Pt(), negtrackVec.Eta(), negtrackVec.Phi(), aod::femtodreamparticle::ParticleType::kV0Child, cutContainerV0.at(2), 0., childIDs);
        const int rowOfNegTrack = outputTracks.lastIndex();
        int indexChildID[2] = {rowOfPosTrack, rowOfNegTrack};
        outputTracks(outputCollision.lastIndex(), v0.pt(), v0.eta(), v0.phi(), aod::femtodreamparticle::ParticleType::kV0, cutContainerV0.at(0), v0.v0cosPA(col.posX(), col.posY(), col.posZ()), indexChildID);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<femtoDreamProducerTask>(cfgc)};
  return workflow;
}
