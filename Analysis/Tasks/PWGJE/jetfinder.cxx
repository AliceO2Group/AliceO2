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

// jet finder task
//
// Author: Jochen Klein, Nima Zardoshti, Raymond Ehlers

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/EMCALClusters.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"

#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetFinder.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec dataType = {"particle-level-jet-finding",
                              VariantType::Bool,
                              false,
                              {"If true, perform particle level jet finding"}};
  workflowOptions.push_back(dataType);
  ConfigParamSpec jetType = {"jet-type",
                              VariantType::String,
                              "full",
                              {"Jet type (charged, neutral, or full)"}};
  workflowOptions.push_back(jetType);
}

#include "Framework/runDataProcessing.h"

enum class JetType_t {
  full = 0,
  charged = 1,
  neutral = 2,
};

template <typename JetTable, typename TrackConstituentTable, typename ClusterConstituentTable, typename ConstituentSubTable>
struct JetFinderTask {
  Produces<JetTable> jetsTable;
  Produces<TrackConstituentTable> trackConstituentsTable;
  Produces<ClusterConstituentTable> clusterConstituentsTable;
  Produces<ConstituentSubTable> constituentsSubTable;
  OutputObj<TH1F> hJetPt{"h_jet_pt"};
  OutputObj<TH1F> hJetPhi{"h_jet_phi"};
  OutputObj<TH1F> hJetEta{"h_jet_eta"};
  OutputObj<TH1F> hJetN{"h_jet_n"};

  Configurable<float> vertexZCut{"vertexZCut", 10.0f, "Accepted z-vertex range"};
  Configurable<float> trackPtCut{"trackPtCut", 0.1, "minimum constituent pT"};
  Configurable<float> trackEtaCut{"trackEtaCut", 0.9, "constituent eta cut"};
  Configurable<bool> DoRhoAreaSub{"DoRhoAreaSub", false, "do rho area subtraction"};
  Configurable<bool> DoConstSub{"DoConstSub", false, "do constituent subtraction"};
  Configurable<float> jetPtMin{"jetPtMin", 10.0, "minimum jet pT"};
  Configurable<float> jetR{"jetR", 0.4, "jet resolution"};
  Configurable<int> jetType{"jetType", 0, "Type of stored jets. 0 = full, 1 = charged, 2 = neutral"};

  Filter collisionFilter = nabs(aod::collision::posZ) < vertexZCut;
  Filter trackFilter = (nabs(aod::track::eta) < trackEtaCut) && (aod::track::isGlobalTrack == (uint8_t) true) && (aod::track::pt > trackPtCut);

  std::vector<fastjet::PseudoJet> jets;
  std::vector<fastjet::PseudoJet> inputParticles;
  JetFinder jetFinder; //should be a configurable but for now this cant be changed on hyperloop

  void init(InitContext const&)
  {
    hJetPt.setObject(new TH1F("h_jet_pt", "jet p_{T};p_{T} (GeV/#it{c})",
                              100, 0., 100.));
    hJetPhi.setObject(new TH1F("h_jet_phi", "jet #phi;#phi",
                               80, -1., 7.));
    hJetEta.setObject(new TH1F("h_jet_eta", "jet #eta;#eta",
                               70, -0.7, 0.7));
    hJetN.setObject(new TH1F("h_jet_n", "jet n;n constituents",
                             30, 0., 30.));
    if (DoRhoAreaSub) {
      jetFinder.setBkgSubMode(JetFinder::BkgSubMode::rhoAreaSub);
    }
    if (DoConstSub) {
      jetFinder.setBkgSubMode(JetFinder::BkgSubMode::constSub);
    }
    jetFinder.jetPtMin = jetPtMin;
    jetFinder.jetR = jetR;
  }

  template <typename T>
  bool processInit(T const& collision)
  {
    if (!collision.alias()[kINT7]) {
      return false; //remove hard code
    }
    if (!collision.sel7()) {
      return false; //remove hard code
    }

    jets.clear();
    inputParticles.clear();

    return true;
  }

  template <typename T>
  void processImpl(T const& collision)
  {
    fastjet::ClusterSequenceArea clusterSeq(jetFinder.findJets(inputParticles, jets));

    for (const auto& jet : jets) {
      jetsTable(collision, jet.pt(), jet.eta(), jet.phi(),
                jet.E(), jet.m(), jet.area(), -1);
      hJetPt->Fill(jet.pt());
      hJetPhi->Fill(jet.phi());
      hJetEta->Fill(jet.eta());
      hJetN->Fill(jet.constituents().size());
      for (const auto& constituent : jet.constituents()) { //event or jetwise
        if (DoConstSub) {
          // Since we're copying the consituents, we can combine the tracks and clusters together
          // We only have to keep the uncopied versions separated due to technical constraints.
          constituentsSubTable(jetsTable.lastIndex(), constituent.pt(), constituent.eta(), constituent.phi(),
                              constituent.E(), constituent.m(), constituent.user_index());
        }
        if (constituent.user_index() < 0) {
          // Cluster
          // -1 to account for the convention of negative indices for clusters.
          clusterConstituentsTable(jetsTable.lastIndex(), -1 * constituent.user_index());
        }
        else {
          // Tracks
          trackConstituentsTable(jetsTable.lastIndex(), constituent.user_index());
        }
      }
    }
  }

  void processParticleLevel(aod::McCollision const& collision, aod::McParticles const& particles)
  {
    // Setup
    // As of June 2021, I don't think enums are supported as configurables, so we have to handle the conversion here.
    // TODO: Double cast is to work around conversion failure.
    auto _jetType = static_cast<JetType_t>(static_cast<int>(jetType));

    // Initialziation and event selection
    // TODO: MC event selection?
    jets.clear();
    inputParticles.clear();

    // As of June 2021, how best to check for charged particles? It doesn't seem to be in
    // the McParticles table, so for now we select by PID.
    // charged hadron (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
    // We define full jets as taking everything, and neutral jets as not charged.
    std::vector <unsigned int> selectedPIDs{11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334};
    auto selectedBegin = selectedPIDs.begin();
    auto selectedEnd = selectedPIDs.end();
    for (auto& particle : particles) {
      if (_jetType != JetType_t::full) {
        bool foundChargedParticle = (std::find(selectedBegin, selectedEnd, particle.pdgCode()) == selectedEnd);
        if (_jetType == JetType_t::charged && foundChargedParticle == false) {
          continue;
        }
        if (_jetType == JetType_t::neutral && foundChargedParticle == true) {
          continue;
        }
      }

      inputParticles.emplace_back(
        fastjet::PseudoJet(
          particle.px(), particle.py(), particle.pz(), particle.e()
        )
      );
      inputParticles.back().set_user_index(particle.globalIndex());
    }

    processImpl(collision);
  }

  void processData(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision,
                   soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>> const& tracks,
                   aod::EMCALClusters const & clusters)
  {
    // Setup
    // As of June 2021, I don't think enums are supported as configurables, so we have to handle the conversion here.
    // FIXME: Double cast is to work around conversion failure.
    auto _jetType = static_cast<JetType_t>(static_cast<int>(jetType));

    // Initialziation and event selection
    bool accepted = processInit(collision);
    if (!accepted) {
      return;
    }

    if (_jetType == JetType_t::full || _jetType == JetType_t::charged) {
      for (auto& track : tracks) {
        fillConstituents(track, inputParticles);
        inputParticles.back().set_user_index(track.globalIndex());
      }
    }
    if (_jetType == JetType_t::full || _jetType == JetType_t::neutral) {
      for (auto& cluster : clusters) {
        // The right thing to do here would be to fully calculate the momentum correcting for the vertex position.
        // However, it's not clear that this functionality exists yet (21 June 2021)
        double pt = cluster.energy() / std::cosh(cluster.eta());
        inputParticles.emplace_back(
          fastjet::PseudoJet(
            pt * std::cos(cluster.phi()),
            pt * std::sin(cluster.phi()),
            pt * std::sinh(cluster.eta()),
            cluster.energy()
          )
        );
        // Clusters are denoted with negative indices.
        inputParticles.back().set_user_index(-1 * cluster.globalIndex());
      }
    }

    processImpl(collision);
  }
};

using StandardJetFinder = JetFinderTask<o2::aod::Jets, o2::aod::JetTrackConstituents, o2::aod::JetClusterConstituents, o2::aod::JetConstituentsSub>;

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto particleLevelJetFinding = cfgc.options().get<bool>("particle-level-jet-finding");
  if (particleLevelJetFinding) {
      return WorkflowSpec{
        adaptAnalysisTask<StandardJetFinder>(cfgc, Processes{&StandardJetFinder::processParticleLevel}, TaskName{"jet-finder-MC"})};
  }
  return WorkflowSpec{
    adaptAnalysisTask<StandardJetFinder>(cfgc, Processes{&StandardJetFinder::processData}, TaskName{"jet-finder-data"})};
}

