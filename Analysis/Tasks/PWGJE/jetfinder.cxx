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
  ConfigParamSpec jetData = {
    "jet-input-data",
    VariantType::String,
    "",
    {"Jet data type. Options include Data, MCParticleLevel, MCDetectorLevel, and HybridIntermediate."},
  };
  workflowOptions.push_back(jetData);
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
  OutputObj<TH2F> hJetPt{"h_jet_pt"};
  OutputObj<TH2F> hJetPhi{"h_jet_phi"};
  OutputObj<TH2F> hJetEta{"h_jet_eta"};
  OutputObj<TH2F> hJetN{"h_jet_n"};

  Configurable<float> vertexZCut{"vertexZCut", 10.0f, "Accepted z-vertex range"};
  Configurable<float> trackPtCut{"trackPtCut", 0.1, "minimum constituent pT"};
  Configurable<float> trackEtaCut{"trackEtaCut", 0.9, "constituent eta cut"};
  Configurable<bool> DoRhoAreaSub{"DoRhoAreaSub", false, "do rho area subtraction"};
  Configurable<bool> DoConstSub{"DoConstSub", false, "do constituent subtraction"};
  Configurable<float> jetPtMin{"jetPtMin", 10.0, "minimum jet pT"};
  Configurable<std::vector<double>> jetR{"jetR", {0.4}, "jet resolution parameters"};
  Configurable<int> jetType{"jetType", 0, "Type of stored jets. 0 = full, 1 = charged, 2 = neutral"};

  Filter collisionFilter = nabs(aod::collision::posZ) < vertexZCut;
  Filter trackFilter = (nabs(aod::track::eta) < trackEtaCut) && (aod::track::isGlobalTrack == (uint8_t) true) && (aod::track::pt > trackPtCut);

  std::vector<fastjet::PseudoJet> jets;
  std::vector<fastjet::PseudoJet> inputParticles;
  JetFinder jetFinder; //should be a configurable but for now this cant be changed on hyperloop

  void init(InitContext const&)
  {
    hJetPt.setObject(new TH2F("h_jet_pt", "jet p_{T};p_{T} (GeV/#it{c})",
                              100, 0., 100., 10, 0.05, 1.05));
    hJetPhi.setObject(new TH2F("h_jet_phi", "jet #phi;#phi",
                               80, -1., 7., 10, 0.05, 1.05));
    hJetEta.setObject(new TH2F("h_jet_eta", "jet #eta;#eta",
                               70, -0.7, 0.7, 10, 0.05, 1.05));
    hJetN.setObject(new TH2F("h_jet_n", "jet n;n constituents",
                             30, 0., 30., 10, 0.05, 1.05));
    if (DoRhoAreaSub) {
      jetFinder.setBkgSubMode(JetFinder::BkgSubMode::rhoAreaSub);
    }
    if (DoConstSub) {
      jetFinder.setBkgSubMode(JetFinder::BkgSubMode::constSub);
    }
    jetFinder.jetPtMin = jetPtMin;
    // Start with the first jet R
    //jetFinder.jetR = jetR[0];
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
    auto jetRValues = static_cast<std::vector<double>>(jetR);
    for (auto R : jetRValues) {
      // Update jet finder R and find jets
      jetFinder.jetR = R;
      fastjet::ClusterSequenceArea clusterSeq(jetFinder.findJets(inputParticles, jets));

      for (const auto& jet : jets) {
        jetsTable(collision, jet.pt(), jet.eta(), jet.phi(),
                  jet.E(), jet.m(), jet.area(), std::round(R * 100));
        hJetPt->Fill(jet.pt(), R);
        hJetPhi->Fill(jet.phi(), R);
        hJetEta->Fill(jet.eta(), R);
        hJetN->Fill(jet.constituents().size(), R);
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
          } else {
            // Tracks
            trackConstituentsTable(jetsTable.lastIndex(), constituent.user_index());
          }
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
    std::vector<unsigned int> selectedPIDs{11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334};
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
          particle.px(), particle.py(), particle.pz(), particle.e()));
      inputParticles.back().set_user_index(particle.globalIndex());
    }

    processImpl(collision);
  }

  void processData(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision,
                   soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>> const& tracks,
                   aod::EMCALClusters const& clusters)
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
            cluster.energy()));
        // Clusters are denoted with negative indices.
        inputParticles.back().set_user_index(-1 * cluster.globalIndex());
      }
    }

    processImpl(collision);
  }
};

using JetFinderData = JetFinderTask<o2::aod::Jets, o2::aod::JetTrackConstituents, o2::aod::JetClusterConstituents, o2::aod::JetConstituentsSub>;
using JetFinderMCParticleLevel = JetFinderTask<o2::aod::MCParticleLevelJets, o2::aod::MCParticleLevelJetTrackConstituents, o2::aod::MCParticleLevelJetClusterConstituents, o2::aod::MCParticleLevelJetConstituentsSub>;
using JetFinderMCDetectorLevel = JetFinderTask<o2::aod::MCDetectorLevelJets, o2::aod::MCDetectorLevelJetTrackConstituents, o2::aod::MCDetectorLevelJetClusterConstituents, o2::aod::MCDetectorLevelJetConstituentsSub>;
using JetFinderHybridIntermediate = JetFinderTask<o2::aod::HybridIntermediateJets, o2::aod::HybridIntermediateJetTrackConstituents, o2::aod::HybridIntermediateJetClusterConstituents, o2::aod::HybridIntermediateJetConstituentsSub>;

enum class JetInputData_t {
  Data,
  MCParticleLevel,
  MCDetectorLevel,
  HybridIntermediate
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto jetInputData = cfgc.options().get<std::string>("jet-input-data");
  const std::map<std::string, JetInputData_t> jetInputDataTypes = {
    {"Data", JetInputData_t::Data},
    {"MCParticleLevel", JetInputData_t::MCParticleLevel},
    {"MCDetectorLevel", JetInputData_t::MCDetectorLevel},
    {"HybridIntermediate", JetInputData_t::HybridIntermediate},
    {"", JetInputData_t::Data}, // Default to data
  };
  auto jetData = jetInputDataTypes.at(jetInputData);
  switch (jetData) {
    case JetInputData_t::MCParticleLevel:
      return WorkflowSpec{
        adaptAnalysisTask<JetFinderMCParticleLevel>(cfgc, Processes{&JetFinderMCParticleLevel::processParticleLevel}, TaskName{"jet-finder-MC"})};
      break;
    case JetInputData_t::MCDetectorLevel:
      return WorkflowSpec{
        adaptAnalysisTask<JetFinderMCDetectorLevel>(cfgc, Processes{&JetFinderMCDetectorLevel::processData}, TaskName{"jet-finder-MC-detector-level"})};
      break;
    case JetInputData_t::HybridIntermediate:
      return WorkflowSpec{
        adaptAnalysisTask<JetFinderHybridIntermediate>(cfgc, Processes{&JetFinderHybridIntermediate::processData}, TaskName{"jet-finder-hybrid-intermedaite"})};
      break;
    case JetInputData_t::Data: // intentionally fall through to the default which is outside of the switch.
    default:
      break;
  }
  // Default to data
  return WorkflowSpec{
    adaptAnalysisTask<JetFinderHybridIntermediate>(cfgc, Processes{&JetFinderHybridIntermediate::processData}, TaskName{"jet-finder-hybrid-intermedaite"})};
}
