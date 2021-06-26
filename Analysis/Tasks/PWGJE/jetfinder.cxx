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
  ConfigParamSpec centspec = {"jet-type",
                              VariantType::String,
                              "charged",
                              {"Jet type (charged or full)"}};
  workflowOptions.push_back(centspec);
}

#include "Framework/runDataProcessing.h"

struct JetFinderTask {
  Produces<o2::aod::Jets> jetsTable;
  Produces<o2::aod::JetConstituents> constituentsTable;
  Produces<o2::aod::JetConstituentsSub> constituentsSubTable;
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

  void processImpl(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision)
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
        if (constituent.user_index() < 0) {
          // Cluster
          // TODO: Implement cluster constituents...
        }
        else {
          if (DoConstSub) {
            constituentsSubTable(jetsTable.lastIndex(), constituent.pt(), constituent.eta(), constituent.phi(),
                                constituent.E(), constituent.m());
          }
          constituentsTable(jetsTable.lastIndex(), constituent.user_index());
        }
      }
    }
  }

  bool processInit(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision)
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

  void processChargedJets(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision,
               soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>> const& tracks)
  {
    bool accepted = processInit(collision);
    if (!accepted) {
      return;
    }

    for (auto& track : tracks) {
      fillConstituents(track, inputParticles);
      inputParticles.back().set_user_index(track.globalIndex());
    }

    processImpl(collision);
  }

  void processFullJets(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision,
               soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>> const& tracks,
               aod::EMCALClusters const & clusters)
  {
    bool accepted = processInit(collision);
    if (!accepted) {
      return;
    }

    for (auto& track : tracks) {
      fillConstituents(track, inputParticles);
      inputParticles.back().set_user_index(track.globalIndex());
    }
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

    processImpl(collision);
  }

};

enum class JetType_t {
  charged,
  full,
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  // TODO: Is there a better way to do this?
  const std::map<std::string, JetType_t> jetTypeMap = {
    {"charged", JetType_t::charged},
    {"full", JetType_t::full},
  };
  auto jetTypeStr = cfgc.options().get<std::string>("jet-type");
  auto jetType = jetTypeMap.at(jetTypeStr);
  switch (jetType) {
    case JetType_t::full:
      return WorkflowSpec{
        adaptAnalysisTask<JetFinderTask>(cfgc, Processes{&JetFinderTask::processFullJets}, TaskName{"jet-finder-full"})};
      break;
    case JetType_t::charged:
      return WorkflowSpec{
        adaptAnalysisTask<JetFinderTask>(cfgc, Processes{&JetFinderTask::processChargedJets}, TaskName{"jet-finder-charged"})};
      break;
    default:
      break;
  }
  LOG(FATAL) << "Jet type unsupported: " << static_cast<int>(jetType) << ", input string: " << jetTypeStr << "\n";
}

