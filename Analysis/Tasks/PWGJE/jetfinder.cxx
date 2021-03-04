// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// jet finder task
//
// Author: Jochen Klein, Nima Zardoshti

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"

#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetFinder.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

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

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision,
               soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>> const& tracks)
  {

    if (!collision.alias()[kINT7]) {
      return; //remove hard code
    }
    if (!collision.sel7()) {
      return; //remove hard code
    }
    jets.clear();
    inputParticles.clear();

    for (auto& track : tracks) {
      /*  auto energy = std::sqrt(track.p() * track.p() + mPion*mPion);
      inputParticles.emplace_back(track.px(), track.py(), track.pz(), energy);
      inputParticles.back().set_user_index(track.globalIndex());*/
      fillConstituents(track, inputParticles);
      inputParticles.back().set_user_index(track.globalIndex());
    }

    fastjet::ClusterSequenceArea clusterSeq(jetFinder.findJets(inputParticles, jets));

    for (const auto& jet : jets) {
      jetsTable(collision, jet.pt(), jet.eta(), jet.phi(),
                jet.E(), jet.m(), jet.area());
      hJetPt->Fill(jet.pt());
      hJetPhi->Fill(jet.phi());
      hJetEta->Fill(jet.eta());
      hJetN->Fill(jet.constituents().size());
      for (const auto& constituent : jet.constituents()) { //event or jetwise
        if (DoConstSub) {
          constituentsSubTable(jetsTable.lastIndex(), constituent.pt(), constituent.eta(), constituent.phi(),
                               constituent.E(), constituent.m());
        }
        constituentsTable(jetsTable.lastIndex(), constituent.user_index());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetFinderTask>(cfgc, "jet-finder")};
}
