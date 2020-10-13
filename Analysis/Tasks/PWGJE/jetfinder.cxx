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

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"

#include "Analysis/Jet.h"
#include "Analysis/JetFinder.h"

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

  Configurable<bool> b_DoRhoAreaSub{"b_DoRhoAreaSub", false, "do rho area subtraction"};
  Configurable<bool> b_DoConstSub{"b_DoConstSub", false, "do constituent subtraction"};

  Filter trackCuts = aod::track::pt >= 0.15f && aod::track::eta >= -0.9f && aod::track::eta <= 0.9f;

  std::vector<fastjet::PseudoJet> jets;
  std::vector<fastjet::PseudoJet> inputParticles;
  JetFinder jetFinder;

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
    if (b_DoRhoAreaSub)
      jetFinder.setBkgSubMode(JetFinder::BkgSubMode::rhoAreaSub);
    if (b_DoConstSub)
      jetFinder.setBkgSubMode(JetFinder::BkgSubMode::constSub);
  }

  void process(aod::Collision const& collision,
               soa::Filtered<aod::Tracks> const& tracks)
  {

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
        if (b_DoConstSub)
          constituentsSubTable(jetsTable.lastIndex(), constituent.pt(), constituent.eta(), constituent.phi(),
                               constituent.E(), constituent.m());
        constituentsTable(jetsTable.lastIndex(), constituent.user_index());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetFinderTask>("jet-finder")};
}
