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
// Author: Nima Zardoshti

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"

#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"

#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetFinder.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetFinderHFTask {
  Produces<o2::aod::Jets> jetsTable;
  Produces<o2::aod::JetConstituents> constituents;
  OutputObj<TH1F> hJetPt{"h_jet_pt"};
  OutputObj<TH1F> hD0Pt{"h_D0_pt"};

  std::vector<fastjet::PseudoJet> jets;
  std::vector<fastjet::PseudoJet> inputParticles;
  JetFinder jetFinder;

  void init(InitContext const&)
  {
    hJetPt.setObject(new TH1F("h_jet_pt", "jet p_{T};p_{T} (GeV/#it{c})",
                              100, 0., 100.));
    hD0Pt.setObject(new TH1F("h_D0_pt", "jet p_{T,D};p_{T,D} (GeV/#it{c})",
                             60, 0., 60.));
  }

  Configurable<int> d_selectionFlagD0ToPiK{"d_selectionFlagD0ToPiK", 1, "Selection Flag for D0ToPiK"};
  Configurable<int> d_selectionFlagD0ToPiKbar{"d_selectionFlagD0ToPiKbar", 1, "Selection Flag for D0ToPiKbar"};

  //need enum as configurable
  enum pdgCode { pdgD0 = 421 };

  Filter trackCuts = (aod::track::pt > 0.15f && aod::track::eta > -0.9f && aod::track::eta < 0.9f);
  Filter seltrack = (aod::hf_selcandidate_d0::isSelD0ToPiK >= d_selectionFlagD0ToPiK || aod::hf_selcandidate_d0::isSelD0ToPiKbar >= d_selectionFlagD0ToPiKbar);

  void process(aod::Collision const& collision,
               soa::Filtered<aod::Tracks> const& tracks,
               soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0ToPiKCandidate>> const& candidates)
  {
    std::cout << "Per Event" << std::endl;
    // TODO: retrieve pion mass from somewhere
    bool isHFJet;

    //this loop should be made more efficient
    for (auto& candidate : candidates) {
      jets.clear();
      inputParticles.clear();
      for (auto& track : tracks) {
        auto energy = std::sqrt(track.p() * track.p() + JetFinder::mPion * JetFinder::mPion);
        if (candidate.index0().globalIndex() == track.globalIndex() || candidate.index1().globalIndex() == track.globalIndex()) { //is it global index?
          continue;
        }
        inputParticles.emplace_back(track.px(), track.py(), track.pz(), energy);
        inputParticles.back().set_user_index(track.globalIndex());
      }
      inputParticles.emplace_back(candidate.px(), candidate.py(), candidate.pz(), candidate.e(RecoDecay::getMassPDG(pdgD0)));
      inputParticles.back().set_user_index(1);

      fastjet::ClusterSequenceArea clusterSeq(jetFinder.findJets(inputParticles, jets));

      for (const auto& jet : jets) {
        isHFJet = false;
        for (const auto& constituent : jet.constituents()) {
          if (constituent.user_index() == 1 && (candidate.isSelD0ToPiK() == 1 || candidate.isSelD0ToPiKbar() == 1)) {
            isHFJet = true;
            break;
          }
        }
        if (isHFJet) {
          jetsTable(collision, jet.eta(), jet.phi(), jet.pt(),
                    jet.area(), jet.E(), jet.m());
          for (const auto& constituent : jet.constituents()) {
            constituents(jetsTable.lastIndex(), constituent.user_index());
          }
          hJetPt->Fill(jet.pt());
          std::cout << "Filling" << std::endl;
          hD0Pt->Fill(candidate.pt());
          break;
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetFinderHFTask>("jet-finder-hf")};
}
