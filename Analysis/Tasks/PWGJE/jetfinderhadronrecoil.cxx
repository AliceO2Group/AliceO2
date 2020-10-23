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

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"

#include "Analysis/Jet.h"
#include "Analysis/JetFinder.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetFinderHadronRecoilTask {
  OutputObj<TH1F> hJetPt{"h_jet_pt"};
  OutputObj<TH1F> hHadronPt{"h_hadron_pt"};
  OutputObj<TH1F> hJetHadronDeltaPhi{"h_jet_hadron_deltaphi"};

  Configurable<float> f_trackTTMin{"f_trackTTMin", 8.0, "TT hadron min pT"};
  Configurable<float> f_trackTTMax{"f_trackTTMax", 9.0, "TT hadron max pT"};
  Configurable<float> f_recoilWindow{"f_recoilWindow", 0.6, "jet finding phi window reecoilling from hadron"};

  Filter trackCuts = aod::track::pt >= 0.15f && aod::track::eta > -0.9f && aod::track::eta < 0.9f;
  int collisionSplit = 0; //can we partition the collisions?
  //can we also directly filter the collision based on the max track::pt?

  std::vector<float> trackTTPt;
  std::vector<fastjet::PseudoJet> jets;
  std::vector<fastjet::PseudoJet> inputParticles;
  JetFinder jetFinder;

  template <typename T>
  T relativePhi(T phi1, T phi2)
  {
    if (phi1 < -TMath::Pi()) {
      phi1 += (2 * TMath::Pi());
    } else if (phi1 > TMath::Pi()) {
      phi1 -= (2 * TMath::Pi());
    }
    if (phi2 < -TMath::Pi()) {
      phi2 += (2 * TMath::Pi());
    } else if (phi2 > TMath::Pi()) {
      phi2 -= (2 * TMath::Pi());
    }
    T deltaPhi = phi2 - phi1;
    if (deltaPhi < -TMath::Pi()) {
      deltaPhi += (2 * TMath::Pi());
    } else if (deltaPhi > TMath::Pi()) {
      deltaPhi -= (2 * TMath::Pi());
    }
    return deltaPhi;
  }

  void init(InitContext const&)
  {
    hJetPt.setObject(new TH1F("h_jet_pt", "jet p_{T};jet p_{T} (GeV/#it{c})",
                              100, 0., 100.));
    hHadronPt.setObject(new TH1F("h_hadron_pt", "hadron p_{T};hadron p_{T} (GeV/#it{c})",
                                 120, 0., 60.));
    hJetHadronDeltaPhi.setObject(new TH1F("h_jet_hadron_deltaphi", "jet #eta;#eta",
                                          40, 0.0, 4.));
  }

  void process(aod::Collision const& collision,
               soa::Filtered<aod::Tracks> const& tracks)
  {

    jets.clear();
    inputParticles.clear();
    auto trackTTPhi = 0.0;
    auto trackTTPt = 0.0;
    auto comparisonPt = 0.0;
    bool isTT = false;
    for (auto& track : tracks) {
      if (track.pt() >= f_trackTTMin && track.pt() < f_trackTTMax) { //can this also go into a partition?
        isTT = true;
        if (track.pt() >= comparisonPt) { //currently take highest pT but later to randomise
          comparisonPt = track.pt();
          trackTTPt = track.pt();
          trackTTPhi = track.phi();
        }
      }
      auto energy = std::sqrt(track.p() * track.p() + JetFinder::mPion * JetFinder::mPion);
      inputParticles.emplace_back(track.px(), track.py(), track.pz(), energy);
      inputParticles.back().set_user_index(track.globalIndex());
    }
    if (!isTT) {
      return;
    }
    hHadronPt->Fill(trackTTPt);

    // you can set phi selector here for jets
    fastjet::ClusterSequenceArea clusterSeq(jetFinder.findJets(inputParticles, jets));

    for (const auto& jet : jets) {
      auto deltaPhi = TMath::Abs(relativePhi(jet.phi(), trackTTPhi));
      if (deltaPhi >= (TMath::Pi() - f_recoilWindow)) {
        hJetPt->Fill(jet.pt());
      }
      if (deltaPhi >= TMath::Pi() / 2.0 && deltaPhi <= TMath::Pi()) {
        hJetHadronDeltaPhi->Fill(deltaPhi);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetFinderHadronRecoilTask>("jet-finder-hadron-recoil")};
}
