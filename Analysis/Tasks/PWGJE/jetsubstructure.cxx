// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// jet analysis tasks (subscribing to jet finder task)
//
// Author: Nima Zardoshti
//

#include "TH1F.h"
#include "TTree.h"

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoA.h"

#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetFinder.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
namespace jetsubstructure
{
DECLARE_SOA_COLUMN(Zg, zg, float);
DECLARE_SOA_COLUMN(Rg, rg, float);
DECLARE_SOA_COLUMN(Nsd, nsd, float);
} // namespace jetsubstructure
DECLARE_SOA_TABLE(JetSubtructure, "AOD", "JETSUBSTRUCTURE", jetsubstructure::Zg, jetsubstructure::Rg, jetsubstructure::Nsd);
} // namespace o2::aod

struct JetSubstructure {
  Produces<aod::JetSubtructure> jetSubstructure;
  OutputObj<TH1F> hZg{"h_jet_zg"};
  OutputObj<TH1F> hRg{"h_jet_rg"};
  OutputObj<TH1F> hNsd{"h_jet_nsd"};

  Configurable<float> f_jetPtMin{"f_jetPtMin", 0.0, "minimum jet pT cut"};
  Configurable<float> f_zCut{"f_zCut", 0.1, "soft drop z cut"};
  Configurable<float> f_beta{"f_beta", 0.0, "soft drop beta"};
  Configurable<float> f_jetR{"f_jetR", 0.4, "jer resolution parameter"}; //possible to get configurable from another task? jetR
  Configurable<bool> b_DoConstSub{"b_DoConstSub", false, "do constituent subtraction"};

  std::vector<fastjet::PseudoJet> jetConstituents;
  std::vector<fastjet::PseudoJet> jetReclustered;
  JetFinder jetReclusterer;

  void init(InitContext const&)
  {
    hZg.setObject(new TH1F("h_jet_zg", "zg ;zg",
                           10, 0.0, 0.5));
    hRg.setObject(new TH1F("h_jet_rg", "rg ;rg",
                           10, 0.0, 0.5));
    hNsd.setObject(new TH1F("h_jet_nsd", "nsd ;nsd",
                            7, -0.5, 6.5));
    jetReclusterer.isReclustering = true;
    jetReclusterer.algorithm = fastjet::JetAlgorithm::cambridge_algorithm;
    jetReclusterer.jetR = f_jetR * 2.5;
  }

  //Filter jetCuts = aod::jet::pt > f_jetPtMin; //how does this work?

  void process(aod::Jet const& jet,
               aod::Tracks const& tracks,
               aod::JetConstituents const& constituents,
               aod::JetConstituentsSub const& constituentsSub)
  {
    jetConstituents.clear();
    jetReclustered.clear();
    if (b_DoConstSub) {
      for (const auto constituent : constituentsSub) {
        fillConstituents(constituent, jetConstituents);
      }
    } else {
      for (const auto constituentIndex : constituents) {
        auto constituent = constituentIndex.track();
        fillConstituents(constituent, jetConstituents);
      }
    }
    fastjet::ClusterSequenceArea clusterSeq(jetReclusterer.findJets(jetConstituents, jetReclustered));
    jetReclustered = sorted_by_pt(jetReclustered);
    fastjet::PseudoJet daughterSubJet = jetReclustered[0];
    fastjet::PseudoJet parentSubJet1;
    fastjet::PseudoJet parentSubJet2;
    bool softDropped = false;
    int nsd = 0.0;
    auto zg = -1.0;
    auto rg = -1.0;
    while (daughterSubJet.has_parents(parentSubJet1, parentSubJet2)) {
      if (parentSubJet1.perp() < parentSubJet2.perp()) {
        std::swap(parentSubJet1, parentSubJet2);
      }
      auto z = parentSubJet2.perp() / (parentSubJet1.perp() + parentSubJet2.perp());
      auto r = parentSubJet1.delta_R(parentSubJet2);
      if (z >= f_zCut * TMath::Power(r / f_jetR, f_beta)) {
        if (!softDropped) {
          zg = z;
          rg = r;
          hZg->Fill(zg);
          hRg->Fill(rg);
          softDropped = true;
        }
        nsd++;
      }
      daughterSubJet = parentSubJet1;
    }
    hNsd->Fill(nsd);
    jetSubstructure(zg, rg, nsd);
  }
};
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetSubstructure>("jet-substructure")};
}
