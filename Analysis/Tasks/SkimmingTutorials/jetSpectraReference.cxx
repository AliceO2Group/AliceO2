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
#include "Framework/HistogramRegistry.h"

#include "AnalysisDataModel/Jet.h"
#include "AnalysisCore/JetFinder.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetSpectraReference {

  HistogramRegistry registry{
    "registry",
    {{"hJetPt", "Jet pT;Jet #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 100.}}}},
     {"hNJetConstituents", "Number of constituents;N;entries", {HistType::kTH1F, {{100, -0.5, 99.5}}}},
     {"hConstituentPt", "Constituent pT; Constituent #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 100.}}}}}};

  //Configurable<float> f_jetPtMin{"f_jetPtMin", 0.0, "minimum jet pT cut"};
  Configurable<bool> b_DoConstSub{"b_DoConstSub", false, "do constituent subtraction"};

  //Filter jetCuts = aod::jet::pt > f_jetPtMin; //how does this work?

  void process(aod::Jet const& jet,
               aod::Tracks const& tracks,
               aod::JetConstituents const& constituents,
               aod::JetConstituentsSub const& constituentsSub)
  {
    registry.fill(HIST("hJetPt"), jet.pt());
    if (b_DoConstSub) {
      registry.fill(HIST("hNJetConstituents"), constituentsSub.size());
      for (const auto constituent : constituentsSub) {
        registry.fill(HIST("hConstituentPt"), constituent.pt());
      }
    } else {
      registry.fill(HIST("hNJetConstituents"), constituents.size());
      for (const auto constituentIndex : constituents) {
        auto constituent = constituentIndex.track();
        registry.fill(HIST("hConstituentPt"), constituent.pt());
      }
    }
  }
};
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetSpectraReference>(cfgc, "jetspectra-task-skim-reference")};
}
