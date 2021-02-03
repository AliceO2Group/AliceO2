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
#include "DataModel/JEDerived.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetSpectraAnalyser {

  HistogramRegistry registry{
    "registry1",
    {{"hJetPt", "Jet pT;Jet #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 100.}}}},
     {"hNJetConstituents", "Number of constituents;N;entries", {HistType::kTH1F, {{100, -0.5, 99.5}}}},
     {"hConstituentPt", "Constituent pT; Constituent #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 100.}}}}}};

  //Filter jetCuts = aod::jet::pt > f_jetPtMin;

  void process(aod::JEJet const& jet,
               aod::JEConstituents const& constituents)
  {
    registry.fill(HIST("hJetPt"), jet.pt());
    registry.fill(HIST("hNJetConstituents"), constituents.size());
    for (const auto constituent : constituents) {
      registry.fill(HIST("hConstituentPt"), constituent.pt());
    }
  }
};
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<JetSpectraAnalyser>("jetspectra-task-skim-analyser")};
}