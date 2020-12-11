// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskDPlus.cxx
/// \brief D± analysis task
/// \note Extended from taskD0
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
//using namespace o2::framework::expressions;

/// D± analysis task
struct TaskDPlus {
  HistogramRegistry registry{
    "registry",
    {
      {"hMass", "3-prong candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{350, 1.7, 2.05}}}},
      {"hPt", "3-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
      {"hEta", "3-prong candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
      {"hCt", "3-prong candidates;proper lifetime (D^{#pm}) * #it{c} (cm);entries", {HistType::kTH1F, {{120, -20., 100.}}}},
      {"hDecayLength", "3-prong candidates;decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
      {"hDecayLengthXY", "3-prong candidates;decay length xy (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
      {"hNormalisedDecayLengthXY", "3-prong candidates;norm. decay length xy;entries", {HistType::kTH1F, {{80, 0., 80.}}}},
      {"hCPA", "3-prong candidates;cos. pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
      {"hCPAxy", "3-prong candidates;cos. pointing angle xy;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
      {"hImpactParameterXY", "3-prong candidates;impact parameter xy (cm);entries", {HistType::kTH1F, {{200, -1., 1.}}}},
      {"hMaxNormalisedDeltaIP", "3-prong candidates;norm. IP;entries", {HistType::kTH1F, {{200, -20., 20.}}}},
      {"hImpactParameterProngSqSum", "3-prong candidates;squared sum of prong imp. par. (cm^{2});entries", {HistType::kTH1F, {{100, 0., 1.}}}},
      {"hDecayLengthError", "3-prong candidates;decay length error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
      {"hDecayLengthXYError", "3-prong candidates;decay length xy error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
      {"hImpactParameterError", "3-prong candidates;impact parameter error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
      {"hPtProng0", "3-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
      {"hPtProng1", "3-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
      {"hPtProng2", "3-prong candidates;prong 2 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
      {"hd0Prong0", "3-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
      {"hd0Prong1", "3-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
      {"hd0Prong2", "3-prong candidates;prong 2 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}}
      //{"hSelectionStatus", "3-prong candidates;selection status;entries",  {HistType::kTH1F, {{5, -0.5, 4.5}}}}
    }};

  Configurable<int> d_selectionFlagDPlus{"d_selectionFlagDPlus", 1, "Selection Flag for DPlus"};

  //Filter filterSelectCandidates = (aod::hf_selcandidate::isSelDPlus >= d_selectionFlagDPlus);

  //void process(soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelDPlusCandidate>> const& candidates)
  void process(aod::HfCandProng3 const& candidates)
  {
    for (auto& candidate : candidates) {
      registry.get<TH1>(HIST("hMass"))->Fill(InvMassDPlus(candidate));
      registry.get<TH1>(HIST("hPt"))->Fill(candidate.pt());
      registry.get<TH1>(HIST("hEta"))->Fill(candidate.eta());
      registry.get<TH1>(HIST("hCt"))->Fill(CtDPlus(candidate));
      registry.get<TH1>(HIST("hDecayLength"))->Fill(candidate.decayLength());
      registry.get<TH1>(HIST("hDecayLengthXY"))->Fill(candidate.decayLengthXY());
      registry.get<TH1>(HIST("hNormalisedDecayLengthXY"))->Fill(candidate.decayLengthXYNormalised());
      registry.get<TH1>(HIST("hCPA"))->Fill(candidate.cpa());
      registry.get<TH1>(HIST("hCPAxy"))->Fill(candidate.cpaXY());
      registry.get<TH1>(HIST("hImpactParameterXY"))->Fill(candidate.impactParameterXY());
      registry.get<TH1>(HIST("hMaxNormalisedDeltaIP"))->Fill(candidate.maxNormalisedDeltaIP());
      registry.get<TH1>(HIST("hImpactParameterProngSqSum"))->Fill(candidate.impactParameterProngSqSum());
      registry.get<TH1>(HIST("hDecayLengthError"))->Fill(candidate.errorDecayLength());
      registry.get<TH1>(HIST("hDecayLengthXYError"))->Fill(candidate.errorDecayLengthXY());
      registry.get<TH1>(HIST("hImpactParameterError"))->Fill(candidate.errorImpactParameter0());
      registry.get<TH1>(HIST("hImpactParameterError"))->Fill(candidate.errorImpactParameter1());
      registry.get<TH1>(HIST("hImpactParameterError"))->Fill(candidate.errorImpactParameter2());
      registry.get<TH1>(HIST("hPtProng0"))->Fill(candidate.ptProng0());
      registry.get<TH1>(HIST("hPtProng1"))->Fill(candidate.ptProng1());
      registry.get<TH1>(HIST("hPtProng2"))->Fill(candidate.ptProng2());
      registry.get<TH1>(HIST("hd0Prong0"))->Fill(candidate.impactParameter0());
      registry.get<TH1>(HIST("hd0Prong1"))->Fill(candidate.impactParameter1());
      registry.get<TH1>(HIST("hd0Prong2"))->Fill(candidate.impactParameter2());
      //registry.get<TH1>(HIST("hSelectionStatus"))->Fill(candidate.isSelDPlus());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TaskDPlus>("hf-task-dplus")};
}
