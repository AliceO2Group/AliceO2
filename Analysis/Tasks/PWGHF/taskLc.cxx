// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskLc.cxx
/// \brief Lc± analysis task
/// \note Extended from taskD0
///
/// \author Luigi Dello Stritto <luigi.dello.stritto@cern.ch>, University and INFN SALERNO
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// Lc± analysis task
struct TaskLc {
  HistogramRegistry registry{
    "registry",
    {{"hmass", "3-prong candidates;inv. mass (p K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hptcand", "3-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0", "3-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1", "3-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong2", "3-prong candidates;prong 2 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hdeclength", "3-prong candidates;decay length (cm);entries", {HistType::kTH1F, {{400, -2., 2.}}}},
     {"hd0Prong0", "3-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong1", "3-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong2", "3-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hCt", "3-prong candidates;proper lifetime (#Lambda_{c}) * #it{c} (cm);entries", {HistType::kTH1F, {{120, -20., 100.}}}},
     {"hCPA", "3-prong candidates;cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEta", "3-prong candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hselectionstatus", "3-prong candidates;selection status;entries", {HistType::kTH1F, {{5, -0.5, 4.5}}}},
     {"hImpParErr", "3-prong candidates;impact parameter error (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hDecLenErr", "3-prong candidates;decay length error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
     {"hdca2", "3-prong candidates;prong DCA to sec. vertex (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}}}};

  Configurable<int> d_selectionFlagLc{"d_selectionFlagLc", 1, "Selection Flag for Lc"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_lc::isSelLcpKpi >= d_selectionFlagLc || aod::hf_selcandidate_lc::isSelLcpiKp >= d_selectionFlagLc);

  //void process(aod::HfCandProng3 const& candidates)
  void process(soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelLcCandidate>> const& candidates)
  {
    for (auto& candidate : candidates) {
      /* if (candidate.pt()>5){
	 continue;}*/
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        //Printf("Candidate: eta rejection: %g", candidate.eta());
        continue;
      }
      if (candidate.isSelLcpKpi() >= d_selectionFlagLc) {
        registry.get<TH1>("hmass")->Fill(InvMassLcpKpi(candidate));
      }
      if (candidate.isSelLcpiKp() >= d_selectionFlagLc) {
        registry.get<TH1>("hmass")->Fill(InvMassLcpiKp(candidate));
      }
      registry.get<TH1>("hptcand")->Fill(candidate.pt());
      registry.get<TH1>("hptprong0")->Fill(candidate.ptProng0());
      registry.get<TH1>("hptprong1")->Fill(candidate.ptProng1());
      registry.get<TH1>("hptprong2")->Fill(candidate.ptProng2());
      registry.get<TH1>("hdeclength")->Fill(candidate.decayLength());
      registry.get<TH1>("hd0Prong0")->Fill(candidate.impactParameter0());
      registry.get<TH1>("hd0Prong1")->Fill(candidate.impactParameter1());
      registry.get<TH1>("hd0Prong2")->Fill(candidate.impactParameter2());
      registry.get<TH1>("hCt")->Fill(CtLc(candidate));
      registry.get<TH1>("hCPA")->Fill(candidate.cpa());
      registry.get<TH1>("hEta")->Fill(candidate.eta());
      registry.get<TH1>("hselectionstatus")->Fill(candidate.isSelLcpKpi());
      registry.get<TH1>("hselectionstatus")->Fill(candidate.isSelLcpiKp());
      registry.get<TH1>("hImpParErr")->Fill(candidate.errorImpactParameter0());
      registry.get<TH1>("hImpParErr")->Fill(candidate.errorImpactParameter1());
      registry.get<TH1>("hImpParErr")->Fill(candidate.errorImpactParameter2());
      registry.get<TH1>("hDecLenErr")->Fill(candidate.errorDecayLength());
      registry.get<TH1>("hDecLenErr")->Fill(candidate.chi2PCA());
    }
  }
};

/// Fills MC histograms.
struct TaskLcMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "3-prong candidates (rec. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "3-prong candidates (rec. unmatched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "3-prong candidates (gen. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hCPARecSig", "3-prong candidates (rec. matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "3-prong candidates (rec. unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "3-prong candidates (rec. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "3-prong candidates (rec. unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "3-prong candidates (gen. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}}}};

  Configurable<int> d_selectionFlagLc{"d_selectionFlagLc", 1, "Selection Flag for Lc"};
  Configurable<int> d_selectionFlagLcbar{"d_selectionFlagLcbar", 1, "Selection Flag for Lcbar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_lc::isSelLcpKpi >= d_selectionFlagLc || aod::hf_selcandidate_lc::isSelLcpiKp >= d_selectionFlagLc);

  void process(soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelLcCandidate, aod::HfCandProng3MCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandProng3MCGen> const& particlesMC)
  {
    // MC rec.
    //Printf("MC Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        //Printf("MC Rec.: eta rejection: %g", candidate.eta());
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == LcToPKPi) {
        registry.get<TH1>("hPtRecSig")->Fill(candidate.pt());
        registry.get<TH1>("hCPARecSig")->Fill(candidate.cpa());
        registry.get<TH1>("hEtaRecSig")->Fill(candidate.eta());
      } else {
        registry.get<TH1>("hPtRecBg")->Fill(candidate.pt());
        registry.get<TH1>("hCPARecBg")->Fill(candidate.cpa());
        registry.get<TH1>("hEtaRecBg")->Fill(candidate.eta());
      }
    }
    // MC gen.
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (cutEtaCandMax >= 0. && std::abs(particle.eta()) > cutEtaCandMax) {
        //Printf("MC Gen.: eta rejection: %g", particle.eta());
        continue;
      }
      if (std::abs(particle.flagMCMatchGen()) == LcToPKPi) {
        registry.get<TH1>("hPtGen")->Fill(particle.pt());
        registry.get<TH1>("hEtaGen")->Fill(particle.eta());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskLc>("hf-task-lc")};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskLcMC>("hf-task-lc-mc"));
  }
  return workflow;
}
