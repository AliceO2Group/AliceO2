// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskJpsi.cxx
/// \brief Jpsi analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// jpsi2ee analysis task
struct TaskJpsi {
  HistogramRegistry registry{
    "registry",
    {{"hmass", "2-prong candidates;inv. mass (#e+ e-) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hptcand", "2-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0", "2-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1", "2-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hdeclength", "2-prong candidates;decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hdeclengthxy", "2-prong candidates;decay length xy (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hd0Prong0", "2-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong1", "2-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0d0", "2-prong candidates;product of DCAxy to prim. vertex (cm^{2});entries", {HistType::kTH1F, {{500, -1., 1.}}}},
     {"hCPA", "2-prong candidates;cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEta", "2-prong candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hImpParErr", "2-prong candidates;impact parameter error (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hDecLenErr", "2-prong candidates;decay length error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
     {"hDecLenXYErr", "2-prong candidates;decay length xy error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}}}};

  Configurable<int> d_selectionFlagJpsi{"d_selectionFlagJpsi", 1, "Selection Flag for Jpsi"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_jpsi::isSelJpsi >= d_selectionFlagJpsi);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelJpsiCandidate>> const& candidates)
  //  void process(aod::HfCandProng2 const& candidates)
  {
    //Printf("Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        //Printf("Candidate: eta rejection: %g", candidate.eta());
        continue;
      }

      registry.get<TH1>(HIST("hmass"))->Fill(InvMassJpsi(candidate));
      registry.get<TH1>(HIST("hptcand"))->Fill(candidate.pt());
      registry.get<TH1>(HIST("hptprong0"))->Fill(candidate.ptProng0());
      registry.get<TH1>(HIST("hptprong1"))->Fill(candidate.ptProng1());
      registry.get<TH1>(HIST("hdeclength"))->Fill(candidate.decayLength());
      registry.get<TH1>(HIST("hdeclengthxy"))->Fill(candidate.decayLengthXY());
      registry.get<TH1>(HIST("hd0Prong0"))->Fill(candidate.impactParameter0());
      registry.get<TH1>(HIST("hd0Prong1"))->Fill(candidate.impactParameter1());
      registry.get<TH1>(HIST("hd0d0"))->Fill(candidate.impactParameterProduct());
      registry.get<TH1>(HIST("hCPA"))->Fill(candidate.cpa());
      registry.get<TH1>(HIST("hEta"))->Fill(candidate.eta());
      registry.get<TH1>(HIST("hImpParErr"))->Fill(candidate.errorImpactParameter0());
      registry.get<TH1>(HIST("hImpParErr"))->Fill(candidate.errorImpactParameter1());
      registry.get<TH1>(HIST("hDecLenErr"))->Fill(candidate.errorDecayLength());
      registry.get<TH1>(HIST("hDecLenXYErr"))->Fill(candidate.errorDecayLengthXY());
    }
  }
};

/// Fills MC histograms.
struct TaskJpsiMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "2-prong candidates (rec. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "2-prong candidates (rec. unmatched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "2-prong candidates (gen. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hCPARecSig", "2-prong candidates (rec. matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "2-prong candidates (rec. unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "2-prong candidates (rec. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "2-prong candidates (rec. unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "2-prong candidates (gen. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}}}};

  Configurable<int> d_selectionFlagJpsi{"d_selectionFlagJpsi", 1, "Selection Flag for Jpsi"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Filter filterSelectCandidates = (aod::hf_selcandidate_jpsi::isSelJpsi >= d_selectionFlagJpsi);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelJpsiCandidate, aod::HfCandProng2MCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    // MC rec.
    //Printf("MC Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        //Printf("MC Rec.: eta rejection: %g", candidate.eta());
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == JpsiToEE) {
        registry.get<TH1>(HIST("hPtRecSig"))->Fill(candidate.pt());
        registry.get<TH1>(HIST("hCPARecSig"))->Fill(candidate.cpa());
        registry.get<TH1>(HIST("hEtaRecSig"))->Fill(candidate.eta());
      } else {
        registry.get<TH1>(HIST("hPtRecBg"))->Fill(candidate.pt());
        registry.get<TH1>(HIST("hCPARecBg"))->Fill(candidate.cpa());
        registry.get<TH1>(HIST("hEtaRecBg"))->Fill(candidate.eta());
      }
    }
    // MC gen.
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (cutEtaCandMax >= 0. && std::abs(particle.eta()) > cutEtaCandMax) {
        //Printf("MC Gen.: eta rejection: %g", particle.eta());
        continue;
      }
      if (std::abs(particle.flagMCMatchGen()) == JpsiToEE) {
        registry.get<TH1>(HIST("hPtGen"))->Fill(particle.pt());
        registry.get<TH1>(HIST("hEtaGen"))->Fill(particle.eta());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskJpsi>("hf-task-jpsi")};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskJpsiMC>("hf-task-jpsi-mc"));
  }
  return workflow;
}
