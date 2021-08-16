// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskD0.cxx
/// \brief D0 analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_d0_topik;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, true, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// D0 analysis task
struct TaskD0 {
  HistogramRegistry registry{
    "registry",
    {{"hptcand", "2-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0", "2-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1", "2-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<std::vector<double>> bins{"pTBins", std::vector<double>{hf_cuts_d0_topik::pTBins_v}, "pT bin limits"};

  void init(o2::framework::InitContext&)
  {
    auto vbins = (std::vector<double>)bins;
    registry.add("hmass", "2-prong candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{500, 0., 5.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclength", "2-prong candidates;decay length (cm);entries", {HistType::kTH2F, {{200, 0., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclengthxy", "2-prong candidates;decay length xy (cm);entries", {HistType::kTH2F, {{200, 0., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0", "2-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{100, -1., 1.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1", "2-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{100, -1., 1.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0d0", "2-prong candidates;product of DCAxy to prim. vertex (cm^{2});entries", {HistType::kTH2F, {{500, -1., 1.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCTS", "2-prong candidates;cos #it{#theta}* (D^{0});entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCt", "2-prong candidates;proper lifetime (D^{0}) * #it{c} (cm);entries", {HistType::kTH2F, {{120, -20., 100.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPA", "2-prong candidates;cosine of pointing angle;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEta", "2-prong candidates;candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hselectionstatus", "2-prong candidates;selection status;entries", {HistType::kTH2F, {{5, -0.5, 4.5}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr", "2-prong candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{100, -1., 1.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLenErr", "2-prong candidates;decay length error (cm);entries", {HistType::kTH2F, {{100, 0., 1.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLenXYErr", "2-prong candidates;decay length xy error (cm);entries", {HistType::kTH2F, {{100, 0., 1.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << DecayType::D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate)) > cutYCandMax) {
        continue;
      }

      if (candidate.isSelD0() >= d_selectionFlagD0) {
        registry.fill(HIST("hmass"), InvMassD0(candidate), candidate.pt());
      }
      if (candidate.isSelD0bar() >= d_selectionFlagD0bar) {
        registry.fill(HIST("hmass"), InvMassD0bar(candidate), candidate.pt());
      }

      registry.fill(HIST("hptcand"), candidate.pt());
      registry.fill(HIST("hptprong0"), candidate.ptProng0());
      registry.fill(HIST("hptprong1"), candidate.ptProng1());
      registry.fill(HIST("hdeclength"), candidate.decayLength(), candidate.pt());
      registry.fill(HIST("hdeclengthxy"), candidate.decayLengthXY(), candidate.pt());
      registry.fill(HIST("hd0Prong0"), candidate.impactParameter0(), candidate.pt());
      registry.fill(HIST("hd0Prong1"), candidate.impactParameter1(), candidate.pt());
      registry.fill(HIST("hd0d0"), candidate.impactParameterProduct(), candidate.pt());
      registry.fill(HIST("hCTS"), CosThetaStarD0(candidate), candidate.pt());
      registry.fill(HIST("hCt"), CtD0(candidate), candidate.pt());
      registry.fill(HIST("hCPA"), candidate.cpa(), candidate.pt());
      registry.fill(HIST("hEta"), candidate.eta(), candidate.pt());
      registry.fill(HIST("hselectionstatus"), candidate.isSelD0() + (candidate.isSelD0bar() * 2), candidate.pt());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter0(), candidate.pt());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter1(), candidate.pt());
      registry.fill(HIST("hDecLenErr"), candidate.errorDecayLength(), candidate.pt());
      registry.fill(HIST("hDecLenXYErr"), candidate.errorDecayLengthXY(), candidate.pt());
    }
  }
};

/// Fills MC histograms.
struct TaskD0MC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "2-prong candidates (matched);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecSigPrompt", "2-prong candidates (matched, prompt);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecSigNonPrompt", "2-prong candidates (matched, non-prompt);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "2-prong candidates (unmatched);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "MC particles (matched);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenPrompt", "MC particles (matched, prompt);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenNonPrompt", "MC particles (matched, non-prompt);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenSig", "2-prong candidates (matched);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hCPARecSig", "2-prong candidates (matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "2-prong candidates (unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "2-prong candidates (matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "2-prong candidates (unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "MC particles (matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hPtvsYRecSig_RecoPID", "2-prong candidates (RecoPID - matched);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigPrompt_RecoPID", "2-prong candidates (RecoPID - matched, prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigNonPrompt_RecoPID", "2-prong candidates (RecoPID - matched, non-prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSig_RecoCand", "2-prong candidates (RecoCand - matched);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigPrompt_RecoCand", "2-prong candidates (RecoCand - matched, prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigNonPrompt_RecoCand", "2-prong candidates (RecoCand - matched, non-prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSig_RecoTopol", "2-prong candidates (RecoTopol - matched);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigPrompt_RecoTopol", "2-prong candidates (RecoTopol - matched, prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigNonPrompt_RecoTopol", "2-prong candidates (RecoTopol - matched, non-prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSig_RecoHFFlag", "2-prong candidates (RecoHFFlag - matched);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigPrompt_RecoHFFlag", "2-prong candidates (RecoHFFlag - matched, prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYRecSigNonPrompt_RecoHFFlag", "2-prong candidates (RecoHFFlag - matched, non-prompt);#it{p}_{T}^{rec.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYGen", "2-prong candidates (matched);#it{p}_{T}^{gen.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYGenPrompt", "2-prong candidates (matched, prompt);#it{p}_{T}^{gen.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}},
     {"hPtvsYGenNonPrompt", "2-prong candidates (matched, non-prompt);#it{p}_{T}^{gen.}; #it{y}", {HistType::kTH2F, {{100, 0., 10.}, {100, -2., 2.}}}}}};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<int> d_selectionHFFlag{"d_selectionHFFlag", 1, "Selection Flag for HF flagged candidates"};
  Configurable<int> d_selectionTopol{"d_selectionTopol", 1, "Selection Flag for topologically selected candidates"};
  Configurable<int> d_selectionCand{"d_selectionCand", 1, "Selection Flag for conj. topol. selected candidates"};
  Configurable<int> d_selectionPID{"d_selectionPID", 1, "Selection Flag for reco PID candidates"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isRecoHFFlag >= d_selectionHFFlag);
  //Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HfCandProng2MCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC, aod::BigTracksMC const& tracks)
  {
    // MC rec.
    //Printf("MC Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << DecayType::D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate)) > cutYCandMax) {
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == 1 << DecayType::D0ToPiK) {
        // Get the corresponding MC particle.
        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index0_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandProng2MCGen>>(), pdg::Code::kD0, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt()); // gen. level pT
        auto ptRec = candidate.pt();
        auto yRec = YD0(candidate);
        registry.fill(HIST("hPtRecSig"), ptRec); // rec. level pT
        if (candidate.isRecoHFFlag() >= d_selectionHFFlag) {
          registry.fill(HIST("hPtvsYRecSig_RecoHFFlag"), ptRec, yRec);
        }
        if (candidate.isRecoTopol() >= d_selectionTopol) {
          registry.fill(HIST("hPtvsYRecSig_RecoTopol"), ptRec, yRec);
        }
        if (candidate.isRecoCand() >= d_selectionCand) {
          registry.fill(HIST("hPtvsYRecSig_RecoCand"), ptRec, yRec);
        }
        if (candidate.isRecoPID() >= d_selectionPID) {
          registry.fill(HIST("hPtvsYRecSig_RecoPID"), ptRec, yRec);
        }

        if (candidate.originMCRec() == OriginType::Prompt) {
          registry.fill(HIST("hPtRecSigPrompt"), ptRec); // rec. level pT, prompt
          if (candidate.isRecoHFFlag() >= d_selectionHFFlag) {
            registry.fill(HIST("hPtvsYRecSigPrompt_RecoHFFlag"), ptRec, yRec);
          }
          if (candidate.isRecoTopol() >= d_selectionTopol) {
            registry.fill(HIST("hPtvsYRecSigPrompt_RecoTopol"), ptRec, yRec);
          }
          if (candidate.isRecoCand() >= d_selectionCand) {
            registry.fill(HIST("hPtvsYRecSigPrompt_RecoCand"), ptRec, yRec);
          }
          if (candidate.isRecoPID() >= d_selectionPID) {
            registry.fill(HIST("hPtvsYRecSigPrompt_RecoPID"), ptRec, yRec);
          }
        } else {
          registry.fill(HIST("hPtRecSigNonPrompt"), ptRec); // rec. level pT, non-prompt
          if (candidate.isRecoHFFlag() >= d_selectionHFFlag) {
            registry.fill(HIST("hPtvsYRecSigNonPrompt_RecoHFFlag"), ptRec, yRec);
          }
          if (candidate.isRecoTopol() >= d_selectionTopol) {
            registry.fill(HIST("hPtvsYRecSigNonPrompt_RecoTopol"), ptRec, yRec);
          }
          if (candidate.isRecoCand() >= d_selectionCand) {
            registry.fill(HIST("hPtvsYRecSigNonPrompt_RecoCand"), ptRec, yRec);
          }
          if (candidate.isRecoPID() >= d_selectionPID) {
            registry.fill(HIST("hPtvsYRecSigNonPrompt_RecoPID"), ptRec, yRec);
          }
        }
        registry.fill(HIST("hCPARecSig"), candidate.cpa());
        registry.fill(HIST("hEtaRecSig"), candidate.eta());
      } else {
        registry.fill(HIST("hPtRecBg"), candidate.pt());
        registry.fill(HIST("hCPARecBg"), candidate.cpa());
        registry.fill(HIST("hEtaRecBg"), candidate.eta());
      }
    }
    // MC gen.
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (std::abs(particle.flagMCMatchGen()) == 1 << DecayType::D0ToPiK) {
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode()))) > cutYCandMax) {
          continue;
        }
        auto ptGen = particle.pt();
        auto yGen = RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode()));
        registry.fill(HIST("hPtGen"), ptGen);
        registry.fill(HIST("hPtvsYGen"), ptGen, yGen);
        if (particle.originMCGen() == OriginType::Prompt) {
          registry.fill(HIST("hPtGenPrompt"), ptGen);
          registry.fill(HIST("hPtvsYGenPrompt"), ptGen, yGen);
        } else {
          registry.fill(HIST("hPtGenNonPrompt"), ptGen);
          registry.fill(HIST("hPtvsYGenNonPrompt"), ptGen, yGen);
        }
        registry.fill(HIST("hEtaGen"), particle.eta());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskD0>(cfgc, TaskName{"hf-task-d0"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskD0MC>(cfgc, TaskName{"hf-task-d0-mc"}));
  }
  return workflow;
}
