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
/// \author Fabio Catalano <fabio.catalano@cern.ch>, Politecnico and INFN Torino
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

/// D± analysis task
struct TaskDPlus {
  HistogramRegistry registry{
    "registry",
    {{"hMass", "3-prong candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{350, 1.7, 2.05}}}},
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
     {"hd0Prong2", "3-prong candidates;prong 2 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}}}};

  Configurable<int> d_selectionFlagDPlus{"d_selectionFlagDPlus", 1, "Selection Flag for DPlus"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_dplus::isSelDplusToPiKPi >= d_selectionFlagDPlus);

  void process(soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelDplusToPiKPiCandidate>> const& candidates)
  {
    for (auto& candidate : candidates) {
      //not possible in Filter since expressions do not support binary operators
      if (!(candidate.hfflag() & 1 << DPlusToPiKPi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YDPlus(candidate)) > cutYCandMax) {
        continue;
      }
      registry.fill(HIST("hMass"), InvMassDPlus(candidate));
      registry.fill(HIST("hPt"), candidate.pt());
      registry.fill(HIST("hEta"), candidate.eta());
      registry.fill(HIST("hCt"), CtDPlus(candidate));
      registry.fill(HIST("hDecayLength"), candidate.decayLength());
      registry.fill(HIST("hDecayLengthXY"), candidate.decayLengthXY());
      registry.fill(HIST("hNormalisedDecayLengthXY"), candidate.decayLengthXYNormalised());
      registry.fill(HIST("hCPA"), candidate.cpa());
      registry.fill(HIST("hCPAxy"), candidate.cpaXY());
      registry.fill(HIST("hImpactParameterXY"), candidate.impactParameterXY());
      registry.fill(HIST("hMaxNormalisedDeltaIP"), candidate.maxNormalisedDeltaIP());
      registry.fill(HIST("hImpactParameterProngSqSum"), candidate.impactParameterProngSqSum());
      registry.fill(HIST("hDecayLengthError"), candidate.errorDecayLength());
      registry.fill(HIST("hDecayLengthXYError"), candidate.errorDecayLengthXY());
      registry.fill(HIST("hImpactParameterError"), candidate.errorImpactParameter0());
      registry.fill(HIST("hImpactParameterError"), candidate.errorImpactParameter1());
      registry.fill(HIST("hImpactParameterError"), candidate.errorImpactParameter2());
      registry.fill(HIST("hPtProng0"), candidate.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate.ptProng1());
      registry.fill(HIST("hPtProng2"), candidate.ptProng2());
      registry.fill(HIST("hd0Prong0"), candidate.impactParameter0());
      registry.fill(HIST("hd0Prong1"), candidate.impactParameter1());
      registry.fill(HIST("hd0Prong2"), candidate.impactParameter2());
    }
  }
};

/// D± analysis task for MC
struct TaskDPlusMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "3-prong candidates (matched);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "3-prong candidates (unmatched);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "MC particles (matched);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenSig", "3-prong candidates (matched);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hCPARecSig", "3-prong candidates (matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "3-prong candidates (unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "3-prong candidates (matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "3-prong candidates (unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "MC particles (matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}}}};

  Configurable<int> d_selectionFlagDPlus{"d_selectionFlagDPlus", 1, "Selection Flag for DPlus"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_dplus::isSelDplusToPiKPi >= d_selectionFlagDPlus);

  void process(soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelDplusToPiKPiCandidate, aod::HfCandProng3MCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandProng3MCGen> const& particlesMC, aod::BigTracksMC const& tracks)
  {
    // MC rec.
    for (auto& candidate : candidates) {
      //not possible in Filter since expressions do not support binary operators
      if (!(candidate.hfflag() & 1 << DPlusToPiKPi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YDPlus(candidate)) > cutYCandMax) {
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == 1 << DPlusToPiKPi) {
        // Get the corresponding MC particle.
        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index0_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandProng3MCGen>>(), 411, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt()); //gen. level pT
        registry.fill(HIST("hPtRecSig"), candidate.pt());      //rec. level pT
        registry.fill(HIST("hCPARecSig"), candidate.cpa());
        registry.fill(HIST("hEtaRecSig"), candidate.eta());
      } else {
        registry.fill(HIST("hPtRecBg"), candidate.pt());
        registry.fill(HIST("hCPARecBg"), candidate.cpa());
        registry.fill(HIST("hEtaRecBg"), candidate.eta());
      }
    }
    // MC gen.
    for (auto& particle : particlesMC) {
      if (std::abs(particle.flagMCMatchGen()) == 1 << DPlusToPiKPi) {
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode()))) > cutYCandMax) {
          continue;
        }
        registry.fill(HIST("hPtGen"), particle.pt());
        registry.fill(HIST("hEtaGen"), particle.eta());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskDPlus>(cfgc, "hf-task-dplus")};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskDPlusMC>(cfgc, "hf-task-dplus-mc"));
  }
  return workflow;
}
