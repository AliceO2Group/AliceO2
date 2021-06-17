// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskLcK0sp.cxx
/// \brief Lc -> K0S+p analysis task
///
/// \author Chiara Zampolli, <Chiara.Zampolli@cern.ch>, CERN
///
/// based on taskD0.cxx, taskLc.cxx

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_casc;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, true, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// LcK0sp analysis task
struct TaskLcK0sP {
  HistogramRegistry registry{
    "registry",
    {{"hMass", "cascade candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0.0f, 5.0f}}}},
     {"hPtCand", "cascade candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0.0f, 10.0f}}}},
     {"hPtBach", "cascade candidates;bachelor #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0.0f, 10.0f}}}},
     {"hPtV0", "cascade candidates;v0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0.0f, 10.0f}}}},
     {"hd0Bach", "cascade candidates;bachelor DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1.0f, 1.0f}}}},
     {"hd0V0pos", "cascade candidates;pos daugh v0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -5.0f, 5.0f}}}},
     {"hd0V0neg", "cascade candidates;neg daugh v0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -5.0f, 5.0f}}}},
     {"hV0CPA", "cascade candidates;v0 cosine of pointing angle;entries", {HistType::kTH1F, {{110, -0.98f, 1.1f}}}},
     {"hEta", "cascade candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -2.0f, 2.0f}}}},
     {"hSelectionStatus", "cascade candidates;selection status;entries", {HistType::kTH1F, {{5, -0.5f, 4.5f}}}}}};

  Configurable<int> selectionFlagLcK0sp{"selectionFlagLcK0sp", 1, "Selection Flag for LcK0sp"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_lc_k0sp::isSelLcK0sP >= selectionFlagLcK0sp);

  void process(soa::Filtered<soa::Join<aod::HfCandCascExt, aod::HFSelLcK0sPCandidate>> const& candidates)
  {
    //Printf("Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      /*
      // no such selection for LcK0sp for now - it is the only cascade
      if (!(candidate.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      */
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        //Printf("Candidate: eta rejection: %g", candidate.eta());
        continue;
      }

      registry.fill(HIST("hMass"), InvMassLcToK0sP(candidate));
      registry.fill(HIST("hPtCand"), candidate.pt());
      registry.fill(HIST("hPtBach"), candidate.ptProng0());
      registry.fill(HIST("hPtV0"), candidate.ptProng1());
      registry.fill(HIST("hd0Bach"), candidate.impactParameter0());
      registry.fill(HIST("hd0V0pos"), candidate.dcapostopv());
      registry.fill(HIST("hd0V0neg"), candidate.dcanegtopv());
      registry.fill(HIST("hV0CPA"), candidate.v0cosPA());
      registry.fill(HIST("hEta"), candidate.eta());
      registry.fill(HIST("hSelectionStatus"), candidate.isSelLcK0sP());
    }
  }
};

/// Fills MC histograms.
struct TaskLcK0SpMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "cascade candidates (MC);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "cascade candidates (unmatched);#it{p}_{T}^{rec.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "cascade (MC);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenSig", "cascade candidates (MC);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hCPARecSig", "cascade candidates (matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "cascade candidates (unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "cascade candidates (matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "cascade candidates (unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "MC particles (MC);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}}}};

  Configurable<int> selectionFlagLc{"selectionFlagLc", 1, "Selection Flag for Lc"};
  Configurable<int> selectionFlagLcbar{"selectionFlagLcbar", 1, "Selection Flag for Lcbar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_lc_k0sp::isSelLcK0sP >= selectionFlagLc || aod::hf_selcandidate_lc_k0sp::isSelLcK0sP >= selectionFlagLcbar);

  void process(soa::Filtered<soa::Join<aod::HfCandCascExt, aod::HFSelLcK0sPCandidate, aod::HfCandCascadeMCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandCascadeMCGen> const& particlesMC, aod::BigTracksMC const& tracks)
  {
    // MC rec.
    //Printf("MC Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      if (cutEtaCandMax >= 0. && std::abs(candidate.eta()) > cutEtaCandMax) {
        //Printf("MC Rec.: eta rejection: %g", candidate.eta());
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == 1) {
        // Get the corresponding MC particle.
        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index0_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandCascadeMCGen>>(), pdg::Code::kLambdaCPlus, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt()); // gen. level pT
        registry.fill(HIST("hPtRecSig"), candidate.pt());      // rec. level pT
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
      if (cutEtaCandMax >= 0. && std::abs(particle.eta()) > cutEtaCandMax) {
        //Printf("MC Gen.: eta rejection: %g", particle.eta());
        continue;
      }
      if (std::abs(particle.flagMCMatchGen()) == 1) {
        registry.fill(HIST("hPtGen"), particle.pt());
        registry.fill(HIST("hEtaGen"), particle.eta());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskLcK0sP>(cfgc, TaskName{"hf-task-lc-tok0sP"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskLcK0SpMC>(cfgc, TaskName{"hf-task-lc-tok0sP-mc"}));
  }
  return workflow;
}
