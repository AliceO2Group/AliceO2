// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskBPlus.cxx
/// \brief B+ analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Antonio Palasciano <antonio.palasciano@cern.ch>,
/// \author Deepa Thomas <deepa.thomas@cern.ch>, UT Austin

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// B+ analysis task
struct TaskBPlus {
  HistogramRegistry registry{
    "registry",
    {{"hMass", "2-prong candidates;inv. mass (D0(bar) #pi) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 10.}}}},
     {"hPtProng0", "2-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtProng1", "2-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hdeclength", "2-prong candidates;decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hdeclengthxy", "2-prong candidates;decay length xy (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hd0Prong0", "2-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong1", "2-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hCPA", "2-prong candidates;cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEta", "2-prong candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hImpParErr", "2-prong candidates;impact parameter error (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hDecLenErr", "2-prong candidates;decay length error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
     {"hDecLenXYErr", "2-prong candidates;decay length xy error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
     {"hPtCand", "2-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  // Configurable<int> selectionFlagBPlus{"selectionFlagBPlus", 1, "Selection Flag for BPlus"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  // Filter filterSelectCandidates = (aod::hf_selcandidate_bplus::isSelBPlusToD0Pi >= selectionFlagBPlus);

  // To be Filtered once selector ready
  void process(aod::HfCandBPlus const& candidates)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1)) { // << BPlusToD0Pi
        continue;
      }

      registry.fill(HIST("hMass"), InvMassBplus(candidate));
      registry.fill(HIST("hPtCand"), candidate.pt());
      registry.fill(HIST("hPtProng0"), candidate.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate.ptProng1());
      registry.fill(HIST("hdeclength"), candidate.decayLength());
      registry.fill(HIST("hdeclengthxy"), candidate.decayLengthXY());
      registry.fill(HIST("hd0Prong0"), candidate.impactParameter0());
      registry.fill(HIST("hd0Prong1"), candidate.impactParameter1());
      registry.fill(HIST("hCPA"), candidate.cpa());
      registry.fill(HIST("hEta"), candidate.eta());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter0());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter1());
      registry.fill(HIST("hDecLenErr"), candidate.errorDecayLength());
      registry.fill(HIST("hDecLenXYErr"), candidate.errorDecayLengthXY());
    } // candidate loop
  }   // process
};    // struct

struct TaskBPlusMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "2-prong candidates (rec. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "2-prong candidates (rec. unmatched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "2-prong candidates (gen. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenProng0", "2-prong candidates (gen. matched);prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenProng1", "2-prong candidates (gen. matched);prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenSig", "2-prong candidates (rec. matched);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hCPARecSig", "2-prong candidates (rec. matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "2-prong candidates (rec. unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "2-prong candidates (rec. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "2-prong candidates (rec. unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "2-prong candidates (gen. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hMassRecSig", "2-prong candidates (rec. matched);inv. mass (D0bar #pi+) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 10.}}}},
     {"hMassRecBg", "2-prong candidates (rec. unmatched);inv. mass (D0bar #pi+) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 10.}}}},
     {"hDecLengthRecSig", "2-prong candidates (rec. matched);decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hDecLengthRecBg", "2-prong candidates (rec. unmatched);decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}}}};

  // Configurable<int> selectionFlagB{"selectionFlagB", 1, "Selection Flag for B"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "Upper cut rap."};
  // Filter filterSelectCandidates = (aod::hf_selcandidate_bplus::isSelBplusToD0Pi >= selectionFlagB);

  void process(soa::Join<aod::HfCandBPlus, aod::HfCandBPMCRec> const& candidates, soa::Join<aod::McParticles, aod::HfCandBPMCGen> const& particlesMC, aod::BigTracksMC const& tracks)
  {
    // MC rec.
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1)) { // << BPlusToD0Pi
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YBplus(candidate)) > cutYCandMax) {
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == 1) { // << BPlusToD0Pi
        // Get the corresponding MC particle.
        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index1_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandBPMCGen>>(), 521, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt());
        registry.fill(HIST("hPtRecSig"), candidate.pt());
        registry.fill(HIST("hCPARecSig"), candidate.cpa());
        registry.fill(HIST("hEtaRecSig"), candidate.eta());
        registry.fill(HIST("hDecLengthRecSig"), candidate.decayLength());
        if (candidate.flagMCMatchRec() == 1) {
          registry.fill(HIST("hMassRecSig"), InvMassBplus(candidate));
        } else {
          registry.fill(HIST("hMassRecSig"), InvMassBminus(candidate));
        }
      } else {
        registry.fill(HIST("hPtRecBg"), candidate.pt());
        registry.fill(HIST("hCPARecBg"), candidate.cpa());
        registry.fill(HIST("hEtaRecBg"), candidate.eta());
        registry.fill(HIST("hDecLengthRecBg"), candidate.decayLength());
        if (candidate.flagMCMatchRec() == 1) {
          registry.fill(HIST("hMassRecBg"), InvMassBplus(candidate));
        } else {
          registry.fill(HIST("hMassRecBg"), InvMassBminus(candidate));
        }
      }
    } // rec
    // MC gen. level
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (std::abs(particle.flagMCMatchGen()) == 1) { // << BPlusToD0Pi)
        registry.fill(HIST("hPtGen"), particle.pt());
        float ptProngs[2];
        int counter = 0;
        for (int iD = particle.daughter0(); iD <= particle.daughter1(); ++iD) {
          ptProngs[counter] = particlesMC.iteratorAt(iD).pt();
          counter++;
        }
        registry.fill(HIST("hPtGenProng0"), ptProngs[0]);
        registry.fill(HIST("hPtGenProng1"), ptProngs[1]);
        registry.fill(HIST("hEtaGen"), particle.eta());
      }
    } //gen
  }   // process
};    // struct

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskBPlus>(cfgc, TaskName{"hf-task-bplus"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskBPlusMC>(cfgc, TaskName{"hf-task-bplus-mc"}));
  }
  return workflow;
}