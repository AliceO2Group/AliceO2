// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskX.cxx
/// \brief X(3872) analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Rik Spijkers <r.spijkers@students.uu.nl>, Utrecht University

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::aod::hf_cand_x;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand_prong2;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, true, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// X(3872) analysis task
struct TaskX {
  HistogramRegistry registry{
    "registry",
    {{"hMass", "3-prong candidates;inv. mass (J/#psi #pi+ #pi-) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 10.}}}},
     {"hPtProng0", "3-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtProng1", "3-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtProng2", "3-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hdeclength", "3-prong candidates;decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hdeclengthxy", "3-prong candidates;decay length xy (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hd0Prong0", "3-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong1", "3-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong2", "3-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hCPA", "3-prong candidates;cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEta", "3-prong candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hImpParErr", "3-prong candidates;impact parameter error (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hDecLenErr", "3-prong candidates;decay length error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
     {"hDecLenXYErr", "3-prong candidates;decay length xy error (cm);entries", {HistType::kTH1F, {{100, 0., 1.}}}},
     {"hPtCand", "3-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  // Configurable<int> selectionFlagX{"selectionFlagX", 1, "Selection Flag for X"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};

  // Filter filterSelectCandidates = (aod::hf_selcandidate_x::isSelXToJpsiPiPi >= selectionFlagX);

  // X candidates are not soa::Filtered, should be added when candidate selector is added
  void process(aod::HfCandX const& candidates)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << XToJpsiPiPi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YX(candidate)) > cutYCandMax) {
        continue;
      }

      registry.fill(HIST("hMass"), InvMassXToJpsiPiPi(candidate));
      registry.fill(HIST("hPtCand"), candidate.pt());
      registry.fill(HIST("hPtProng0"), candidate.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate.ptProng1());
      registry.fill(HIST("hPtProng2"), candidate.ptProng2());
      registry.fill(HIST("hdeclength"), candidate.decayLength());
      registry.fill(HIST("hdeclengthxy"), candidate.decayLengthXY());
      registry.fill(HIST("hd0Prong0"), candidate.impactParameter0());
      registry.fill(HIST("hd0Prong1"), candidate.impactParameter1());
      registry.fill(HIST("hd0Prong2"), candidate.impactParameter2());
      registry.fill(HIST("hCPA"), candidate.cpa());
      registry.fill(HIST("hEta"), candidate.eta());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter0());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter1());
      registry.fill(HIST("hDecLenErr"), candidate.errorDecayLength());
      registry.fill(HIST("hDecLenXYErr"), candidate.errorDecayLengthXY());
    } // candidate loop
  }   // process
};    // struct

struct TaskXMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "3-prong candidates (rec. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "3-prong candidates (rec. unmatched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "3-prong candidates (gen. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenSig", "3-prong candidates (rec. matched);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hCPARecSig", "3-prong candidates (rec. matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "3-prong candidates (rec. unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "3-prong candidates (rec. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "3-prong candidates (rec. unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "3-prong candidates (gen. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},

     {"hPtGenProng0", "3-prong candidates (gen. matched);prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenProng1", "3-prong candidates (gen. matched);prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenProng2", "3-prong candidates (gen. matched);prong 2 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hMassRecSig", "3-prong candidates (rec. matched);inv. mass (J/#psi #pi+ #pi-) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 10.}}}},
     {"hMassRecBg", "3-prong candidates (rec. unmatched);inv. mass (J/#psi #pi+ #pi-) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 10.}}}},
     {"hd0Prong0RecSig", "3-prong candidates (rec. matched);prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong1RecSig", "3-prong candidates (rec. matched);prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong2RecSig", "3-prong candidates (rec. matched);prong 2 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong0RecBg", "3-prong candidates (rec. unmatched);prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong1RecBg", "3-prong candidates (rec. unmatched);prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hd0Prong2RecBg", "3-prong candidates (rec. unmatched);prong 2 DCAxy to prim. vertex (cm);entries", {HistType::kTH1F, {{100, -1., 1.}}}},
     {"hDeclengthRecSig", "3-prong candidates (rec. matched);decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}},
     {"hDeclengthRecBg", "3-prong candidates (rec. unmatched);decay length (cm);entries", {HistType::kTH1F, {{200, 0., 2.}}}}}};

  // Configurable<int> selectionFlagX{"selectionFlagX", 1, "Selection Flag for X"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  // Filter filterSelectCandidates = (aod::hf_selcandidate_x::isSelXToJpsiPiPi >= selectionFlagX);

  void process(soa::Join<aod::HfCandX, aod::HfCandXMCRec> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandXMCGen> const& particlesMC,
               aod::BigTracksMC) // aod::BigTracksMC is needed to get corresponding MC particle in MC rec. part
  {
    // MC rec.
    //Printf("MC Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << XToJpsiPiPi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YX(candidate)) > cutYCandMax) {
        continue;
      }
      if (candidate.flagMCMatchRec() == 1 << XToJpsiPiPi) {
        // Get the corresponding MC particle. // TODO: fix the next 3 lines, gives very strange error
        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index1_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandXMCGen>>(), 9920443, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt());
        registry.fill(HIST("hPtRecSig"), candidate.pt());
        registry.fill(HIST("hCPARecSig"), candidate.cpa());
        registry.fill(HIST("hEtaRecSig"), candidate.eta());

        registry.fill(HIST("hDeclengthRecSig"), candidate.decayLength());
        registry.fill(HIST("hMassRecSig"), InvMassXToJpsiPiPi(candidate));
        registry.fill(HIST("hd0Prong0RecSig"), candidate.impactParameter0());
        registry.fill(HIST("hd0Prong1RecSig"), candidate.impactParameter1());
        registry.fill(HIST("hd0Prong2RecSig"), candidate.impactParameter2());
      } else {
        registry.fill(HIST("hPtRecBg"), candidate.pt());
        registry.fill(HIST("hCPARecBg"), candidate.cpa());
        registry.fill(HIST("hEtaRecBg"), candidate.eta());

        registry.fill(HIST("hDeclengthRecBg"), candidate.decayLength());
        registry.fill(HIST("hMassRecBg"), InvMassXToJpsiPiPi(candidate));
        registry.fill(HIST("hd0Prong0RecBg"), candidate.impactParameter0());
        registry.fill(HIST("hd0Prong1RecBg"), candidate.impactParameter1());
        registry.fill(HIST("hd0Prong2RecBg"), candidate.impactParameter2());
      }
    } // rec
    // MC gen.
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (particle.flagMCMatchGen() == 1 << XToJpsiPiPi) {
        // TODO: add X(3872) mass such that we can use the getMassPDG function
        // if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode()))) > cutYCandMax) {
        //   Printf("MC Gen.: Y rejection: %g", RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode())));
        //   continue;
        // }
        registry.fill(HIST("hPtGen"), particle.pt());
        registry.fill(HIST("hEtaGen"), particle.eta());

        // properties of gen matched X(3872), to get a first look at some cuts
        float ptProngs[3];
        int counter = 0;
        for (int iD = particle.daughter0(); iD <= particle.daughter1(); ++iD) {
          ptProngs[counter] = particlesMC.iteratorAt(iD).pt();
          counter++;
        }
        registry.fill(HIST("hPtGenProng0"), ptProngs[0]);
        registry.fill(HIST("hPtGenProng1"), ptProngs[1]);
        registry.fill(HIST("hPtGenProng2"), ptProngs[2]);
        registry.fill(HIST("hEtaGen"), particle.eta());

        // properties of gen matched X(3872), to get a first look at some cuts
        float ptProngs[3];
        int counter = 0;
        for (int iD = particle.daughter0(); iD <= particle.daughter1(); ++iD) {
          ptProngs[counter] = particlesMC.iteratorAt(iD).pt();
          counter++;
        }
        registry.fill(HIST("hPtGenProng0"), ptProngs[0]);
        registry.fill(HIST("hPtGenProng1"), ptProngs[1]);
        registry.fill(HIST("hPtGenProng2"), ptProngs[2]);
      }
    } //gen
  }   // process
};    // struct

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskX>(cfgc, TaskName{"hf-task-x"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskXMC>(cfgc, TaskName{"hf-task-x-mc"}));
  }
  return workflow;
}
