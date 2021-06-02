// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskXicc.cxx
/// \brief Ξcc±± analysis task
/// \note Inspired from taskLc.cxx
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Luigi Dello Stritto <luigi.dello.stritto@cern.ch >, SALERNO
/// \author Mattia Faggin <mattia.faggin@cern.ch>, University and INFN PADOVA

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand_xicc;
//using namespace o2::aod::hf_cand_prong3;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// Ξcc±± analysis task
struct TaskXicc {
  HistogramRegistry registry{
    "registry",
    {{"hptcand", "#Xi^{++}_{cc}-candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0", "#Xi^{++}_{cc}-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1", "#Xi^{++}_{cc}-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong2", "#Xi^{++}_{cc}-prong candidates;prong 2 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}}}};

  Configurable<int> d_selectionFlagXicc{"d_selectionFlagXicc", 1, "Selection Flag for Xicc"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<std::vector<double>> bins{"pTBins", std::vector<double>{hf_cuts_xicc_topkpipi::pTBins_v}, "pT bin limits"};

  void init(o2::framework::InitContext&)
  {
    auto vbins = (std::vector<double>)bins;
    registry.add("hmass", "#Xi^{++}_{cc} candidates;inv. mass (p K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{400, 3.2, 4.0}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLength", "#Xi^{++}_{cc} candidates;decay length (cm);entries", {HistType::kTH2F, {{500, 0., 0.05}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hChi2PCA", "#Xi^{++}_{cc} candidates;chi2 PCA (cm);entries", {HistType::kTH2F, {{500, 0., 0.01}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0", "#Xi^{++}_{cc} candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1", "#Xi^{++}_{cc} candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCt", "#Xi^{++}_{cc} candidates;proper lifetime (#Xi^{++}_{cc}) * #it{c} (cm);entries", {HistType::kTH2F, {{100, 0., 0.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPA", "#Xi^{++}_{cc} candidates;cosine of pointing angle;entries", {HistType::kTH2F, {{220, -1.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("habsCPA", "#Xi^{++}_{cc} candidates;abs. cosine of pointing angle;entries", {HistType::kTH2F, {{120, -0.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEta", "#Xi^{++}_{cc} candidates;candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hY", "#Xi^{++}_{cc} candidates;candidate rapidity;entries", {HistType::kTH2F, {{100, -2., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hselectionstatus", "#Xi^{++}_{cc} candidates;selection status;entries", {HistType::kTH2F, {{5, -0.5, 4.5}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr0", "#Xi^{++}_{cc} candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr1", "#Xi^{++}_{cc} candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLenErr", "#Xi^{++}_{cc} candidates;decay length error (cm);entries", {HistType::kTH2F, {{100, 0., 1.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_xicc::isSelXiccToPKPiPi >= d_selectionFlagXicc);
  void process(soa::Filtered<soa::Join<aod::HfCandXicc, aod::HFSelXiccToPKPiPiCandidate>> const& candidates)
  //void process(aod::HfCandXicc const& candidates)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << DecayType::XiccToXicPi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YXicc(candidate)) > cutYCandMax) {
        continue;
      }
      registry.fill(HIST("hmass"), InvMassXiccToXicPi(candidate), candidate.pt()); //FIXME need to consider the two mass hp
      registry.fill(HIST("hptcand"), candidate.pt());
      registry.fill(HIST("hptprong0"), candidate.ptProng0());
      registry.fill(HIST("hptprong1"), candidate.ptProng1());
      registry.fill(HIST("hDecLength"), candidate.decayLength(), candidate.pt());
      registry.fill(HIST("hChi2PCA"), candidate.chi2PCA(), candidate.pt());
      registry.fill(HIST("hd0Prong0"), abs(candidate.impactParameter0()), candidate.pt());
      registry.fill(HIST("hd0Prong1"), abs(candidate.impactParameter1()), candidate.pt());
      registry.fill(HIST("hCt"), CtXicc(candidate), candidate.pt());
      registry.fill(HIST("hCPA"), candidate.cpa(), candidate.pt());
      registry.fill(HIST("habsCPA"), abs(candidate.cpa()), candidate.pt());
      registry.fill(HIST("hEta"), candidate.eta(), candidate.pt());
      registry.fill(HIST("hY"), YXicc(candidate), candidate.pt());
      registry.fill(HIST("hselectionstatus"), candidate.isSelXiccToPKPiPi(), candidate.pt());
      registry.fill(HIST("hImpParErr0"), candidate.errorImpactParameter0(), candidate.pt());
      registry.fill(HIST("hImpParErr1"), candidate.errorImpactParameter1(), candidate.pt());
      registry.fill(HIST("hDecLenErr"), candidate.errorDecayLength(), candidate.pt());
    }
  }
};

/// Fills MC histograms.
struct TaskXiccMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "#Xi^{++}_{cc} candidates (rec. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtRecBg", "#Xi^{++}_{cc} candidates (rec. unmatched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGen", "#Xi^{++}_{cc} MC particles (matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hPtGenSig", "#Xi^{++}_{cc} candidates (rec. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hEtaRecSig", "#Xi^{++}_{cc} candidates (rec. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "#Xi^{++}_{cc} candidates (rec. unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hYRecSig", "#Xi^{++}_{cc} candidates (rec. matched);rapidity;entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hYRecBg", "#Xi^{++}_{cc} candidates (rec. unmatched);rapidity;entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "#Xi^{++}_{cc} MC particles (matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hYGen", "#Xi^{++}_{cc} MC particles (matched);rapidity;entries", {HistType::kTH1F, {{100, -2., 2.}}}}}};

  Configurable<int> d_selectionFlagXicc{"d_selectionFlagXicc", 1, "Selection Flag for Xicc"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<std::vector<double>> bins{"pTBins", std::vector<double>{hf_cuts_xicc_topkpipi::pTBins_v}, "pT bin limits"};

  void init(o2::framework::InitContext&)
  {
    auto vbins = (std::vector<double>)bins;
    registry.add("hmassSig", "#Xi^{++}_{cc} (rec. matched) candidates;inv. mass (p K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{400, 3.2, 4.0}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hmassBg", "#Xi^{++}_{cc} (rec. unmatched) candidates;inv. mass (p K #pi) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{400, 3.2, 4.0}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hChi2PCASig", "#Xi^{++}_{cc} (rec. matched) candidates;chi2 PCA (cm);entries", {HistType::kTH2F, {{500, 0., 0.01}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hChi2PCABg", "#Xi^{++}_{cc} (rec. unmatched) candidates;chi2 PCA (cm);entries", {HistType::kTH2F, {{500, 0., 0.01}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthSig", "#Xi^{++}_{cc} (rec. matched) candidates;decay length (cm);entries", {HistType::kTH2F, {{500, 0., 0.05}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthBg", "#Xi^{++}_{cc} (rec. unmatched) candidates;decay length (cm);entries", {HistType::kTH2F, {{500, 0., 0.05}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0Sig", "#Xi^{++}_{cc} (rec. matched) candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0Bg", "#Xi^{++}_{cc} (rec. unmatched) candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1Sig", "#Xi^{++}_{cc} (rec. matched) candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1Bg", "#Xi^{++}_{cc} (rec. unmatched) candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCtSig", "#Xi^{++}_{cc} (rec. matched) candidates;proper lifetime (#Xi_{cc}) * #it{c} (cm);entries", {HistType::kTH2F, {{100, 0., 0.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCtBg", "#Xi^{++}_{cc} (rec. unmatched) candidates;proper lifetime (#Xi_{cc}) * #it{c} (cm);entries", {HistType::kTH2F, {{100, 0., 0.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPASig", "#Xi^{++}_{cc} (rec. matched) candidates;cosine of pointing angle;entries", {HistType::kTH2F, {{220, -1.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPABg", "#Xi^{++}_{cc} (rec. unmatched) candidates;cosine of pointing angle;entries", {HistType::kTH2F, {{220, -1.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("habsCPASig", "#Xi^{++}_{cc} (rec. matched) candidates;abs. cosine of pointing angle;entries", {HistType::kTH2F, {{120, -0.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("habsCPABg", "#Xi^{++}_{cc} (rec. unmatched) candidates;abs. cosine of pointing angle;entries", {HistType::kTH2F, {{120, -0.1, 1.1}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEtaSig", "#Xi^{++}_{cc} (rec. matched) candidates;candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEtaBg", "#Xi^{++}_{cc} (rec. unmatched) candidates;candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYSig", "#Xi^{++}_{cc} (rec. matched) candidates;candidate rapidity;entries", {HistType::kTH2F, {{100, -2., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYBg", "#Xi^{++}_{cc} (rec. unmatched) candidates;candidate rapidity;entries", {HistType::kTH2F, {{100, -2., 2.}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hselectionstatusSig", "#Xi^{++}_{cc} (rec. matched) candidates;selection status;entries", {HistType::kTH2F, {{5, -0.5, 4.5}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hselectionstatusBg", "#Xi^{++}_{cc} (rec. unmatched) candidates;selection status;entries", {HistType::kTH2F, {{5, -0.5, 4.5}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr0Sig", "#Xi^{++}_{cc} (rec. matched) candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr0Bg", "#Xi^{++}_{cc} (rec. unmatched) candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr1Sig", "#Xi^{++}_{cc} (rec. matched) candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr1Bg", "#Xi^{++}_{cc} (rec. unmatched) candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{200, 0, 0.02}, {vbins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_xicc::isSelXiccToPKPiPi >= d_selectionFlagXicc);
  //void process(soa::Filtered<soa::Join<aod::HfCandXicc, aod::HFSelXiccToPKPiPiCandidate>> const& candidates)
  void process(soa::Filtered<soa::Join<aod::HfCandXicc, aod::HFSelXiccToPKPiPiCandidate, aod::HfCandXiccMCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandXiccMCGen> const& particlesMC, aod::BigTracksMC const& tracks)
  {
    // MC rec.
    //Printf("MC Candidates: %d", candidates.size());
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << DecayType::XiccToXicPi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YXicc(candidate)) > cutYCandMax) {
        continue;
      }
      if (std::abs(candidate.flagMCMatchRec()) == 1 << DecayType::XiccToXicPi) {
        // Get the corresponding MC particle.
        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index1_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandXiccMCGen>>(), 4422, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt()); // gen. level pT
        registry.fill(HIST("hPtRecSig"), candidate.pt());      // rec. level pT
        registry.fill(HIST("hEtaRecSig"), candidate.eta());
        registry.fill(HIST("hYRecSig"), YXicc(candidate));

        registry.fill(HIST("hmassSig"), InvMassXiccToXicPi(candidate), candidate.pt()); //FIXME need to consider the two mass hp
        registry.fill(HIST("hDecLengthSig"), candidate.decayLength(), candidate.pt());
        registry.fill(HIST("hChi2PCASig"), candidate.chi2PCA(), candidate.pt());
        registry.fill(HIST("hCPASig"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("hd0Prong0Sig"), abs(candidate.impactParameter0()), candidate.pt());
        registry.fill(HIST("hd0Prong1Sig"), abs(candidate.impactParameter1()), candidate.pt());
        registry.fill(HIST("hCtSig"), CtXicc(candidate), candidate.pt());
        registry.fill(HIST("hCPASig"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("habsCPASig"), abs(candidate.cpa()), candidate.pt());
        registry.fill(HIST("hEtaSig"), candidate.eta(), candidate.pt());
        registry.fill(HIST("hYSig"), YXicc(candidate), candidate.pt());
        registry.fill(HIST("hImpParErr0Sig"), candidate.errorImpactParameter0(), candidate.pt());
        registry.fill(HIST("hImpParErr1Sig"), candidate.errorImpactParameter1(), candidate.pt());
      } else {
        registry.fill(HIST("hPtRecBg"), candidate.pt());
        registry.fill(HIST("hEtaRecBg"), candidate.eta());
        registry.fill(HIST("hYRecBg"), YXicc(candidate));
        registry.fill(HIST("hmassBg"), InvMassXiccToXicPi(candidate), candidate.pt()); //FIXME need to consider the two mass hp
        registry.fill(HIST("hDecLengthBg"), candidate.decayLength(), candidate.pt());
        registry.fill(HIST("hChi2PCABg"), candidate.chi2PCA(), candidate.pt());
        registry.fill(HIST("hCPABg"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("habsCPABg"), abs(candidate.cpa()), candidate.pt());
        registry.fill(HIST("hd0Prong0Bg"), abs(candidate.impactParameter0()), candidate.pt());
        registry.fill(HIST("hd0Prong1Bg"), abs(candidate.impactParameter1()), candidate.pt());
        registry.fill(HIST("hCtBg"), CtXicc(candidate), candidate.pt());
        registry.fill(HIST("hCPABg"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("hEtaBg"), candidate.eta(), candidate.pt());
        registry.fill(HIST("hYBg"), YXicc(candidate), candidate.pt());
        registry.fill(HIST("hImpParErr0Bg"), candidate.errorImpactParameter0(), candidate.pt());
        registry.fill(HIST("hImpParErr1Bg"), candidate.errorImpactParameter1(), candidate.pt());
      }
    } // end of loop over reconstructed candidates
    // MC gen.
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (std::abs(particle.flagMCMatchGen()) == 1 << DecayType::XiccToXicPi) {
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode()))) > cutYCandMax) {
          continue;
        }
        registry.fill(HIST("hPtGen"), particle.pt());
        registry.fill(HIST("hEtaGen"), particle.eta());
        registry.fill(HIST("hYGen"), RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode())));
      }
    } // end of loop of MC particles
  }   // end of process function
};    // end of struct

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskXicc>(cfgc, TaskName{"hf-task-xicc"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskXiccMC>(cfgc, TaskName{"hf-task-xicc-mc"}));
  }
  return workflow;
}
