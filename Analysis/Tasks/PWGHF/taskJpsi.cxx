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
/// \author Biao Zhang <biao.zhang@cern.ch>, CCNU

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_jpsi_toee;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, true, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// jpsitoee analysis task
struct TaskJpsi {
  HistogramRegistry registry{
    "registry",
    {{"hptcand", "2-prong candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 20.}}}},
     {"hptprong0", "2-prong candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 20.}}}},
     {"hptprong1", "2-prong candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 20.}}}}}};

  Configurable<int> d_selectionFlagJpsi{"d_selectionFlagJpsi", 0, "Selection Flag for Jpsi"};
  Configurable<bool> d_modeJpsiToMuMu{"d_modeJpsiToMuMu", false, "Perform Jpsi to mu+mu- analysis"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<std::vector<double>> bins{"pTBins", std::vector<double>{hf_cuts_jpsi_toee::pTBins_v}, "pT bin limits"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hmass", "2-prong candidates;inv. mass (l^{#plus} l^{#minus}) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{200, 2., 4.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclength", "2-prong candidates;decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclengthxy", "2-prong candidates;decay length xy (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0", "2-prong candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1", "2-prong candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0d0", "2-prong candidates;product of DCAxy to prim. vertex (cm^{2});entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPA", "2-prong candidates;cosine of pointing angle;entries", {HistType::kTH2F, {{1000, 0.5, 1.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEta", "2-prong candidates;candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr", "2-prong candidates;impact parameter error (cm);entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLenErr", "2-prong candidates;decay length error (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLenXYErr", "2-prong candidates;decay length xy error (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_jpsi::isSelJpsiToEE >= d_selectionFlagJpsi || aod::hf_selcandidate_jpsi::isSelJpsiToMuMu >= d_selectionFlagJpsi);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelJpsiCandidate>> const& candidates)
  {
    int decaymode = DecayType::JpsiToEE;
    if (d_modeJpsiToMuMu)
      decaymode = DecayType::JpsiToMuMu;

    for (auto& candidate : candidates) {
      if (d_selectionFlagJpsi > 0) {
        if (!d_modeJpsiToMuMu) {
          if (candidate.isSelJpsiToEE() <= 0)
            continue;
        } else {
          if (candidate.isSelJpsiToMuMu() <= 0)
            continue;
        }
      }
      if (!(candidate.hfflag() & 1 << decaymode)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YJpsi(candidate)) > cutYCandMax) {
        continue;
      }

      if (d_modeJpsiToMuMu) {
        registry.fill(HIST("hmass"), InvMassJpsiToEE(candidate), candidate.pt());
      } else {
        registry.fill(HIST("hmass"), InvMassJpsiToMuMu(candidate), candidate.pt());
      }
      registry.fill(HIST("hptcand"), candidate.pt());
      registry.fill(HIST("hptprong0"), candidate.ptProng0());
      registry.fill(HIST("hptprong1"), candidate.ptProng1());
      registry.fill(HIST("hdeclength"), candidate.decayLength(), candidate.pt());
      registry.fill(HIST("hdeclengthxy"), candidate.decayLengthXY(), candidate.pt());
      registry.fill(HIST("hd0Prong0"), candidate.impactParameter0(), candidate.pt());
      registry.fill(HIST("hd0Prong1"), candidate.impactParameter1(), candidate.pt());
      registry.fill(HIST("hd0d0"), candidate.impactParameterProduct(), candidate.pt());
      registry.fill(HIST("hCPA"), candidate.cpa(), candidate.pt());
      registry.fill(HIST("hEta"), candidate.eta(), candidate.pt());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter0(), candidate.pt());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter1(), candidate.pt());
      registry.fill(HIST("hDecLenErr"), candidate.errorDecayLength(), candidate.pt());
      registry.fill(HIST("hDecLenXYErr"), candidate.errorDecayLengthXY(), candidate.pt());
    }
  }
};

/// Fills MC histograms.
struct TaskJpsiMC {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "2-prong candidates (rec. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 20.}}}},
     {"hPtRecBg", "2-prong candidates (rec. unmatched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 20.}}}},
     {"hPtGen", "2-prong candidates (gen. matched);#it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 20.}}}},
     {"hPtGenSig", "2-prong candidates (rec. matched);#it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 20.}}}},
     {"hCPARecSig", "2-prong candidates (rec. matched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hCPARecBg", "2-prong candidates (rec. unmatched);cosine of pointing angle;entries", {HistType::kTH1F, {{110, -1.1, 1.1}}}},
     {"hEtaRecSig", "2-prong candidates (rec. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaRecBg", "2-prong candidates (rec. unmatched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}},
     {"hEtaGen", "2-prong candidates (gen. matched);#it{#eta};entries", {HistType::kTH1F, {{100, -2., 2.}}}}}};

  Configurable<int> d_selectionFlagJpsi{"d_selectionFlagJpsi", 1, "Selection Flag for Jpsi"};
  Configurable<bool> d_modeJpsiToMuMu{"d_modeJpsiToMuMu", false, "Perform Jpsi to mu+mu- analysis"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<std::vector<double>> bins{"pTBins", std::vector<double>{hf_cuts_jpsi_toee::pTBins_v}, "pT bin limits"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hmassSig", "2-prong candidates (rec matched);inv. mass (l^{#plus} l^{#minus}) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{200, 2., 4.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hmassBg", "2-prong candidates (rec unmatched);inv. mass (l^{#plus} l^{#minus}) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{200, 2., 4.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclengthSig", "2-prong candidates (rec matched);decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclengthBg", "2-prong candidates (rec unmatched);decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclengthxySig", "2-prong candidates (rec matched);decay length xy (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hdeclengthxyBg", "2-prong candidates (rec unmatched);decay length xy (cm);entries", {HistType::kTH2F, {{100, 0., 0.01}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0Sig", "2-prong candidates (rec matched);prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0Bg", "2-prong candidates (rec unmatched);prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1Sig", "2-prong candidates (rec matched);prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1Bg", "2-prong candidates (rec unmatched);prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0d0Sig", "2-prong candidates (rec matched);product of DCAxy to prim. vertex (cm^{2});entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0d0Bg", "2-prong candidates (rec unmatched);product of DCAxy to prim. vertex (cm^{2});entries", {HistType::kTH2F, {{400, -0.002, 0.002}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hChi2PCASig", "2-prong candidates (rec. matched);chi2 PCA (cm);entries", {HistType::kTH2F, {{1000, 0., 0.0001}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hChi2PCABg", "2-prong candidates (rec. unmatched);chi2 PCA (cm);entries", {HistType::kTH2F, {{1000, 0., 0.0001}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCtSig", "2-prong candidates (rec. matched);proper lifetime X(3872) * #it{c} (cm);entries", {HistType::kTH2F, {{400, 0., 0.001}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCtBg", "2-prong candidates (rec. unmatched);proper lifetime X(3872) * #it{c} (cm);entries", {HistType::kTH2F, {{400, 0., 0.001}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYGenSig", "2-prong candidates (rec. matched);candidate rapidity;entries", {HistType::kTH2F, {{10, -2., 2.}, {(std::vector<double>)bins, "#it{p}^{gen}_{T} (GeV/#it{c})"}}});
    registry.add("hYSig", "2-prong candidates (rec. matched);candidate rapidity;entries", {HistType::kTH2F, {{10, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYBg", "2-prong candidates (rec. unmatched);candidate rapidity;entries", {HistType::kTH2F, {{10, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYGen", "2-prong MC particles (gen. matched);candidate rapidity;entries", {HistType::kTH2F, {{10, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_jpsi::isSelJpsiToEE >= d_selectionFlagJpsi || aod::hf_selcandidate_jpsi::isSelJpsiToMuMu >= d_selectionFlagJpsi);

  void process(soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelJpsiCandidate, aod::HfCandProng2MCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC, aod::BigTracksMC const& tracks)
  {
    // MC rec.
    //Printf("MC Candidates: %d", candidates.size());
    int decaymode = DecayType::JpsiToEE;
    if (d_modeJpsiToMuMu)
      decaymode = DecayType::JpsiToMuMu;

    for (auto& candidate : candidates) {
      if (d_selectionFlagJpsi > 0) {
        if (!d_modeJpsiToMuMu) {
          if (candidate.isSelJpsiToEE() <= 0)
            continue;
        } else {
          if (candidate.isSelJpsiToMuMu() <= 0)
            continue;
        }
      }

      if (!(candidate.hfflag() & 1 << decaymode)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YJpsi(candidate)) > cutYCandMax) {
        continue;
      }
      if (candidate.flagMCMatchRec() == 1 << decaymode) {
        //Get the corresponding MC particle.
        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index0_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandProng2MCGen>>(), 443, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt()); // gen. level pT
        registry.fill(HIST("hPtRecSig"), candidate.pt());      // rec. level pT
        registry.fill(HIST("hCPARecSig"), candidate.cpa());
        registry.fill(HIST("hEtaRecSig"), candidate.eta());
        if (d_modeJpsiToMuMu) {
          registry.fill(HIST("hmassSig"), InvMassJpsiToMuMu(candidate), candidate.pt());
        } else {
          registry.fill(HIST("hmassSig"), InvMassJpsiToEE(candidate), candidate.pt());
        }
        registry.fill(HIST("hmassSig"), InvMassJpsiToEE(candidate), candidate.pt());
        registry.fill(HIST("hdeclengthSig"), candidate.decayLength(), candidate.pt());
        registry.fill(HIST("hdeclengthxySig"), candidate.decayLengthXY(), candidate.pt());
        registry.fill(HIST("hd0Prong0Sig"), candidate.impactParameter0(), candidate.pt());
        registry.fill(HIST("hd0Prong1Sig"), candidate.impactParameter1(), candidate.pt());
        registry.fill(HIST("hd0d0Sig"), candidate.impactParameterProduct(), candidate.pt());
        registry.fill(HIST("hChi2PCASig"), candidate.chi2PCA(), candidate.pt());
        registry.fill(HIST("hCtSig"), CtJpsi(candidate), candidate.pt());
        registry.fill(HIST("hYSig"), YJpsi(candidate), candidate.pt());
        registry.fill(HIST("hYGenSig"), RecoDecay::Y(array{particleMother.px(), particleMother.py(), particleMother.pz()}, RecoDecay::getMassPDG(particleMother.pdgCode())), particleMother.pt());

      } else {
        registry.fill(HIST("hPtRecBg"), candidate.pt());
        registry.fill(HIST("hCPARecBg"), candidate.cpa());
        registry.fill(HIST("hEtaRecBg"), candidate.eta());
        if (d_modeJpsiToMuMu) {
          registry.fill(HIST("hmassBg"), InvMassJpsiToMuMu(candidate), candidate.pt());
        } else {
          registry.fill(HIST("hmassBg"), InvMassJpsiToEE(candidate), candidate.pt());
        }
        registry.fill(HIST("hdeclengthBg"), candidate.decayLength(), candidate.pt());
        registry.fill(HIST("hdeclengthxyBg"), candidate.decayLengthXY(), candidate.pt());
        registry.fill(HIST("hd0Prong0Bg"), candidate.impactParameter0(), candidate.pt());
        registry.fill(HIST("hd0Prong1Bg"), candidate.impactParameter1(), candidate.pt());
        registry.fill(HIST("hd0d0Bg"), candidate.impactParameterProduct(), candidate.pt());
        registry.fill(HIST("hChi2PCABg"), candidate.chi2PCA(), candidate.pt());
        registry.fill(HIST("hCtBg"), CtJpsi(candidate), candidate.pt());
        registry.fill(HIST("hYBg"), YJpsi(candidate), candidate.pt());
      }
    }
    // MC gen.
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (particle.flagMCMatchGen() == 1 << decaymode) {
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode()))) > cutYCandMax) {
          continue;
        }
        registry.fill(HIST("hPtGen"), particle.pt());
        registry.fill(HIST("hEtaGen"), particle.eta());
        registry.fill(HIST("hYGen"), RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode())), particle.pt());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<TaskJpsi>(cfgc, TaskName{"hf-task-jpsi"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskJpsiMC>(cfgc, TaskName{"hf-task-jpsi-mc"}));
  }
  return workflow;
}
