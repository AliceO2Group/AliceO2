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

/// \file taskBplus.cxx
/// \brief B± analysis task
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN
/// \author Antonio Palasciano <antonio.palasciano@cern.ch>, Università degli Studi di Bari & INFN, Sezione di Bari
/// \author Deepa Thomas <deepa.thomas@cern.ch>, UT Austin

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::aod;
using namespace o2::analysis;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::aod::hf_cand_bplus;
using namespace o2::analysis::hf_cuts_bplus_tod0pi;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, true, {"Fill MC histograms."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// B± analysis task
struct HfTaskBplus {
  HistogramRegistry registry{
    "registry",
    {{"hPtProng0", "B+ candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{1000, 0., 50.}}}},
     {"hPtProng1", "B+ candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 10.}}}},
     {"hPtCand", "B+ candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{1000, 0., 50.}}}}}};

  Configurable<int> selectionFlagBPlus{"selectionFlagBPlus", 1, "Selection Flag for B+"};
  Configurable<double> cutYCandMax{"cutYCandMax", 0.8, "max. cand. rapidity"};
  Configurable<std::vector<double>> bins{"pTBins", std::vector<double>{hf_cuts_bplus_tod0pi::pTBins_v}, "pT bin limits"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMass", "B+ candidates;inv. mass #bar{D^{0}}#pi^{+} (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{500, 0., 10.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLength", "B+ candidates;decay length (cm);entries", {HistType::kTH2F, {{200, 0., 0.4}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthXY", "B+ candidates;decay length xy (cm);entries", {HistType::kTH2F, {{200, 0., 0.4}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0", "B+ candidates;prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{100, -0.05, 0.05}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1", "B+ candidates;prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{100, -0.05, 0.05}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPA", "B+ candidates;candidate cosine of pointing angle;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEta", "B+ candidates;candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hRapidity", "B+ candidates;candidate #it{y};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParErr", "B+ candidates;candidate impact parameter error (cm);entries", {HistType::kTH2F, {{100, -1., 1.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLenErr", "B+ candidates;candidate decay length error (cm);entries", {HistType::kTH2F, {{100, 0., 1.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLenXYErr", "B+ candidates;candidate decay length xy error (cm);entries", {HistType::kTH2F, {{100, 0., 1.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hIPProd", "B+ candidates;candidate impact parameter product;entries", {HistType::kTH2F, {{100, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hInvMassD0", "B+ candidates;prong0, D0 inv. mass (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{500, 0, 5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_bplus::isSelBPlusToD0Pi >= selectionFlagBPlus);

  void process(soa::Filtered<soa::Join<aod::HfCandBPlus, aod::HFSelBPlusToD0PiCandidate>> const& candidates, soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>, aod::BigTracksPID const& tracks)
  {
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << hf_cand_bplus::DecayType::BPlusToD0Pi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YBPlus(candidate)) > cutYCandMax) {
        continue;
      }

      auto D0cand = candidate.index0_as<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>>();
      auto pion = candidate.index1_as<aod::BigTracksPID>();

      registry.fill(HIST("hMass"), InvMassBPlus(candidate), candidate.pt());
      registry.fill(HIST("hPtCand"), candidate.pt());
      registry.fill(HIST("hPtProng0"), candidate.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate.ptProng1());
      registry.fill(HIST("hIPProd"), candidate.impactParameterProduct(), candidate.pt());
      registry.fill(HIST("hDecLength"), candidate.decayLength(), candidate.pt());
      registry.fill(HIST("hDecLengthXY"), candidate.decayLengthXY(), candidate.pt());
      registry.fill(HIST("hd0Prong0"), candidate.impactParameter0(), candidate.pt());
      registry.fill(HIST("hd0Prong1"), candidate.impactParameter1(), candidate.pt());
      registry.fill(HIST("hCPA"), candidate.cpa(), candidate.pt());
      registry.fill(HIST("hEta"), candidate.eta(), candidate.pt());
      registry.fill(HIST("hRapidity"), YBPlus(candidate), candidate.pt());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter0(), candidate.pt());
      registry.fill(HIST("hImpParErr"), candidate.errorImpactParameter1(), candidate.pt());
      registry.fill(HIST("hDecLenErr"), candidate.errorDecayLength(), candidate.pt());
      registry.fill(HIST("hDecLenXYErr"), candidate.errorDecayLengthXY(), candidate.pt());
      if (pion.sign() > 0) {
        registry.fill(HIST("hInvMassD0"), InvMassD0bar(D0cand), candidate.pt());
      } else {
        registry.fill(HIST("hInvMassD0"), InvMassD0(D0cand), candidate.pt());
      }
    } // candidate loop
  }   // process
};    // struct

/// BPlus MC analysis and fill histograms
struct HfTaskBplusMc {
  HistogramRegistry registry{
    "registry",
    {{"hPtRecSig", "B+ candidates (matched);candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{300, 0., 30.}}}},
     {"hPtRecBg", "B+ candidates (unmatched);candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{300, 0., 30.}}}},
     {"hPtGenSig", "B+ candidates (matched);candidate #it{p}_{T}^{gen.} (GeV/#it{c});entries", {HistType::kTH1F, {{300, 0., 10.}}}},
     {"hPtGen", "MC particles (matched);candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{300, 0., 30.}}}}}};

  Configurable<int> selectionFlagBPlus{"selectionFlagBPlus", 1, "Selection Flag for B+"};
  Configurable<double> cutYCandMax{"cutYCandMax", 0.8, "max. cand. rapidity"};
  Configurable<std::vector<double>> bins{"pTBins", std::vector<double>{hf_cuts_bplus_tod0pi::pTBins_v}, "pT bin limits"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hEtaGen", "MC particles (matched);candidate #it{#eta}^{gen};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYGen", "MC particles (matched);candidate #it{y}^{gen};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hPtGenProng0", "MC particles (matched);prong 0 #it{p}_{T}^{gen} (GeV/#it{c});entries", {HistType::kTH2F, {{100, 0., 10.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hPtGenProng1", "MC particles (matched);prong 1 #it{p}_{T}^{gen} (GeV/#it{c});entries", {HistType::kTH2F, {{100, 0., 10.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYGenProng0", "MC particles (matched);prong 0 #it{y}^{gen};entries", {HistType::kTH2F, {{100, -2, 2}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hYGenProng1", "MC particles (matched);prong 1 #it{y}^{gen};entries", {HistType::kTH2F, {{100, -2, 2}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEtaGenProng0", "MC particles (matched);prong 0 #it{#eta}^{gen};entries", {HistType::kTH2F, {{100, -2, 2}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEtaGenProng1", "MC particles (matched);prong 1 #it{#eta}^{gen};entries", {HistType::kTH2F, {{100, -2, 2}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPARecSig", "B+ candidates (matched);candidate cosine of pointing angle;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPARecBg", "B+ candidates (unmatched);candidate cosine of pointing angle;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPAxyRecSig", "B+ candidates (matched);candidate CPAxy;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPAxyRecBg", "B+ candidates (unmatched);candidate CPAxy;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPARecSigD0", "B+ candidates (matched);prong 0 cosine of pointing angle;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hCPARecBgD0", "B+ candidates (unmatched);prong 0 cosine of pointing angle;entries", {HistType::kTH2F, {{110, -1.1, 1.1}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEtaRecSig", "B+ candidates (matched);candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hEtaRecBg", "B+ candidates (unmatched);candidate #it{#eta};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hRapidityRecSig", "B+ candidates (matched);candidate #it{y};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hRapidityRecBg", "B+ candidates (unmatched);candidate #it{#y};entries", {HistType::kTH2F, {{100, -2., 2.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});

    registry.add("hPtProng0RecSig", "B+ candidates (matched);prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH2F, {{100, 0., 10.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hPtProng1RecSig", "B+ candidates (matched);prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH2F, {{100, 0., 10.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hPtProng0RecBg", "B+ candidates (unmatched);prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH2F, {{100, 0., 10.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hPtProng1RecBg", "B+ candidates (unmatched);prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH2F, {{100, 0., 10.}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassRecSig", "B+ candidates (matched);inv. mass #bar{D^{0}}#pi^{+} (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{300, 4.0, 7.00}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassRecBg", "B+ candidates (unmatched);inv. mass #bar{D^{0}}#pi^{+} (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{300, 4.0, 7.0}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0RecSig", "B+ candidates (matched);prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, -0.05, 0.05}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1RecSig", "B+ candidates (matched);prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, -0.05, 0.05}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong0RecBg", "B+ candidates (unmatched);prong 0 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, -0.05, 0.05}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hd0Prong1RecBg", "B+ candidates (unmatched);prong 1 DCAxy to prim. vertex (cm);entries", {HistType::kTH2F, {{200, -0.05, 0.05}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthRecSig", "B+ candidates (matched);candidate decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthXYRecSig", "B+ candidates (matched);candidate decay length xy (cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthRecBg", "B+ candidates (unmatched);candidate decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthXYRecBg", "B+ candidates (unmatched);candidate decay length xy(cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthRecD0Sig", "B+ candidates (matched);candidate decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthRecD0Bg", "B+ candidates (unmatched);candidate decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthNormRecSig", "B+ candidates (matched);candidate decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hDecLengthNormRecBg", "B+ candidates (unmatched);candidate decay length (cm);entries", {HistType::kTH2F, {{100, 0., 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParProdBPlusRecSig", "B+ candidates (matched);candidate impact parameter product ;entries", {HistType::kTH2F, {{100, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hImpParProdBPlusRecBg", "B+ candidates (unmatched);candidate impact parameter product ;entries", {HistType::kTH2F, {{100, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_bplus::isSelBPlusToD0Pi >= selectionFlagBPlus);

  void process(soa::Filtered<soa::Join<aod::HfCandBPlus, aod::HFSelBPlusToD0PiCandidate, aod::HfCandBPMCRec>> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandBPMCGen> const& particlesMC, aod::BigTracksMC const& tracks, aod::HfCandProng2 const&)
  {
    //MC rec
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << hf_cand_bplus::DecayType::BPlusToD0Pi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YBPlus(candidate)) > cutYCandMax) {
        continue;
      }
      auto candD0 = candidate.index0_as<aod::HfCandProng2>();
      if (std::abs(candidate.flagMCMatchRec()) == 1 << hf_cand_bplus::DecayType::BPlusToD0Pi) {

        auto indexMother = RecoDecay::getMother(particlesMC, candidate.index1_as<aod::BigTracksMC>().mcParticle_as<soa::Join<aod::McParticles, aod::HfCandBPMCGen>>(), pdg::Code::kBPlus, true);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        registry.fill(HIST("hPtGenSig"), particleMother.pt());
        registry.fill(HIST("hPtRecSig"), candidate.pt());
        registry.fill(HIST("hCPARecSig"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("hCPAxyRecSig"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("hEtaRecSig"), candidate.eta(), candidate.pt());
        registry.fill(HIST("hRapidityRecSig"), YBPlus(candidate), candidate.pt());
        registry.fill(HIST("hDecLengthRecSig"), candidate.decayLength(), candidate.pt());
        registry.fill(HIST("hDecLengthXYRecSig"), candidate.decayLengthXY(), candidate.pt());
        registry.fill(HIST("hMassRecSig"), InvMassBPlus(candidate), candidate.pt());
        registry.fill(HIST("hd0Prong0RecSig"), candidate.impactParameter0(), candidate.pt());
        registry.fill(HIST("hd0Prong1RecSig"), candidate.impactParameter1(), candidate.pt());
        registry.fill(HIST("hPtProng0RecSig"), candidate.ptProng0(), candidate.pt());
        registry.fill(HIST("hPtProng1RecSig"), candidate.ptProng1(), candidate.pt());
        registry.fill(HIST("hImpParProdBPlusRecSig"), candidate.impactParameterProduct(), candidate.pt());
        registry.fill(HIST("hDecLengthNormRecSig"), candidate.decayLengthXYNormalised(), candidate.pt());
        registry.fill(HIST("hCPARecSigD0"), candD0.cpa(), candidate.pt());
        registry.fill(HIST("hDecLengthRecD0Sig"), candD0.decayLength(), candidate.pt());
      } else {
        registry.fill(HIST("hPtRecBg"), candidate.pt());
        registry.fill(HIST("hCPARecBg"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("hCPAxyRecBg"), candidate.cpa(), candidate.pt());
        registry.fill(HIST("hEtaRecBg"), candidate.eta(), candidate.pt());
        registry.fill(HIST("hRapidityRecBg"), YBPlus(candidate), candidate.pt());
        registry.fill(HIST("hDecLengthRecBg"), candidate.decayLength(), candidate.pt());
        registry.fill(HIST("hDecLengthXYRecBg"), candidate.decayLengthXY(), candidate.pt());
        registry.fill(HIST("hMassRecBg"), InvMassBPlus(candidate), candidate.pt());
        registry.fill(HIST("hd0Prong0RecBg"), candidate.impactParameter0(), candidate.pt());
        registry.fill(HIST("hd0Prong1RecBg"), candidate.impactParameter1(), candidate.pt());
        registry.fill(HIST("hPtProng0RecBg"), candidate.ptProng0(), candidate.pt());
        registry.fill(HIST("hPtProng1RecBg"), candidate.ptProng1(), candidate.pt());
        registry.fill(HIST("hImpParProdBPlusRecBg"), candidate.impactParameterProduct(), candidate.pt());
        registry.fill(HIST("hDecLengthNormRecBg"), candidate.decayLengthXYNormalised(), candidate.pt());
        registry.fill(HIST("hCPARecBgD0"), candD0.cpa(), candidate.pt());
        registry.fill(HIST("hDecLengthRecD0Bg"), candD0.decayLength(), candidate.pt());
      }
    } // rec

    // MC gen. level
    //Printf("MC Particles: %d", particlesMC.size());
    for (auto& particle : particlesMC) {
      if (std::abs(particle.flagMCMatchGen()) == 1 << hf_cand_bplus::DecayType::BPlusToD0Pi) {

        auto yParticle = RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(pdg::Code::kBPlus));
        if (cutYCandMax >= 0. && std::abs(yParticle) > cutYCandMax) {
          continue;
        }

        float ptProngs[2], yProngs[2], etaProngs[2];
        for (int iD = particle.daughter0Id(), counter = 0; iD <= particle.daughter1Id(); ++iD, counter++) {
          auto daught = particlesMC.iteratorAt(iD);
          ptProngs[counter] = daught.pt();
          etaProngs[counter] = daught.eta();
          yProngs[counter] = RecoDecay::Y(array{daught.px(), daught.py(), daught.pz()}, RecoDecay::getMassPDG(daught.pdgCode()));
        }

        registry.fill(HIST("hPtGenProng0"), ptProngs[0], particle.pt());
        registry.fill(HIST("hPtGenProng1"), ptProngs[1], particle.pt());
        registry.fill(HIST("hYGenProng0"), yProngs[0], particle.pt());
        registry.fill(HIST("hYGenProng1"), yProngs[1], particle.pt());
        registry.fill(HIST("hEtaGenProng0"), etaProngs[0], particle.pt());
        registry.fill(HIST("hEtaGenProng1"), etaProngs[1], particle.pt());

        //  if (cutYCandMax >= 0. && (std::abs(yProngs[0]) > cutYCandMax || std::abs(yProngs[1]) > cutYCandMax))
        //    continue;

        registry.fill(HIST("hPtGen"), particle.pt());
        registry.fill(HIST("hYGen"), yParticle, particle.pt());
        registry.fill(HIST("hEtaGen"), particle.eta(), particle.pt());
      }
    } //gen
  }   // process
};    // struct

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};
  const bool doMC = cfgc.options().get<bool>("doMC");
  workflow.push_back(adaptAnalysisTask<HfTaskBplus>(cfgc));
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<HfTaskBplusMc>(cfgc));
  }
  return workflow;
}
