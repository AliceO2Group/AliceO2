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

/// \file HFSelOptimisation.cxx
/// \brief task to study preselections
///
/// \author Fabrizio Grosa <fabrizio.grosa@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand;

namespace
{

static constexpr int nCutsToTestCosp = 15;
static constexpr int nCutsToTestDecLen = 11;
static constexpr int nCutsToTestImpParProd = 11;
static constexpr int nCutsToTestMinDCAxy = 9;
static constexpr int nCutsToTestMinTrackPt = 7;

constexpr float cutsCosp[nCutsToTestCosp] = {0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995};
constexpr float cutsDecLen[nCutsToTestDecLen] = {0., 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1};
constexpr float cutsImpParProd[nCutsToTestImpParProd] = {-0.00005, -0.00004, -0.00003, -0.00002, -0.00001, 0., 0.00001, 0.00002, 0.00003, 0.00004, 0.00005};
constexpr float cutsMinDCAxy[nCutsToTestMinDCAxy] = {0., 0.0005, 0.001, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050};
constexpr float cutsMinTrackPt[nCutsToTestMinTrackPt] = {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60};

auto vecCutsCosp = std::vector<float>{cutsCosp, cutsCosp + nCutsToTestCosp};
auto vecCutsDecLen = std::vector<float>{cutsDecLen, cutsDecLen + nCutsToTestDecLen};
auto vecCutsImpParProd = std::vector<float>{cutsImpParProd, cutsImpParProd + nCutsToTestImpParProd};
auto vecCutsMinDCAxy = std::vector<float>{cutsMinDCAxy, cutsMinDCAxy + nCutsToTestMinDCAxy};
auto vecCutsMinTrackPt = std::vector<float>{cutsMinTrackPt, cutsMinTrackPt + nCutsToTestMinTrackPt};

static const int n2Prong = o2::aod::hf_cand_prong2::DecayType::N2ProngDecays;
static const int n3Prong = o2::aod::hf_cand_prong3::DecayType::N3ProngDecays;

static constexpr std::array<std::array<std::string_view, n2Prong + 1>, 3> histoNames2Prong = {{{"hPromptVsPtD0ToPiK", "hPromptVsPtJpsiToEE", "hPromptVsPt2Prong"},
                                                                                               {"hNonPromptVsPtD0ToPiK", "hNonPromptVsPtJpsiToEE", "hNonPromptVsPt2Prong"},
                                                                                               {"hBkgVsPtD0ToPiK", "hBkgVsPtJpsiToEE", "hBkgVsPt2Prong"}}};
static constexpr std::array<std::array<std::string_view, n2Prong + 1>, 3> histoNamesCosp2Prong = {{{"hPromptCospVsPtD0ToPiK", "hPromptCospVsPtJpsiToEE", "hPromptCospVsPt2Prong"},
                                                                                                   {"hNonPromptCospVsPtD0ToPiK", "hNonPromptCospVsPtJpsiToEE", "hNonPromptCospVsPt2Prong"},
                                                                                                   {"hBkgCospVsPtD0ToPiK", "hBkgCospVsPtJpsiToEE", "hBkgCospVsPt2Prong"}}};
static constexpr std::array<std::array<std::string_view, n2Prong + 1>, 3> histoNamesDecLen2Prong = {{{"hPromptDecLenVsPtD0ToPiK", "hPromptDecLenVsPtJpsiToEE", "hPromptDecLenVsPt2Prong"},
                                                                                                     {"hNonPromptDecLenVsPtD0ToPiK", "hNonPromptDecLenVsPtJpsiToEE", "hNonPromptDecLenVsPt2Prong"},
                                                                                                     {"hBkgDecLenVsPtD0ToPiK", "hBkgDecLenVsPtJpsiToEE", "hBkgDecLenVsPt2Prong"}}};
static constexpr std::array<std::array<std::string_view, n2Prong + 1>, 3> histoNamesImpParProd2Prong = {{{"hPromptImpParProdVsPtD0ToPiK", "hPromptImpParProdVsPtJpsiToEE", "hPromptImpParProdVsPt2Prong"},
                                                                                                         {"hNonPromptImpParProdVsPtD0ToPiK", "hNonPromptImpParProdVsPtJpsiToEE", "hNonPromptImpParProdVsPt2Prong"},
                                                                                                         {"hBkgImpParProdVsPtD0ToPiK", "hBkgImpParProdVsPtJpsiToEE", "hBkgImpParProdVsPt2Prong"}}};
static constexpr std::array<std::array<std::string_view, n2Prong + 1>, 3> histoNamesMinDCAxy2Prong = {{{"hPromptMinDCAxyVsPtD0ToPiK", "hPromptMinDCAxyVsPtJpsiToEE", "hPromptMinDCAxyVsPt2Prong"},
                                                                                                       {"hNonPromptMinDCAxyVsPtD0ToPiK", "hNonPromptMinDCAxyVsPtJpsiToEE", "hNonPromptMinDCAxyVsPt2Prong"},
                                                                                                       {"hBkgMinDCAxyVsPtD0ToPiK", "hBkgMinDCAxyVsPtJpsiToEE", "hBkgMinDCAxyVsPt2Prong"}}};
static constexpr std::array<std::array<std::string_view, n2Prong + 1>, 3> histoNamesMinTrackPt2Prong = {{{"hPromptMinTrackPtVsPtD0ToPiK", "hPromptMinTrackPtVsPtJpsiToEE", "hPromptMinTrackPtVsPt2Prong"},
                                                                                                         {"hNonPromptMinTrackPtVsPtD0ToPiK", "hNonPromptMinTrackPtVsPtJpsiToEE", "hNonPromptMinTrackPtVsPt2Prong"},
                                                                                                         {"hBkgMinTrackPtVsPtD0ToPiK", "hBkgMinTrackPtVsPtJpsiToEE", "hBkgMinTrackPtVsPt2Prong"}}};

static constexpr std::array<std::array<std::string_view, n3Prong + 1>, 3> histoNames3Prong = {{{"hPromptVsPtDPlusToPiKPi", "hPromptVsPtLcToPKPi", "hPromptVsPtDsToPiKK", "hPromptVsPtXicToPKPi", "hPromptVsPt3Prong"},
                                                                                               {"hNonPromptVsPtDPlusToPiKPi", "hNonPromptVsPtLcToPKPi", "hNonPromptVsPtDsToPiKK", "hNonPromptVsPtXicToPKPi", "hNonPromptVsPt3Prong"},
                                                                                               {"hBkgVsPtDPlusToPiKPi", "hBkgVsPtLcToPKPi", "hBkgVsPtDsToPiKK", "hBkgVsPtXicToPKPi", "hBkgVsPt3Prong"}}};
static constexpr std::array<std::array<std::string_view, n3Prong + 1>, 3> histoNamesCosp3Prong = {{{"hPromptCospVsPtDPlusToPiKPi", "hPromptCospVsPtLcToPKPi", "hPromptCospVsPtDsToPiKK", "hPromptCospVsPtXicToPKPi", "hPromptCospVsPt3Prong"},
                                                                                                   {"hNonPromptCospVsPtDPlusToPiKPi", "hNonPromptCospVsPtLcToPKPi", "hNonPromptCospVsPtDsToPiKK", "hNonPromptCospVsPtXicToPKPi", "hNonPromptCospVsPt3Prong"},
                                                                                                   {"hBkgCospVsPtDPlusToPiKPi", "hBkgCospVsPtLcToPKPi", "hBkgCospVsPtDsToPiKK", "hBkgCospVsPtXicToPKPi", "hBkgCospVsPt3Prong"}}};
static constexpr std::array<std::array<std::string_view, n3Prong + 1>, 3> histoNamesDecLen3Prong = {{{"hPromptDecLenVsPtDPlusToPiKPi", "hPromptDecLenVsPtLcToPKPi", "hPromptDecLenVsPtDsToPiKK", "hPromptDecLenVsPtXicToPKPi", "hPromptDecLenVsPt3Prong"},
                                                                                                     {"hNonPromptDecLenVsPtDPlusToPiKPi", "hNonPromptDecLenVsPtLcToPKPi", "hNonPromptDecLenVsPtDsToPiKK", "hNonPromptDecLenVsPtXicToPKPi", "hNonPromptDecLenVsPt3Prong"},
                                                                                                     {"hBkgDecLenVsPtDPlusToPiKPi", "hBkgDecLenVsPtLcToPKPi", "hBkgDecLenVsPtDsToPiKK", "hBkgDecLenVsPtXicToPKPi", "hBkgDecLenVsPt3Prong"}}};
static constexpr std::array<std::array<std::string_view, n3Prong + 1>, 3> histoNamesMinDCAxy3Prong = {{{"hPromptMinDCAxyVsPtDPlusToPiKPi", "hPromptMinDCAxyVsPtLcToPKPi", "hPromptMinDCAxyVsPtDsToPiKK", "hPromptMinDCAxyVsPtXicToPKPi", "hPromptMinDCAxyVsPt3Prong"},
                                                                                                       {"hNonPromptMinDCAxyVsPtDPlusToPiKPi", "hNonPromptMinDCAxyVsPtLcToPKPi", "hNonPromptMinDCAxyVsPtDsToPiKK", "hNonPromptMinDCAxyVsPtXicToPKPi", "hNonPromptMinDCAxyVsPt3Prong"},
                                                                                                       {"hBkgMinDCAxyVsPtDPlusToPiKPi", "hBkgMinDCAxyVsPtLcToPKPi", "hBkgMinDCAxyVsPtDsToPiKK", "hBkgMinDCAxyVsPtXicToPKPi", "hBkgMinDCAxyVsPt3Prong"}}};
static constexpr std::array<std::array<std::string_view, n3Prong + 1>, 3> histoNamesMinTrackPt3Prong = {{{"hPromptMinTrackPtVsPtDPlusToPiKPi", "hPromptMinTrackPtVsPtLcToPKPi", "hPromptMinTrackPtVsPtDsToPiKK", "hPromptMinTrackPtVsPtXicToPKPi", "hPromptMinTrackPtVsPt3Prong"},
                                                                                                         {"hNonPromptMinTrackPtVsPtDPlusToPiKPi", "hNonPromptMinTrackPtVsPtLcToPKPi", "hNonPromptMinTrackPtVsPtDsToPiKK", "hNonPromptMinTrackPtVsPtXicToPKPi", "hNonPromptMinTrackPtVsPt3Prong"},
                                                                                                         {"hBkgMinTrackPtVsPtDPlusToPiKPi", "hBkgMinTrackPtVsPtLcToPKPi", "hBkgMinTrackPtVsPtDsToPiKK", "hBkgMinTrackPtVsPtXicToPKPi", "hBkgMinTrackPtVsPt3Prong"}}};
} // namespace

struct HfSelOptimisation {

  Configurable<std::vector<float>> cutsToTestCosp{"cutsToTestCosp", std::vector<float>{vecCutsCosp}, "cos(theta_P) cut values to test"};
  Configurable<std::vector<float>> cutsToTestDecLen{"cutsToTestDecLen", std::vector<float>{vecCutsDecLen}, "decay length cut values to test"};
  Configurable<std::vector<float>> cutsToTestImpParProd{"cutsToTestImpParProd", std::vector<float>{vecCutsImpParProd}, "impact parameter product cut values to test (2-prongs only)"};
  Configurable<std::vector<float>> cutsToTestMinDCAxy{"cutsToTestMinDCAxy", std::vector<float>{vecCutsMinDCAxy}, "min DCA xy cut values to test"};
  Configurable<std::vector<float>> cutsToTestMinTrackPt{"cutsToTestMinTrackPt", std::vector<float>{vecCutsMinTrackPt}, "min track pT cut values to test"};

  ConfigurableAxis ptBinning{"ptBinning", {0, 0., 2., 5., 20.}, "pT bin limits"};

  AxisSpec axisPt = {ptBinning, "#it{p}_{T} (GeV/#it{c})"};
  // quantized axes
  AxisSpec axisCosp = {cutsToTestCosp->size(), 0.5, cutsToTestCosp->size() + 0.5, "cos(#theta_{P}) >"};
  AxisSpec axisDecLen = {cutsToTestDecLen->size(), 0.5, cutsToTestDecLen->size() + 0.5, "decay length (cm) >"};
  AxisSpec axisImpParProd = {cutsToTestImpParProd->size(), 0.5, cutsToTestImpParProd->size() + 0.5, "#it{d}_{0}#times#it{d}_{0} (cm^{2}) <"};
  AxisSpec axisMinDCAxy = {cutsToTestMinDCAxy->size(), 0.5, cutsToTestMinDCAxy->size() + 0.5, "min track #it{d}_{0} (cm) >"};
  AxisSpec axisMinTrackPt = {cutsToTestMinTrackPt->size(), 0.5, cutsToTestMinTrackPt->size() + 0.5, "min track #it{p}_{T} (cm) >"};

  HistogramRegistry registry{"registry", {}};

  void init(InitContext const&)
  {
    for (int iOrig{0}; iOrig < 3; iOrig++) {
      for (int i2Prong = 0; i2Prong < n2Prong + 1; ++i2Prong) {
        registry.add(histoNames2Prong[iOrig][i2Prong].data(), "", HistType::kTH1F, {axisPt});
        auto histCosp2Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesCosp2Prong[iOrig][i2Prong].data(), "", HistType::kTH2F, {axisPt, axisCosp}));
        for(int iBin{0}; iBin<histCosp2Prong->GetYaxis()->GetNbins(); ++iBin) {
          histCosp2Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.4f", cutsToTestCosp[iBin]));
        }
        auto histDecLen2Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesDecLen2Prong[iOrig][i2Prong].data(), "", HistType::kTH2F, {axisPt, axisDecLen}));
        for(int iBin{0}; iBin<histDecLen2Prong->GetYaxis()->GetNbins(); ++iBin) {
          histDecLen2Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.3f", cutsToTestDecLen[iBin]));
        }
        auto histImpParProd2Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesImpParProd2Prong[iOrig][i2Prong].data(), "", HistType::kTH2F, {axisPt, axisImpParProd}));
        for(int iBin{0}; iBin<histImpParProd2Prong->GetYaxis()->GetNbins(); ++iBin) {
          histImpParProd2Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.4f", cutsToTestImpParProd[iBin]));
        }
        auto histMinDCAxy2Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesMinDCAxy2Prong[iOrig][i2Prong].data(), "", HistType::kTH2F, {axisPt, axisMinDCAxy}));
        for(int iBin{0}; iBin<histMinDCAxy2Prong->GetYaxis()->GetNbins(); ++iBin) {
          histMinDCAxy2Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.4f", cutsToTestMinDCAxy[iBin]));
        }
        auto histMinTrackPt2Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesMinTrackPt2Prong[iOrig][i2Prong].data(), "", HistType::kTH2F, {axisPt, axisMinTrackPt}));
        for(int iBin{0}; iBin<histMinTrackPt2Prong->GetYaxis()->GetNbins(); ++iBin) {
          histMinTrackPt2Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.2f", cutsToTestMinTrackPt[iBin]));
        }
      }
      for (int i3Prong{0}; i3Prong < n3Prong + 1; ++i3Prong) {
        registry.add(histoNames3Prong[iOrig][i3Prong].data(), "", HistType::kTH1F, {axisPt});
        auto histCosp3Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesCosp3Prong[iOrig][i3Prong].data(), "", HistType::kTH2F, {axisPt, axisCosp}));
        for(int iBin{0}; iBin<histCosp3Prong->GetYaxis()->GetNbins(); ++iBin) {
          histCosp3Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.4f", cutsToTestCosp[iBin]));
        }
        auto histDecayLen3Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesDecLen3Prong[iOrig][i3Prong].data(), "", HistType::kTH2F, {axisPt, axisDecLen}));
        for(int iBin{0}; iBin<histDecayLen3Prong->GetYaxis()->GetNbins(); ++iBin) {
          histDecayLen3Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.4f", cutsToTestDecLen[iBin]));
        }
        auto histMinDCAxy3Prong = std::get<std::shared_ptr<TH2>>(registry.add(histoNamesMinDCAxy3Prong[iOrig][i3Prong].data(), "", HistType::kTH2F, {axisPt, axisMinDCAxy}));
        for(int iBin{0}; iBin<histMinDCAxy3Prong->GetYaxis()->GetNbins(); ++iBin) {
          histMinDCAxy3Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.4f", cutsToTestMinDCAxy[iBin]));
        }
        auto histMinTrackPt3Prong = std::get<std::shared_ptr<TH2>>(egistry.add(histoNamesMinTrackPt3Prong[iOrig][i3Prong].data(), "", HistType::kTH2F, {axisPt, axisMinTrackPt}));
        for(int iBin{0}; iBin<histMinTrackPt3Prong->GetYaxis()->GetNbins(); ++iBin) {
          histMinTrackPt3Prong->GetYaxis()->SetBinLabel(iBin+1, Form("%0.4f", cutsToTestMinTrackPt[iBin]));
        }
      }
    }
  }

  /// Conjugate-dependent topological cuts
  /// \param candidate is a candidate
  /// \param channel is the candidate channel
  /// \param candType is candidate type (Prompt, NonPrompt, Bkg)
  template <std::size_t candType, std::size_t candOrig, typename T1, typename T2>
  void testSelections2Prong(const T1& candidate, const T2& tracks)
  {
    auto pT = candidate.pt();
    std::array<double, 2> absDCA{std::abs(tracks[0].dcaPrim0()), std::abs(tracks[1].dcaPrim0())};
    std::sort(absDCA.begin(), absDCA.end());

    std::array<double, 2> ptTrack{tracks[0].pt(), tracks[1].pt()};
    std::sort(ptTrack.begin(), ptTrack.end());

    registry.get<TH1>(HIST(histoNames2Prong[candOrig][candType].data()))->Fill(pT);

    for (int iCospCut{0}; iCospCut < cutsToTestCosp->size(); ++iCospCut) {
      if (candidate.cpa() > cutsToTestCosp[iCospCut]) {
        registry.get<TH2>(HIST(histoNamesCosp2Prong[candOrig][candType].data()))->Fill(pT, iCospCut + 1);
      }
    }

    for (int iDecLenCut{0}; iDecLenCut < cutsToTestDecLen->size(); ++iDecLenCut) {
      if (candidate.decayLength() > cutsToTestDecLen[iDecLenCut]) {
        registry.get<TH2>(HIST(histoNamesDecLen2Prong[candOrig][candType].data()))->Fill(pT, iDecLenCut + 1);
      }
    }

    for (int iImpParProd{0}; iImpParProd < cutsToTestImpParProd->size(); ++iImpParProd) {
      if (candidate.impactParameterProduct() < cutsToTestImpParProd[iImpParProd]) {
        registry.get<TH2>(HIST(histoNamesImpParProd2Prong[candOrig][candType].data()))->Fill(pT, iImpParProd + 1);
      }
    }

    for (int iMinDCAxy{0}; iMinDCAxy < cutsToTestMinDCAxy->size(); ++iMinDCAxy) {
      if (absDCA[0] > cutsToTestMinDCAxy[iMinDCAxy]) {
        registry.get<TH2>(HIST(histoNamesMinDCAxy2Prong[candOrig][candType].data()))->Fill(pT, iMinDCAxy + 1);
      }
    }

    for (int iMinTrackPt{0}; iMinTrackPt < cutsToTestMinTrackPt->size(); ++iMinTrackPt) {
      if (ptTrack[0] > cutsToTestMinTrackPt[iMinTrackPt]) {
        registry.get<TH2>(HIST(histoNamesMinTrackPt2Prong[candOrig][candType].data()))->Fill(pT, iMinTrackPt + 1);
      }
    }
  }

  /// Conjugate-dependent topological cuts
  /// \param candidate is a candidate
  /// \param channel is the candidate channel
  /// \param candType is candidate type (Prompt, NonPrompt, Bkg)
  template <std::size_t candType, std::size_t candOrig, typename T1, typename T2>
  void testSelections3Prong(const T1& candidate, const T2& tracks)
  {
    auto pT = candidate.pt();
    std::array<double, 3> absDCA{std::abs(tracks[0].dcaPrim0()), std::abs(tracks[1].dcaPrim0()), std::abs(tracks[2].dcaPrim0())};
    std::sort(absDCA.begin(), absDCA.end());

    std::array<double, 3> ptTrack{tracks[0].pt(), tracks[1].pt(), tracks[2].pt()};
    std::sort(ptTrack.begin(), ptTrack.end());

    registry.get<TH1>(HIST(histoNames3Prong[candOrig][candType].data()))->Fill(pT);

    for (int iCospCut{0}; iCospCut < cutsToTestCosp->size(); ++iCospCut) {
      if (candidate.cpa() > cutsToTestCosp[iCospCut]) {
        registry.get<TH2>(HIST(histoNamesCosp3Prong[candOrig][candType].data()))->Fill(pT, iCospCut + 1);
      }
    }

    for (int iDecLenCut{0}; iDecLenCut < cutsToTestDecLen->size(); ++iDecLenCut) {
      if (candidate.decayLength() > cutsToTestDecLen[iDecLenCut]) {
        registry.get<TH2>(HIST(histoNamesDecLen3Prong[candOrig][candType].data()))->Fill(pT, iDecLenCut + 1);
      }
    }

    for (int iMinDCAxy{0}; iMinDCAxy < cutsToTestMinDCAxy->size(); ++iMinDCAxy) {
      if (absDCA[0] > cutsToTestMinDCAxy[iMinDCAxy]) {
        registry.get<TH2>(HIST(histoNamesMinDCAxy3Prong[candOrig][candType].data()))->Fill(pT, iMinDCAxy + 1);
      }
    }

    for (int iMinTrackPt{0}; iMinTrackPt < cutsToTestMinTrackPt->size(); ++iMinTrackPt) {
      if (ptTrack[0] > cutsToTestMinTrackPt[iMinTrackPt]) {
        registry.get<TH2>(HIST(histoNamesMinTrackPt3Prong[candOrig][candType].data()))->Fill(pT, iMinTrackPt + 1);
      }
    }
  }

  void process(soa::Join<aod::HfCandProng2, aod::HfCandProng2MCRec> const& cand2Prongs,
               soa::Join<aod::HfCandProng3, aod::HfCandProng3MCRec> const& cand3Prongs,
               aod::BigTracks const&)
  {
    // looping over 2-prong candidates
    for (const auto& cand2Prong : cand2Prongs) {

      auto trackPos = cand2Prong.index0_as<aod::BigTracks>(); // positive daughter
      auto trackNeg = cand2Prong.index1_as<aod::BigTracks>(); // negative daughter
      std::array tracks = {trackPos, trackNeg};

      bool isPrompt = false, isNonPrompt = false, isBkg = false;
      for (int iDecay{0}; iDecay < n2Prong; ++iDecay) {
        if (TESTBIT(cand2Prong.hfflag(), iDecay)) {
          if (std::abs(cand2Prong.flagMCMatchRec()) == BIT(iDecay)) {
            if (cand2Prong.originMCRec() == OriginType::Prompt) {
              isPrompt = true;
              switch (iDecay) {
                case o2::aod::hf_cand_prong2::DecayType::D0ToPiK:
                  testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::D0ToPiK, 0>(cand2Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong2::DecayType::JpsiToEE:
                  testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::JpsiToEE, 0>(cand2Prong, tracks);
                  break;
              }
            } else if (cand2Prong.originMCRec() == OriginType::NonPrompt) {
              isNonPrompt = true;
              switch (iDecay) {
                case o2::aod::hf_cand_prong2::DecayType::D0ToPiK:
                  testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::D0ToPiK, 1>(cand2Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong2::DecayType::JpsiToEE:
                  testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::JpsiToEE, 1>(cand2Prong, tracks);
                  break;
              }
            }
          } else {
            isBkg = true;
            switch (iDecay) {
              case o2::aod::hf_cand_prong2::DecayType::D0ToPiK:
                testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::D0ToPiK, 2>(cand2Prong, tracks);
                break;
              case o2::aod::hf_cand_prong2::DecayType::JpsiToEE:
                testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::JpsiToEE, 2>(cand2Prong, tracks);
                break;
            }
          }
        }
      }

      if (isPrompt) {
        testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::N2ProngDecays, 0>(cand2Prong, tracks);
      } else if (isNonPrompt) {
        testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::N2ProngDecays, 1>(cand2Prong, tracks);
      } else if (isBkg) {
        testSelections2Prong<o2::aod::hf_cand_prong2::DecayType::N2ProngDecays, 2>(cand2Prong, tracks);
      }
    } // loop over 2-prong candidates

    // looping over 3-prong candidates
    for (const auto& cand3Prong : cand3Prongs) {

      auto trackFirst = cand3Prong.index0_as<aod::BigTracks>();  // first daughter
      auto trackSecond = cand3Prong.index1_as<aod::BigTracks>(); // second daughter
      auto trackThird = cand3Prong.index2_as<aod::BigTracks>();  // third daughter
      std::array tracks = {trackFirst, trackSecond, trackThird};

      bool isPrompt = false, isNonPrompt = false, isBkg = false;
      for (int iDecay{0}; iDecay < n3Prong; ++iDecay) {
        if (TESTBIT(cand3Prong.hfflag(), iDecay)) {
          if (std::abs(cand3Prong.flagMCMatchRec()) == BIT(iDecay)) {
            if (cand3Prong.originMCRec() == OriginType::Prompt) {
              isPrompt = true;
              switch (iDecay) {
                case o2::aod::hf_cand_prong3::DecayType::DPlusToPiKPi:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::DPlusToPiKPi, 0>(cand3Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong3::DecayType::LcToPKPi:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::LcToPKPi, 0>(cand3Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong3::DecayType::DsToPiKK:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::DsToPiKK, 0>(cand3Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong3::DecayType::XicToPKPi:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::XicToPKPi, 0>(cand3Prong, tracks);
                  break;
              }
            } else if (cand3Prong.originMCRec() == OriginType::NonPrompt) {
              isNonPrompt = true;
              switch (iDecay) {
                case o2::aod::hf_cand_prong3::DecayType::DPlusToPiKPi:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::DPlusToPiKPi, 1>(cand3Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong3::DecayType::LcToPKPi:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::LcToPKPi, 1>(cand3Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong3::DecayType::DsToPiKK:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::DsToPiKK, 1>(cand3Prong, tracks);
                  break;
                case o2::aod::hf_cand_prong3::DecayType::XicToPKPi:
                  testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::XicToPKPi, 1>(cand3Prong, tracks);
                  break;
              }
            }
          } else {
            isBkg = true;
            switch (iDecay) {
              case o2::aod::hf_cand_prong3::DecayType::DPlusToPiKPi:
                testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::DPlusToPiKPi, 2>(cand3Prong, tracks);
                break;
              case o2::aod::hf_cand_prong3::DecayType::LcToPKPi:
                testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::LcToPKPi, 2>(cand3Prong, tracks);
                break;
              case o2::aod::hf_cand_prong3::DecayType::DsToPiKK:
                testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::DsToPiKK, 2>(cand3Prong, tracks);
                break;
              case o2::aod::hf_cand_prong3::DecayType::XicToPKPi:
                testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::XicToPKPi, 2>(cand3Prong, tracks);
                break;
            }
          }
        }
      }
      if (isPrompt) {
        testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::N3ProngDecays, 0>(cand3Prong, tracks);
      } else if (isNonPrompt) {
        testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::N3ProngDecays, 1>(cand3Prong, tracks);
      } else if (isBkg) {
        testSelections3Prong<o2::aod::hf_cand_prong3::DecayType::N3ProngDecays, 2>(cand3Prong, tracks);
      }
    } // loop over 3-prong candidates
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HfSelOptimisation>(cfgc)};
}
