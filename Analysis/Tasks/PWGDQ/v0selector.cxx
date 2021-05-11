// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Example analysis task to select clean V0 sample
// ========================
//
// This code loops over a V0Data table and produces some
// standard analysis output. It requires either
// the lambdakzerofinder or the lambdakzeroproducer tasks
// to have been executed in the workflow (before).
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    daiki.sekihata@cern.ch
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisDataModel/StrangenessTables.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"

#include <Math/Vector4D.h>
#include <array>
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

//using FullTracksExt = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksExtended, aod::TrackSelection, aod::pidTPC, aod::pidTOF, aod::pidTOFbeta>;
using FullTracksExt = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTOFbeta>;

struct v0selector {

  void init(o2::framework::InitContext&)
  {
  }

  //Basic checks
  HistogramRegistry registry{
    "registry",
    {
      {"hMassGamma", "hMassGamma", {HistType::kTH1F, {{100, 0.0f, 0.1f}}}},
      {"hMassK0S", "hMassK0S", {HistType::kTH1F, {{100, 0.45, 0.55}}}},
      {"hMassLambda", "hMasLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},
      {"hMassAntiLambda", "hAntiMasLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},

      {"hMassGamma_AP", "hMassGamma_AP", {HistType::kTH1F, {{100, 0.0f, 0.1f}}}},
      {"hMassK0S_AP", "hMassK0S_AP", {HistType::kTH1F, {{100, 0.45, 0.55}}}},
      {"hMassLambda_AP", "hMasLambda_AP", {HistType::kTH1F, {{100, 1.05, 1.15}}}},
      {"hMassAntiLambda_AP", "hAntiMasLambda_AP", {HistType::kTH1F, {{100, 1.05, 1.15}}}},

      {"h2MassGammaR", "h2MassGammaR", {HistType::kTH2F, {{1000, 0.0, 100}, {100, 0.0f, 0.1f}}}},

      {"hV0Pt", "pT", {HistType::kTH1F, {{100, 0.0f, 10}}}},
      {"hV0EtaPhi", "#eta vs. #varphi", {HistType::kTH2F, {{126, -6.3, 6.3}, {20, -1.0f, 1.0f}}}},

      {"hV0Radius", "hV0Radius", {HistType::kTH1F, {{1000, 0.0f, 100.0f}}}},
      {"hV0CosPA", "hV0CosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
      {"hDCAPosToPV", "hDCAPosToPV", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hDCANegToPV", "hDCANegToPV", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hDCAV0Dau", "hDCAV0Dau", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hV0APplot", "hV0APplot", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {300, 0.0f, 0.3f}}}},
      {"hV0APplot_Gamma", "hV0APplot Gamma", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {300, 0.0f, 0.3f}}}},

      {"h2TPCdEdx_Pin_Pos", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Neg", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_El_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_El_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pi_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pi_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Ka_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Ka_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pr_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pr_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},

      {"h2TOFbeta_Pin_Pos", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Neg", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_El_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_El_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pi_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pi_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Ka_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Ka_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pr_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pr_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},

      {"h2MggPt", "M_{#gamma#gamma} vs. p_{T}", {HistType::kTH2F, {{400, 0.0, 0.8}, {100, 0.0, 10.}}}},
    },
  };

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision, aod::V0Datas const& fullV0s, FullTracksExt const& tracks)
  {

    //if (!collision.alias()[kINT7]) {
    //  return;
    //}
    //if (!collision.sel7()) {
    //  return;
    //}

    for (auto& v0 : fullV0s) {
      if (v0.negTrack_as<FullTracksExt>().tpcChi2NCl() > 4.0 || v0.posTrack_as<FullTracksExt>().tpcChi2NCl() > 4.0) {
        continue;
      }
      if (v0.negTrack_as<FullTracksExt>().tpcNClsCrossedRows() < 70 || v0.posTrack_as<FullTracksExt>().tpcNClsCrossedRows() < 70) {
        continue;
      }
      if (!(v0.negTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit) || !(v0.posTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
        continue;
      }

      registry.fill(HIST("hV0Pt"), v0.pt());
      registry.fill(HIST("hV0EtaPhi"), v0.phi(), v0.eta());

      registry.fill(HIST("hV0Radius"), v0.v0radius());
      registry.fill(HIST("hV0CosPA"), v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()));
      registry.fill(HIST("hDCAPosToPV"), v0.dcapostopv());
      registry.fill(HIST("hDCANegToPV"), v0.dcanegtopv());
      registry.fill(HIST("hDCAV0Dau"), v0.dcaV0daughters());
      registry.fill(HIST("hV0APplot"), v0.alpha(), v0.qtarm());

      registry.fill(HIST("hMassK0S"), v0.mK0Short());
      registry.fill(HIST("hMassLambda"), v0.mLambda());
      registry.fill(HIST("hMassAntiLambda"), v0.mAntiLambda());

      registry.fill(HIST("h2TPCdEdx_Pin_Neg"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().tpcSignal());
      registry.fill(HIST("h2TPCdEdx_Pin_Pos"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().tpcSignal());

      registry.fill(HIST("h2TOFbeta_Pin_Neg"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().beta());
      registry.fill(HIST("h2TOFbeta_Pin_Pos"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().beta());

      if ((70.0 < v0.negTrack_as<FullTracksExt>().tpcSignal() && v0.negTrack_as<FullTracksExt>().tpcSignal() < 90) && (70.0 < v0.posTrack_as<FullTracksExt>().tpcSignal() && v0.posTrack_as<FullTracksExt>().tpcSignal() < 90)) {
        registry.fill(HIST("hMassGamma"), v0.mGamma());
        registry.fill(HIST("hV0APplot_Gamma"), v0.alpha(), v0.qtarm());
        registry.fill(HIST("h2MassGammaR"), v0.v0radius(), v0.mGamma());
      }

      if (TMath::Abs(v0.alpha()) < 0.4 && v0.qtarm() < 0.03 && v0.mGamma() < 0.01) { //photon conversion
        registry.fill(HIST("hMassGamma_AP"), v0.mGamma());
        registry.fill(HIST("h2TPCdEdx_Pin_El_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TPCdEdx_Pin_El_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_El_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().beta());
        registry.fill(HIST("h2TOFbeta_Pin_El_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().beta());

      } else if (TMath::Abs(v0.alpha()) < 0.7 && (0.11 < v0.qtarm() && v0.qtarm() < 0.22) && (0.49 < v0.mK0Short() && v0.mK0Short() < 0.51)) { //K0S-> pi pi
        registry.fill(HIST("hMassK0S_AP"), v0.mK0Short());
        registry.fill(HIST("h2TPCdEdx_Pin_Pi_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TPCdEdx_Pin_Pi_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Pi_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().beta());
        registry.fill(HIST("h2TOFbeta_Pin_Pi_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().beta());

      } else if ((+0.45 < v0.alpha() && v0.alpha() < +0.7) && (0.03 < v0.qtarm() && v0.qtarm() < 0.11) && (1.112 < v0.mLambda() && v0.mLambda() < 1.120)) { //L->p + pi-
        registry.fill(HIST("hMassLambda_AP"), v0.mLambda());
        registry.fill(HIST("h2TPCdEdx_Pin_Pi_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TPCdEdx_Pin_Pr_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Pi_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().beta());
        registry.fill(HIST("h2TOFbeta_Pin_Pr_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().beta());

      } else if ((-0.7 < v0.alpha() && v0.alpha() < -0.45) && (0.03 < v0.qtarm() && v0.qtarm() < 0.11) && (1.112 < v0.mAntiLambda() && v0.mAntiLambda() < 1.120)) { //Lbar -> pbar + pi+
        registry.fill(HIST("hMassAntiLambda_AP"), v0.mAntiLambda());
        registry.fill(HIST("h2TPCdEdx_Pin_Pr_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TPCdEdx_Pin_Pi_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Pr_minus"), v0.negTrack_as<FullTracksExt>().tpcInnerParam(), v0.negTrack_as<FullTracksExt>().beta());
        registry.fill(HIST("h2TOFbeta_Pin_Pi_plus"), v0.posTrack_as<FullTracksExt>().tpcInnerParam(), v0.posTrack_as<FullTracksExt>().beta());
      }
    }

    //loop for pi0/eta->gamma gamma
    for (auto& [t1, t2] : combinations(fullV0s, fullV0s)) {

      if (t1.negTrack_as<FullTracksExt>().tpcSignal() < 70.0 || 90.0 < t1.negTrack_as<FullTracksExt>().tpcSignal()) {
        continue;
      }
      if (t1.posTrack_as<FullTracksExt>().tpcSignal() < 70.0 || 90.0 < t1.posTrack_as<FullTracksExt>().tpcSignal()) {
        continue;
      }
      if (t2.negTrack_as<FullTracksExt>().tpcSignal() < 70.0 || 90.0 < t2.negTrack_as<FullTracksExt>().tpcSignal()) {
        continue;
      }
      if (t2.posTrack_as<FullTracksExt>().tpcSignal() < 70.0 || 90.0 < t2.posTrack_as<FullTracksExt>().tpcSignal()) {
        continue;
      }
      if (t1.negTrack_as<FullTracksExt>().tpcChi2NCl() > 4.0 || t1.posTrack_as<FullTracksExt>().tpcChi2NCl() > 4.0) {
        continue;
      }
      if (t2.negTrack_as<FullTracksExt>().tpcChi2NCl() > 4.0 || t2.posTrack_as<FullTracksExt>().tpcChi2NCl() > 4.0) {
        continue;
      }

      //if(TMath::Abs(t1.alpha()) > 0.35 || TMath::Abs(t2.alpha()) > 0.35) continue;
      if (t1.qtarm() > 0.03 || t2.qtarm() > 0.03) {
        continue;
      }
      if (t1.mGamma() > 0.01 || t2.mGamma() > 0.01) {
        continue;
      }
      if (t1.v0radius() > 50 || t2.v0radius() > 50) {
        continue;
      }

      //ROOT::Math::PtEtaPhiMVector v1(t1.pt(), t1.eta(), t1.phi(), t1.mGamma());
      //ROOT::Math::PtEtaPhiMVector v2(t2.pt(), t2.eta(), t2.phi(), t2.mGamma());
      ROOT::Math::PtEtaPhiMVector v1(t1.pt(), t1.eta(), t1.phi(), 0.0);
      ROOT::Math::PtEtaPhiMVector v2(t2.pt(), t2.eta(), t2.phi(), 0.0);
      ROOT::Math::PtEtaPhiMVector v12 = v1 + v2;
      registry.fill(HIST("h2MggPt"), v12.M(), v12.Pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<v0selector>(cfgc)};
}
