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

#define _USE_MATH_DEFINES

#include <cmath>
#include <memory>

// root includes
#include "TFile.h"
#include "TRandom3.h"

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/dEdxInfo.h"
#include "GPUCommonArray.h"
#include "DetectorsBase/Propagator.h"
#include "TPCQC/Tracks.h"
#include "TPCQC/Helpers.h"

ClassImp(o2::tpc::qc::Tracks);

using namespace o2::tpc::qc;
struct binning {
  int bins;
  double min;
  double max;
};
// DCA histograms
const std::vector<std::string_view> types{"A_Pos", "A_Neg", "C_Pos", "C_Neg"};
const binning binsDCAr{200, -5., 5.};
const binning binsDCArLargerRange{400, -10., 10.};
const binning binsEta{200, -1., 1.};
const binning binsClus{120, 60., 180.};
const binning binsClusLargerRange{140, 60., 200.};
//______________________________________________________________________________
void Tracks::initializeHistograms()
{

  TH1::AddDirectory(false);
  const auto logPtBinning = helpers::makeLogBinning(100, 0.05, 20);
  // 1d hitograms
  mMapHist["hNClustersBeforeCuts"] = std::make_unique<TH1F>("hNClustersBeforeCuts", "Number of clusters (before cuts);# TPC clusters", binsClus.max, 0, binsClus.max);
  mMapHist["hNClustersAfterCuts"] = std::make_unique<TH1F>("hNClustersAfterCuts", "Number of clusters;# TPC clusters", binsClus.bins, binsClus.min, binsClus.max);
  mMapHist["hEta"] = std::make_unique<TH1F>("hEta", "Pseudorapidity;#eta", binsEta.bins, binsEta.min, binsEta.max);
  mMapHist["hPhiAside"] = std::make_unique<TH1F>("hPhiAside", "Azimuthal angle, A side;#phi", 360, 0., 2 * M_PI);
  mMapHist["hPhiCside"] = std::make_unique<TH1F>("hPhiCside", "Azimuthal angle, C side;#phi", 360, 0., 2 * M_PI);
  mMapHist["hPt"] = std::make_unique<TH1F>("hPt", "Transverse momentum;#it{p}_{T} (GeV/#it{c})", logPtBinning.size() - 1, logPtBinning.data());
  mMapHist["hSign"] = std::make_unique<TH1F>("hSign", "Sign of electric charge;charge sign", 3, -1.5, 1.5);
  if (!mTurnOffHistosForAsync) {
    mMapHist["hEtaNeg"] = std::make_unique<TH1F>("hEtaNeg", "Pseudorapidity, neg. tracks;#eta", binsEta.bins, binsEta.min, binsEta.max);
    mMapHist["hEtaPos"] = std::make_unique<TH1F>("hEtaPos", "Pseudorapidity, pos. tracks;#eta", binsEta.bins, binsEta.min, binsEta.max);
    mMapHist["hPhiAsideNeg"] = std::make_unique<TH1F>("hPhiAsideNeg", "Azimuthal angle, A side, neg. tracks;#phi", 360, 0., 2 * M_PI);
    mMapHist["hPhiAsidePos"] = std::make_unique<TH1F>("hPhiAsidePos", "Azimuthal angle, A side, pos. tracks;#phi", 360, 0., 2 * M_PI);
    mMapHist["hPhiCsideNeg"] = std::make_unique<TH1F>("hPhiCsideNeg", "Azimuthal angle, C side, neg. tracks;#phi", 360, 0., 2 * M_PI);
    mMapHist["hPhiCsidePos"] = std::make_unique<TH1F>("hPhiCsidePos", "Azimuthal angle, C side, pos. tracks;#phi", 360, 0., 2 * M_PI);
    // 1d ratio histograms
    mMapHist["hEtaRatio"] = std::make_unique<TH1F>("hEtaRatio", "Pseudorapidity, ratio neg./pos.;#eta", binsEta.bins, binsEta.min, binsEta.max);
    mMapHist["hPhiAsideRatio"] = std::make_unique<TH1F>("hPhiAsideRatio", "Azimuthal angle, A side, ratio neg./pos.;#phi", 360, 0., 2 * M_PI);
    mMapHist["hPhiCsideRatio"] = std::make_unique<TH1F>("hPhiCsideRatio", "Azimuthal angle, C side, ratio neg./pos.;#phi", 360, 0., 2 * M_PI);
    mMapHist["hPtRatio"] = std::make_unique<TH1F>("hPtRatio", "Transverse momentum, ratio neg./pos. ;#it{p}_{T}", logPtBinning.size() - 1, logPtBinning.data());
    mMapHist["hPhiBothSides"] = std::make_unique<TH1F>("hPhiBothSides", "Azimuthal angle, both sides clusters;#phi", 360, 0., 2 * M_PI);
    mMapHist["hPtNeg"] = std::make_unique<TH1F>("hPtNeg", "Transverse momentum, neg. tracks;#it{p}_{T} (GeV/#it{c})", logPtBinning.size() - 1, logPtBinning.data());
    mMapHist["hPtPos"] = std::make_unique<TH1F>("hPtPos", "Transverse momentum, pos. tracks;#it{p}_{T} (GeV/#it{c})", logPtBinning.size() - 1, logPtBinning.data());
  }
  mMapHist["hEtaBeforeCuts"] = std::make_unique<TH1F>("hEtaBeforeCuts", "Pseudorapidity (before cuts);#eta", 400, -2., 2.);
  mMapHist["hPtBeforeCuts"] = std::make_unique<TH1F>("hPtBeforeCuts", "Transverse momentum (before cuts);#it{p}_{T} (GeV/#it{c})", logPtBinning.size() - 1, logPtBinning.data());
  mMapHist["hQOverPt"] = std::make_unique<TH1F>("hQOverPt", "Charge over transverse momentum;q/#it{p}_{T}", 400, -20., 20.);
  // 2d histograms
  mMapHist["h2DNClustersEta"] = std::make_unique<TH2F>("h2DNClustersEta", "Number of clusters vs. #eta;#eta;# TPC clusters", binsEta.bins, binsEta.min, binsEta.max, binsClusLargerRange.bins, binsClusLargerRange.min, binsClusLargerRange.max);
  mMapHist["h2DNClustersPhiAside"] = std::make_unique<TH2F>("h2DNClustersPhiAside", "Number of clusters vs. #phi, A side ;#phi;# TPC clusters", 360, 0., 2 * M_PI, binsClus.bins, binsClus.min, binsClus.max);
  mMapHist["h2DNClustersPhiCside"] = std::make_unique<TH2F>("h2DNClustersPhiCside", "Number of clusters vs. #phi, C side ;#phi;# TPC clusters", 360, 0., 2 * M_PI, binsClus.bins, binsClus.min, binsClus.max);
  mMapHist["h2DNClustersPt"] = std::make_unique<TH2F>("h2DNClustersPt", "Number of clusters vs. #it{p}_{T};#it{p}_{T} (GeV/#it{c});# TPC clusters", logPtBinning.size() - 1, logPtBinning.data(), binsClusLargerRange.bins, binsClusLargerRange.min, binsClusLargerRange.max);
  mMapHist["h2DEtaPhi"] = std::make_unique<TH2F>("h2DEtaPhi", "Tracks in #eta vs. #phi;#phi;#eta", 360, 0., 2 * M_PI, binsEta.bins, binsEta.min, binsEta.max);
  mMapHist["h2DNClustersEtaBeforeCuts"] = std::make_unique<TH2F>("h2DNClustersEtaBeforeCuts", "NClusters vs. #eta (before cuts);#eta;# TPC clusters", 400, -2., 2., 200, -0.5, 199.5);
  mMapHist["h2DNClustersPtBeforeCuts"] = std::make_unique<TH2F>("h2DNClustersPtBeforeCuts", "NClusters vs. #it{p}_{T} (before cuts);#it{p}_{T} (GeV/#it{c});# TPC clusters", logPtBinning.size() - 1, logPtBinning.data(), 200, -0.5, 199.5);
  mMapHist["h2DEtaPhiBeforeCuts"] = std::make_unique<TH2F>("h2DEtaPhiBeforeCuts", "Tracks in #eta vs. #phi (before cuts);#phi;#eta", 360, 0., 2 * M_PI, 400, -2., 2.);
  if (!mTurnOffHistosForAsync) {
    mMapHist["h2DQOverPtPhiAside"] = std::make_unique<TH2F>("h2DQOverPtPhiAside", "Charger over #it{p}_{T} vs. #phi, A side;#phi;q/#it{p}_{T}", 360, 0., 2 * M_PI, 400, -20., 20.);
    mMapHist["h2DQOverPtPhiCside"] = std::make_unique<TH2F>("h2DQOverPtPhiCside", "Charger over #it{p}_{T} vs. #phi, C side;#phi;q/#it{p}_{T}", 360, 0., 2 * M_PI, 400, -20., 20.);
    mMapHist["h2DEtaPhiNeg"] = std::make_unique<TH2F>("h2DEtaPhiNeg", "Negative tracks in #eta vs. #phi;#phi;#eta", 360, 0., 2 * M_PI, binsEta.bins, binsEta.min, binsEta.max);
    mMapHist["h2DEtaPhiPos"] = std::make_unique<TH2F>("h2DEtaPhiPos", "Positive tracks in #eta vs. #phi;#phi;#eta", 360, 0., 2 * M_PI, binsEta.bins, binsEta.min, binsEta.max);
    // eta vs pt and phi vs pt possitive and negative signs
    mMapHist["hEtaVsPtPos"] = std::make_unique<TH2F>("hEtaVsPtPos", "#eta vs. #it{p}_{T} (Pos.);#it{p}_{T} (GeV/#it{c});#eta", logPtBinning.size() - 1, logPtBinning.data(), binsEta.bins, binsEta.min, binsEta.max);
    mMapHist["hEtaVsPtNeg"] = std::make_unique<TH2F>("hEtaVsPtNeg", "#eta vs. #it{p}_{T} (Neg.);#it{p}_{T} (GeV/#it{c});#eta", logPtBinning.size() - 1, logPtBinning.data(), binsEta.bins, binsEta.min, binsEta.max);
    mMapHist["hPhiVsPtPos"] = std::make_unique<TH2F>("hPhiVsPtPos", "#phi vs. #it{p}_{T} (Pos.);#it{p}_{T} (GeV/#it{c});#phi", logPtBinning.size() - 1, logPtBinning.data(), 360, 0., 2 * M_PI);
    mMapHist["hPhiVsPtNeg"] = std::make_unique<TH2F>("hPhiVsPtNeg", "#phi vs. #it{p}_{T} (Neg.);#it{p}_{T} (GeV/#it{c});#phi", logPtBinning.size() - 1, logPtBinning.data(), 360, 0., 2 * M_PI);
  }
  // DCA Histograms
  for (const auto type : types) {
    mMapHist[fmt::format("hDCAr_{}", type).data()] = std::make_unique<TH2F>(fmt::format("hDCAr_{}", type).data(), fmt::format("DCAr {};#phi;DCAr (cm)", type).data(), 360, 0, o2::math_utils::twoPid(), binsDCAr.bins, binsDCAr.min, binsDCAr.max);
    mMapHist[fmt::format("hDCAr_{}_pTmin", type).data()] = std::make_unique<TH2F>(fmt::format("hDCAr_{}_pTmin", type).data(), fmt::format("DCAr {} #it{{p}}_{{T}}^{{min}};#phi;DCAr (cm)", type).data(), 360, 0, o2::math_utils::twoPid(), binsDCAr.bins, binsDCAr.min, binsDCAr.max);
  }
  // DCA vs variables Histograms
  mMapHist["hDCArVsPtPos"] = std::make_unique<TH2F>("hDCArVsPtPos", "DCAr Pos;#it{p}_{T} (GeV/#it{c});DCAr (cm)", logPtBinning.size() - 1, logPtBinning.data(), binsDCArLargerRange.bins, binsDCArLargerRange.min, binsDCArLargerRange.max);
  mMapHist["hDCArVsEtaPos"] = std::make_unique<TH2F>("hDCArVsEtaPos", "DCAr Pos;#eta;DCAr (cm)", binsEta.bins, binsEta.min, binsEta.max, binsDCArLargerRange.bins, binsDCArLargerRange.min, binsDCArLargerRange.max);
  mMapHist["hDCArVsNClsPos"] = std::make_unique<TH2F>("hDCArVsNClsPos", "DCAr Pos;# TPC clusters;DCAr (cm)", binsClus.bins, binsClus.min, binsClus.max, binsDCAr.bins, binsDCAr.min, binsDCAr.max);
  mMapHist["hDCArVsPtNeg"] = std::make_unique<TH2F>("hDCArVsPtNeg", "DCAr Neg;#it{p}_{T} (GeV/#it{c});DCAr (cm)", logPtBinning.size() - 1, logPtBinning.data(), binsDCArLargerRange.bins, binsDCArLargerRange.min, binsDCArLargerRange.max);
  mMapHist["hDCArVsEtaNeg"] = std::make_unique<TH2F>("hDCArVsEtaNeg", "DCAr Neg;#eta;DCAr (cm)", binsEta.bins, binsEta.min, binsEta.max, binsDCArLargerRange.bins, binsDCArLargerRange.min, binsDCArLargerRange.max);
  mMapHist["hDCArVsNClsNeg"] = std::make_unique<TH2F>("hDCArVsNClsNeg", "DCAr Neg;# TPC clusters;DCAr (cm)", binsClus.bins, binsClus.min, binsClus.max, binsDCAr.bins, binsDCAr.min, binsDCAr.max);
  // DCA vs variables Histogram with pT min selection
  mMapHist["hDCArVsEtaPos_pTmin"] = std::make_unique<TH2F>("hDCArVsEtaPos_pTmin", "DCAr Pos #it{p}_{T}^{min};#eta;DCAr (cm)", binsEta.bins, binsEta.min, binsEta.max, binsDCAr.bins, binsDCAr.min, binsDCAr.max);
  mMapHist["hDCArVsNClsPos_pTmin"] = std::make_unique<TH2F>("hDCArVsNClsPos_pTmin", "DCAr Pos #it{p}_{T}^{min};# TPC clusters;DCAr (cm)", binsClus.bins, binsClus.min, binsClus.max, binsDCAr.bins, binsDCAr.min, binsDCAr.max);
  mMapHist["hDCArVsEtaNeg_pTmin"] = std::make_unique<TH2F>("hDCArVsEtaNeg_pTmin", "DCAr Neg #it{p}_{T}^{min};#eta;DCAr (cm)", binsEta.bins, binsEta.min, binsEta.max, binsDCAr.bins, binsDCAr.min, binsDCAr.max);
  mMapHist["hDCArVsNClsNeg_pTmin"] = std::make_unique<TH2F>("hDCArVsNClsNeg_pTmin", "DCAr Neg #it{p}_{T}^{min};# TPC clusters;DCAr (cm)", binsClus.bins, binsClus.min, binsClus.max, binsDCAr.bins, binsDCAr.min, binsDCAr.max);
}
//______________________________________________________________________________
void Tracks::resetHistograms()
{
  for (const auto& pair : mMapHist) {
    pair.second->Reset();
  }
}
//______________________________________________________________________________
bool Tracks::processTrack(const o2::tpc::TrackTPC& track)
{
  // ===| variables required for cutting and filling |===
  const auto eta = track.getEta();
  const auto phi = track.getPhi();
  const auto pt = track.getPt();
  const auto sign = track.getSign();
  const auto qOverPt = track.getQ2Pt();
  const auto nCls = track.getNClusterReferences();
  const auto dEdxTot = track.getdEdx().dEdxTotTPC;
  const auto hasASideOnly = track.hasASideClustersOnly();
  const auto hasCSideOnly = track.hasCSideClustersOnly();

  const auto absEta = std::abs(eta);

  // ===| histogram filling before cuts |===
  mMapHist["hNClustersBeforeCuts"]->Fill(nCls);
  mMapHist["hEtaBeforeCuts"]->Fill(eta);
  mMapHist["hPtBeforeCuts"]->Fill(pt);
  mMapHist["h2DNClustersEtaBeforeCuts"]->Fill(eta, nCls);
  mMapHist["h2DNClustersPtBeforeCuts"]->Fill(pt, nCls);
  mMapHist["h2DEtaPhiBeforeCuts"]->Fill(phi, eta);

  // ===| histogram filling including cuts |===
  if (absEta < mCutAbsEta && nCls > mCutMinnCls && dEdxTot > mCutMindEdxTot) {

    // ===| 1D histogram filling |===
    mMapHist["hNClustersAfterCuts"]->Fill(nCls);
    mMapHist["hEta"]->Fill(eta);

    //---| propagate to 0,0,0 |---
    //
    // propagator instance must be configured before (LUT, MagField)
    auto propagator = o2::base::Propagator::Instance(true);
    const int type = (track.getQ2Pt() < 0) + 2 * track.hasCSideClustersOnly();
    auto dcaHist = mMapHist[fmt::format("hDCAr_{}", types[type]).data()].get();
    auto dcaHist_pTmin = mMapHist[fmt::format("hDCAr_{}_pTmin", types[type]).data()].get();
    const std::string signType((sign < 0) ? "Neg" : "Pos");
    auto dcaHistPT = mMapHist["hDCArVsPt" + signType].get();
    auto dcaHistEta = mMapHist["hDCArVsEta" + signType].get();
    auto dcaHistNCluster = mMapHist["hDCArVsNCls" + signType].get();
    auto dcaHistEta_pTmin = mMapHist["hDCArVsEta" + signType + "_pTmin"].get();
    auto dcaHistNCluster_pTmin = mMapHist["hDCArVsNCls" + signType + "_pTmin"].get();

    // set-up sampling for the DCA calculation
    Double_t sampleProb = 2;

    if (mSamplingFractionDCAr > 0) { // for now no SEED is given.
      TRandom3 randomGenerator(0);
      sampleProb = randomGenerator.Uniform(1);
    }

    if (sampleProb > (Double_t)(1. - mSamplingFractionDCAr)) {

      if (propagator->getMatLUT() && propagator->hasMagFieldSet()) {
        // ---| fill DCA histos |---
        o2::gpu::gpustd::array<float, 2> dca;
        const o2::math_utils::Point3D<float> refPoint{0, 0, 0};
        o2::track::TrackPar propTrack(track);
        if (propagator->propagateToDCABxByBz(refPoint, propTrack, 2.f, o2::base::Propagator::MatCorrType::USEMatCorrLUT, &dca)) {
          const auto phi = o2::math_utils::to02PiGen(track.getPhi());
          dcaHistPT->Fill(pt, dca[0]);
          dcaHist->Fill(phi, dca[0]);
          dcaHistEta->Fill(eta, dca[0]);
          dcaHistNCluster->Fill(nCls, dca[0]);
          if (pt > mCutMinPtDCAr) {
            dcaHist_pTmin->Fill(phi, dca[0]);
            dcaHistEta_pTmin->Fill(eta, dca[0]);
            dcaHistNCluster_pTmin->Fill(nCls, dca[0]);
          }
        }
      } else {
        static bool reported = false;
        if (!reported) {
          LOGP(error, "o2::base::Propagator not properly initialized, MatLUT ({}) and / or Field ({}) missing, will not fill DCA histograms", (void*)propagator->getMatLUT(), propagator->hasMagFieldSet());
          dcaHist->SetTitle(fmt::format("DCAr {} o2::base::Propagator not properly initialized", types[type]).data());
          dcaHistPT->SetTitle(fmt::format("DCAr #it{{p}}_{{T}} {} o2::base::Propagator not properly initialized", signType).data());
          dcaHist_pTmin->SetTitle(fmt::format("DCAr {} #it{{p}}_{{T}}^{{min}} o2::base::Propagator not properly initialized", types[type]).data());
          dcaHistEta->SetTitle(fmt::format("DCAr #eta {} o2::base::Propagator not properly initialized", signType).data());
          dcaHistNCluster->SetTitle(fmt::format("DCAr nClusters {} o2::base::Propagator not properly initialized", signType).data());
          dcaHistEta_pTmin->SetTitle(fmt::format("DCAr #eta {} #it{{p}}_{{T}}^{{min}} o2::base::Propagator not properly initialized", signType).data());
          dcaHistNCluster_pTmin->SetTitle(fmt::format("DCAr nClusters {} #it{{p}}_{{T}}^{{min}} o2::base::Propagator not properly initialized", signType).data());
          reported = true;
        }
      }
    }

    if (hasASideOnly == 1) {
      mMapHist["hPhiAside"]->Fill(phi);
    } else if (hasCSideOnly == 1) {
      mMapHist["hPhiCside"]->Fill(phi);
    } else {
      if (!mTurnOffHistosForAsync) {
        mMapHist["hPhiBothSides"]->Fill(phi);
      }
    }

    mMapHist["hPt"]->Fill(pt);
    mMapHist["hSign"]->Fill(sign);
    mMapHist["hQOverPt"]->Fill(qOverPt);

    if (sign < 0.) {
      if (!mTurnOffHistosForAsync) {
        mMapHist["hEtaNeg"]->Fill(eta);
        mMapHist["hPtNeg"]->Fill(pt);
      }
      if (!mTurnOffHistosForAsync) {
        if (hasASideOnly == 1) {
          mMapHist["hPhiAsideNeg"]->Fill(phi);
        } else if (hasCSideOnly == 1) {
          mMapHist["hPhiCsideNeg"]->Fill(phi);
        }
      }
    } else {
      if (!mTurnOffHistosForAsync) {
        mMapHist["hEtaPos"]->Fill(eta);
        mMapHist["hPtPos"]->Fill(pt);
      }
      if (!mTurnOffHistosForAsync) {
        if (hasASideOnly == 1) {
          mMapHist["hPhiAsidePos"]->Fill(phi);
        } else if (hasCSideOnly == 1) {
          mMapHist["hPhiCsidePos"]->Fill(phi);
        }
      }
    }

    // ===| 2D histogram filling |===
    mMapHist["h2DNClustersEta"]->Fill(eta, nCls);

    if (hasASideOnly == 1) {
      mMapHist["h2DNClustersPhiAside"]->Fill(phi, nCls);
      if (!mTurnOffHistosForAsync) {
        mMapHist["h2DQOverPtPhiAside"]->Fill(phi, qOverPt);
      }
    } else if (hasCSideOnly == 1) {
      mMapHist["h2DNClustersPhiCside"]->Fill(phi, nCls);
      if (!mTurnOffHistosForAsync) {
        mMapHist["h2DQOverPtPhiCside"]->Fill(phi, qOverPt);
      }
    }

    mMapHist["h2DNClustersPt"]->Fill(pt, nCls);
    mMapHist["h2DEtaPhi"]->Fill(phi, eta);

    if (!mTurnOffHistosForAsync) {
      if (sign < 0.) {
        mMapHist["h2DEtaPhiNeg"]->Fill(phi, eta);
        mMapHist["hEtaVsPtNeg"]->Fill(pt, eta);
        mMapHist["hPhiVsPtNeg"]->Fill(pt, phi);
      } else {
        mMapHist["h2DEtaPhiPos"]->Fill(phi, eta);
        mMapHist["hEtaVsPtPos"]->Fill(pt, eta);
        mMapHist["hPhiVsPtPos"]->Fill(pt, phi);
      }
    }
  }

  return true;
}

//______________________________________________________________________________
void Tracks::processEndOfCycle()
{
  if (!mTurnOffHistosForAsync) {
    // ===| Dividing of 1D histograms -> Ratios |===
    mMapHist["hEtaRatio"]->Divide(mMapHist["hEtaNeg"].get(), mMapHist["hEtaPos"].get());
    mMapHist["hPhiAsideRatio"]->Divide(mMapHist["hPhiAsideNeg"].get(), mMapHist["hPhiAsidePos"].get());
    mMapHist["hPhiCsideRatio"]->Divide(mMapHist["hPhiCsideNeg"].get(), mMapHist["hPhiCsidePos"].get());
    mMapHist["hPtRatio"]->Divide(mMapHist["hPtNeg"].get(), mMapHist["hPtPos"].get());
  }
}

//______________________________________________________________________________
void Tracks::dumpToFile(std::string_view filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.data(), "recreate"));
  for (const auto& [name, hist] : mMapHist) {
    TObjArray arr;
    arr.SetName(name.data());
    arr.Add(hist.get());
    arr.Write(arr.GetName(), TObject::kSingleKey);
  }
  f->Close();
}
