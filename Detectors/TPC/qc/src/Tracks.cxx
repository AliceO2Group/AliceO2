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
#include "TMathBase.h"

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/dEdxInfo.h"
#include "TPCQC/Tracks.h"
#include "TPCQC/Helpers.h"

ClassImp(o2::tpc::qc::Tracks);

using namespace o2::tpc::qc;

//______________________________________________________________________________
void Tracks::initializeHistograms()
{

  TH1::AddDirectory(false);
  const auto logPtBinning = helpers::makeLogBinning(100, 0.05, 20);
  // 1d hitograms
  mMapHist["hNClustersBeforeCuts"] = std::make_unique<TH1F>("hNClustersBeforeCuts", "Number of clusters (before cuts);# TPC clusters", 400, -0.5, 399.5);
  mMapHist["hNClustersAfterCuts"] = std::make_unique<TH1F>("hNClustersAfterCuts", "Number of clusters;# TPC clusters", 400, -0.5, 399.5);
  mMapHist["hEta"] = std::make_unique<TH1F>("hEta", "Pseudorapidity;eta", 400, -2., 2.);
  mMapHist["hPhiAside"] = std::make_unique<TH1F>("hPhiAside", "Azimuthal angle, A side;phi", 360, 0., 2 * M_PI);
  mMapHist["hPhiCside"] = std::make_unique<TH1F>("hPhiCside", "Azimuthal angle, C side;phi", 360, 0., 2 * M_PI);
  mMapHist["hPt"] = std::make_unique<TH1F>("hPt", "Transverse momentum;p_T", logPtBinning.size() - 1, logPtBinning.data());
  mMapHist["hSign"] = std::make_unique<TH1F>("hSign", "Sign of electric charge;charge sign", 3, -1.5, 1.5);
  mMapHist["hEtaNeg"] = std::make_unique<TH1F>("hEtaNeg", "Pseudorapidity, neg. tracks;eta", 400, -2., 2.);
  mMapHist["hEtaPos"] = std::make_unique<TH1F>("hEtaPos", "Pseudorapidity, pos. tracks;eta", 400, -2., 2.);
  mMapHist["hPhiAsideNeg"] = std::make_unique<TH1F>("hPhiAsideNeg", "Azimuthal angle, A side, neg. tracks;phi", 360, 0., 2 * M_PI);
  mMapHist["hPhiAsidePos"] = std::make_unique<TH1F>("hPhiAsidePos", "Azimuthal angle, A side, pos. tracks;phi", 360, 0., 2 * M_PI);
  mMapHist["hPhiCsideNeg"] = std::make_unique<TH1F>("hPhiCsideNeg", "Azimuthal angle, C side, neg. tracks;phi", 360, 0., 2 * M_PI);
  mMapHist["hPhiCsidePos"] = std::make_unique<TH1F>("hPhiCsidePos", "Azimuthal angle, C side, pos. tracks;phi", 360, 0., 2 * M_PI);
  mMapHist["hPtNeg"] = std::make_unique<TH1F>("hPtNeg", "Transverse momentum, neg. tracks;p_T", logPtBinning.size() - 1, logPtBinning.data());
  mMapHist["hPtPos"] = std::make_unique<TH1F>("hPtPos", "Transverse momentum, pos. tracks;p_T", logPtBinning.size() - 1, logPtBinning.data());
  mMapHist["hEtaBeforeCuts"] = std::make_unique<TH1F>("hEtaBeforeCuts", "Pseudorapidity (before cuts);eta", 400, -2., 2.);
  mMapHist["hPtBeforeCuts"] = std::make_unique<TH1F>("hPtBeforeCuts", "Transverse momentum (before cuts);p_T", logPtBinning.size() - 1, logPtBinning.data());
  mMapHist["hQOverPt"] = std::make_unique<TH1F>("hQOverPt", "Charge over transverse momentum;q/p_T", 400, -20., 20.);
  mMapHist["hPhiBothSides"] = std::make_unique<TH1F>("hPhiBothSides", "Azimuthal angle, both sides clusters;phi", 360, 0., 2 * M_PI);
  // 2d histograms
  mMapHist["h2DNClustersEta"] = std::make_unique<TH2F>("h2DNClustersEta", "Number of clusters vs. eta;eta;# TPC clusters", 400, -2., 2., 200, -0.5, 199.5);
  mMapHist["h2DNClustersPhiAside"] = std::make_unique<TH2F>("h2DNClustersPhiAside", "Number of clusters vs. phi, A side ;phi;# TPC clusters", 360, 0., 2 * M_PI, 200, -0.5, 199.5);
  mMapHist["h2DNClustersPhiCside"] = std::make_unique<TH2F>("h2DNClustersPhiCside", "Number of clusters vs. phi, C side ;phi;# TPC clusters", 360, 0., 2 * M_PI, 200, -0.5, 199.5);
  mMapHist["h2DNClustersPt"] = std::make_unique<TH2F>("h2DNClustersPt", "Number of clusters vs. p_T;p_T;# TPC clusters", logPtBinning.size() - 1, logPtBinning.data(), 200, -0.5, 199.5);
  mMapHist["h2DEtaPhi"] = std::make_unique<TH2F>("h2DEtaPhi", "Tracks in eta vs. phi;phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);
  mMapHist["h2DEtaPhiNeg"] = std::make_unique<TH2F>("h2DEtaPhiNeg", "Negative tracks in eta vs. phi;phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);
  mMapHist["h2DEtaPhiPos"] = std::make_unique<TH2F>("h2DEtaPhiPos", "Positive tracks in eta vs. phi;phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);
  mMapHist["h2DNClustersEtaBeforeCuts"] = std::make_unique<TH2F>("h2DNClustersEtaBeforeCuts", "NClusters vs. eta (before cuts);eta;# TPC clusters", 400, -2., 2., 200, -0.5, 199.5);
  mMapHist["h2DNClustersPtBeforeCuts"] = std::make_unique<TH2F>("h2DNClustersPtBeforeCuts", "NClusters vs. p_T (before cuts);p_T;# TPC clusters", logPtBinning.size() - 1, logPtBinning.data(), 200, -0.5, 199.5);
  mMapHist["h2DEtaPhiBeforeCuts"] = std::make_unique<TH2F>("h2DEtaPhiBeforeCuts", "Tracks in eta vs. phi (before cuts);phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);
  mMapHist["h2DQOverPtPhiAside"] = std::make_unique<TH2F>("h2DQOverPtPhiAside", "Charger over p_T vs. phi, A side;phi;q/p_T", 360, 0., 2 * M_PI, 400, -20., 20.);
  mMapHist["h2DQOverPtPhiCside"] = std::make_unique<TH2F>("h2DQOverPtPhiCside", "Charger over p_T vs. phi, C side;phi;q/p_T", 360, 0., 2 * M_PI, 400, -20., 20.);
  // 1d histograms
  mMapHist["hEtaRatio"] = std::make_unique<TH1F>("hEtaRatio", "Pseudorapidity, ratio neg./pos. ;eta", 400, -2., 2.);
  mMapHist["hPhiAsideRatio"] = std::make_unique<TH1F>("hPhiAsideRatio", "Azimuthal angle, A side, ratio neg./pos. ;phi", 360, 0., 2 * M_PI);
  mMapHist["hPhiCsideRatio"] = std::make_unique<TH1F>("hPhiCsideRatio", "Azimuthal angle, C side, ratio neg./pos. ;phi", 360, 0., 2 * M_PI);
  mMapHist["hPtRatio"] = std::make_unique<TH1F>("hPtRatio", "Transverse momentum, ratio neg./pos. ;p_T", logPtBinning.size() - 1, logPtBinning.data());
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

  double absEta = TMath::Abs(eta);

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

    if (hasASideOnly == 1) {
      mMapHist["hPhiAside"]->Fill(phi);
    } else if (hasCSideOnly == 1) {
      mMapHist["hPhiCside"]->Fill(phi);
    } else {
      mMapHist["hPhiBothSides"]->Fill(phi);
    }

    mMapHist["hPt"]->Fill(pt);
    mMapHist["hSign"]->Fill(sign);
    mMapHist["hQOverPt"]->Fill(qOverPt);

    if (sign < 0.) {
      mMapHist["hEtaNeg"]->Fill(eta);
      mMapHist["hPtNeg"]->Fill(pt);
      if (hasASideOnly == 1) {
        mMapHist["hPhiAsideNeg"]->Fill(phi);
      } else if (hasCSideOnly == 1) {
        mMapHist["hPhiCsideNeg"]->Fill(phi);
      }
    } else {
      mMapHist["hEtaPos"]->Fill(eta);
      mMapHist["hPtPos"]->Fill(pt);
      if (hasASideOnly == 1) {
        mMapHist["hPhiAsidePos"]->Fill(phi);
      } else if (hasCSideOnly == 1) {
        mMapHist["hPhiCsidePos"]->Fill(phi);
      }
    }

    // ===| 2D histogram filling |===
    mMapHist["h2DNClustersEta"]->Fill(eta, nCls);

    if (hasASideOnly == 1) {
      mMapHist["h2DNClustersPhiAside"]->Fill(phi, nCls);
      mMapHist["h2DQOverPtPhiAside"]->Fill(phi, qOverPt);
    } else if (hasCSideOnly == 1) {
      mMapHist["h2DNClustersPhiCside"]->Fill(phi, nCls);
      mMapHist["h2DQOverPtPhiCside"]->Fill(phi, qOverPt);
    }

    mMapHist["h2DNClustersPt"]->Fill(pt, nCls);
    mMapHist["h2DEtaPhi"]->Fill(phi, eta);

    if (sign < 0.) {
      mMapHist["h2DEtaPhiNeg"]->Fill(phi, eta);
    } else {
      mMapHist["h2DEtaPhiPos"]->Fill(phi, eta);
    }
  }

  return true;
}

//______________________________________________________________________________
void Tracks::processEndOfCycle()
{
  // ===| Dividing of 1D histograms -> Ratios |===
  mMapHist["hEtaRatio"]->Divide(mMapHist["hEtaNeg"].get(), mMapHist["hEtaPos"].get());
  mMapHist["hPhiAsideRatio"]->Divide(mMapHist["hPhiAsideNeg"].get(), mMapHist["hPhiAsidePos"].get());
  mMapHist["hPhiCsideRatio"]->Divide(mMapHist["hPhiCsideNeg"].get(), mMapHist["hPhiCsidePos"].get());
  mMapHist["hPtRatio"]->Divide(mMapHist["hPtNeg"].get(), mMapHist["hPtPos"].get());
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
