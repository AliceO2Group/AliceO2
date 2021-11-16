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
  const auto logPtBinning = helpers::makeLogBinning(100, 0.05, 20);

  mHist1D.emplace_back("hNClustersBeforeCuts", "Number of clusters (before cuts);# TPC clusters", 160, -0.5, 159.5);            //| mHist1D[0]
  mHist1D.emplace_back("hNClustersAfterCuts", "Number of clusters;# TPC clusters", 160, -0.5, 159.5);                           //| mHist1D[1]
  mHist1D.emplace_back("hEta", "Pseudorapidity;eta", 400, -2., 2.);                                                             //| mHist1D[2]
  mHist1D.emplace_back("hPhiAside", "Azimuthal angle, A side;phi", 360, 0., 2 * M_PI);                                          //| mHist1D[3]
  mHist1D.emplace_back("hPhiCside", "Azimuthal angle, C side;phi", 360, 0., 2 * M_PI);                                          //| mHist1D[4]
  mHist1D.emplace_back("hPt", "Transverse momentum;p_T", logPtBinning.size() - 1, logPtBinning.data());                         //| mHist1D[5]
  mHist1D.emplace_back("hSign", "Sign of electric charge;charge sign", 3, -1.5, 1.5);                                           //| mHist1D[6]
  mHist1D.emplace_back("hEtaNeg", "Pseudorapidity, neg. tracks;eta", 400, -2., 2.);                                             //| mHist1D[7]
  mHist1D.emplace_back("hEtaPos", "Pseudorapidity, pos. tracks;eta", 400, -2., 2.);                                             //| mHist1D[8]
  mHist1D.emplace_back("hPhiAsideNeg", "Azimuthal angle, A side, neg. tracks;phi", 360, 0., 2 * M_PI);                          //| mHist1D[9]
  mHist1D.emplace_back("hPhiAsidePos", "Azimuthal angle, A side, pos. tracks;phi", 360, 0., 2 * M_PI);                          //| mHist1D[10]
  mHist1D.emplace_back("hPhiCsideNeg", "Azimuthal angle, C side, neg. tracks;phi", 360, 0., 2 * M_PI);                          //| mHist1D[11]
  mHist1D.emplace_back("hPhiCsidePos", "Azimuthal angle, C side, pos. tracks;phi", 360, 0., 2 * M_PI);                          //| mHist1D[12]
  mHist1D.emplace_back("hPtNeg", "Transverse momentum, neg. tracks;p_T", logPtBinning.size() - 1, logPtBinning.data());         //| mHist1D[13]
  mHist1D.emplace_back("hPtPos", "Transverse momentum, pos. tracks;p_T", logPtBinning.size() - 1, logPtBinning.data());         //| mHist1D[14]
  mHist1D.emplace_back("hEtaBeforeCuts", "Pseudorapidity (before cuts);eta", 400, -2., 2.);                                     //| mHist1D[15]
  mHist1D.emplace_back("hPtBeforeCuts", "Transverse momentum (before cuts);p_T", logPtBinning.size() - 1, logPtBinning.data()); //| mHist1D[16]

  mHist2D.emplace_back("h2DNClustersEta", "Number of clusters vs. eta;eta;# TPC clusters", 400, -2., 2., 160, -0.5, 159.5);                                               //| mHist2D[0]
  mHist2D.emplace_back("h2DNClustersPhiAside", "Number of clusters vs. phi, A side ;phi;# TPC clusters", 360, 0., 2 * M_PI, 160, -0.5, 159.5);                            //| mHist2D[1]
  mHist2D.emplace_back("h2DNClustersPhiCside", "Number of clusters vs. phi, C side ;phi;# TPC clusters", 360, 0., 2 * M_PI, 160, -0.5, 159.5);                            //| mHist2D[2]
  mHist2D.emplace_back("h2DNClustersPt", "Number of clusters vs. p_T;p_T;# TPC clusters", logPtBinning.size() - 1, logPtBinning.data(), 160, -0.5, 159.5);                //| mHist2D[3]
  mHist2D.emplace_back("h2DEtaPhi", "Tracks in eta vs. phi;phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);                                                                    //| mHist2D[4]
  mHist2D.emplace_back("h2DEtaPhiNeg", "Negative tracks in eta vs. phi;phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);                                                        //| mHist2D[5]
  mHist2D.emplace_back("h2DEtaPhiPos", "Positive tracks in eta vs. phi;phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);                                                        //| mHist2D[6]
  mHist2D.emplace_back("h2DNClustersEtaBeforeCuts", "NClusters vs. eta (before cuts);eta;# TPC clusters", 400, -2., 2., 160, -0.5, 159.5);                                //| mHist2D[7]
  mHist2D.emplace_back("h2DNClustersPtBeforeCuts", "NClusters vs. p_T (before cuts);p_T;# TPC clusters", logPtBinning.size() - 1, logPtBinning.data(), 160, -0.5, 159.5); //| mHist2D[8]
  mHist2D.emplace_back("h2DEtaPhiBeforeCuts", "Tracks in eta vs. phi (before cuts);phi;eta", 360, 0., 2 * M_PI, 400, -2., 2.);                                            //| mHist2D[9]

  mHistRatio1D.emplace_back("hEtaRatio", "Pseudorapidity, ratio neg./pos. ;eta", 400, -2., 2.);                                     //| mHistRatio1D[0]
  mHistRatio1D.emplace_back("hPhiAsideRatio", "Azimuthal angle, A side, ratio neg./pos. ;phi", 360, 0., 2 * M_PI);                  //| mHistRatio1D[1]
  mHistRatio1D.emplace_back("hPhiCsideRatio", "Azimuthal angle, C side, ratio neg./pos. ;phi", 360, 0., 2 * M_PI);                  //| mHistRatio1D[2]
  mHistRatio1D.emplace_back("hPtRatio", "Transverse momentum, ratio neg./pos. ;p_T", logPtBinning.size() - 1, logPtBinning.data()); //| mHistRatio1D[3]
}

//______________________________________________________________________________
void Tracks::resetHistograms()
{
  for (auto& hist : mHist1D) {
    hist.Reset();
  }
  for (auto& hist2 : mHist2D) {
    hist2.Reset();
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
  const auto nCls = track.getNClusterReferences();
  const auto dEdxTot = track.getdEdx().dEdxTotTPC;

  double absEta = TMath::Abs(eta);

  // ===| histogram filling before cuts |===
  mHist1D[0].Fill(nCls);
  mHist1D[15].Fill(eta);
  mHist1D[16].Fill(pt);
  mHist2D[7].Fill(eta, nCls);
  mHist2D[8].Fill(pt, nCls);
  mHist2D[9].Fill(phi, eta);

  // ===| histogram filling including cuts |===
  if (absEta < 1. && nCls > 60 && dEdxTot > 20) {

    // ===| 1D histogram filling |===
    mHist1D[1].Fill(nCls);
    mHist1D[2].Fill(eta);

    if (eta > 0.) {
      mHist1D[3].Fill(phi);
    } else {
      mHist1D[4].Fill(phi);
    }

    mHist1D[5].Fill(pt);
    mHist1D[6].Fill(sign);

    if (sign < 0.) {
      mHist1D[7].Fill(eta);
      mHist1D[13].Fill(pt);
      if (eta > 0.) {
        mHist1D[9].Fill(phi);
      } else {
        mHist1D[11].Fill(phi);
      }
    } else {
      mHist1D[8].Fill(eta);
      mHist1D[14].Fill(pt);
      if (eta > 0.) {
        mHist1D[10].Fill(phi);
      } else {
        mHist1D[12].Fill(phi);
      }
    }

    // ===| 2D histogram filling |===
    mHist2D[0].Fill(eta, nCls);

    if (eta > 0.) {
      mHist2D[1].Fill(phi, nCls);
    } else {
      mHist2D[2].Fill(phi, nCls);
    }

    mHist2D[3].Fill(pt, nCls);
    mHist2D[4].Fill(phi, eta);

    if (sign < 0.) {
      mHist2D[5].Fill(phi, eta);
    } else {
      mHist2D[6].Fill(phi, eta);
    }
  }

  return true;
}

//______________________________________________________________________________
void Tracks::processEndOfCycle()
{
  // ===| Dividing of 1D histograms -> Ratios |===
  mHistRatio1D[0].Divide(&mHist1D[7], &mHist1D[8]);
  mHistRatio1D[1].Divide(&mHist1D[9], &mHist1D[10]);
  mHistRatio1D[2].Divide(&mHist1D[11], &mHist1D[12]);
  mHistRatio1D[3].Divide(&mHist1D[13], &mHist1D[14]);
}

//______________________________________________________________________________
void Tracks::dumpToFile(std::string_view filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.data(), "recreate"));
  for (auto& hist : mHist1D) {
    f->WriteObject(&hist, hist.GetName());
  }
  for (auto& hist : mHist2D) {
    f->WriteObject(&hist, hist.GetName());
  }
  f->Close();
}
