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

//root includes
#include "TStyle.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TMathBase.h"

//o2 includes
#include "DataFormatsTPC/dEdxInfo.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "TPCQC/PID.h"
#include "TPCQC/Helpers.h"

ClassImp(o2::tpc::qc::PID);

using namespace o2::tpc::qc;

//______________________________________________________________________________
void PID::initializeHistograms()
{
  mHist1D.emplace_back("hNClusters", "Number of clusters; # of clusters; counts", 160, 0, 160);                         //| mHist1D[0]
  mHist1D.emplace_back("hdEdxTot", "; dEdxTot (a.u.); counts", 200, 0, 200);                                            //| mHist1D[1]
  mHist1D.emplace_back("hdEdxMax", "; dEdxMax (a.u.); counts", 200, 0, 200);                                            //| mHist1D[2]
  mHist1D.emplace_back("hPhi", "; #phi (rad); counts", 180, 0., 2 * M_PI);                                              //| mHist1D[3]
  mHist1D.emplace_back("hTgl", "; tan#lambda; counts", 60, -2, 2);                                                      //| mHist1D[4]
  mHist1D.emplace_back("hSnp", "; sin p; counts", 60, -2, 2);                                                           //| mHist1D[5]
  mHist1D.emplace_back("hdEdxMips", "dEdx (a.u.) of MIPs; dEdx of MIPs (a.u.); counts", 25, 35, 60);                    //| mHist1D[6]
  mHist1D.emplace_back("hdEdxEles", "dEdx (a.u.) of electrons; dEdx (a.u.); counts", 30, 70, 100);                      //| mHist1D[7]
  mHist1D.emplace_back("hNClustersBeforeCuts", "Number of clusters (before cuts); # of clusters; counts", 160, 0, 160); //| mHist1D[8]

  mHist2D.emplace_back("hdEdxVsPhi", "dEdx (a.u.) vs #phi (rad); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 300, 0, 300); //| mHist2D[0]
  mHist2D.emplace_back("hdEdxVsTgl", "dEdx (a.u.) vs tan#lambda; tan#lambda; dEdx (a.u.)", 60, -2, 2, 300, 0, 300);         //| mHist2D[1]
  mHist2D.emplace_back("hdEdxVsncls", "dEdx (a.u.) vs ncls; ncls; dEdx (a.u.)", 80, 0, 160, 300, 0, 300);                   //| mHist2D[2]

  const auto logPtBinning = helpers::makeLogBinning(300, 0.05, 10);
  if (logPtBinning.size() > 0) {
    mHist2D.emplace_back("hdEdxVsp", "dEdx (a.u.) vs p (GeV/#it{c}); p (GeV/#it{c}); dEdx (a.u.)", logPtBinning.size() - 1, logPtBinning.data(), 500, 0, 500);                         //| mHist2D[3]
    mHist2D.emplace_back("hdEdxVspBeforeCuts", "dEdx (a.u.) vs p (GeV/#it{c}) (before cuts); p (GeV/#it{c}); dEdx (a.u.)", logPtBinning.size() - 1, logPtBinning.data(), 500, 0, 500); //| mHist2D[4]
  }

  mHist2D.emplace_back("hdEdxVsPhiMipsAside", "dEdx (a.u.) vs #phi (rad) of MIPs (A side); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 25, 35, 60);       //| mHist2D[5]
  mHist2D.emplace_back("hdEdxVsPhiMipsCside", "dEdx (a.u.) vs #phi (rad) of MIPs (C side); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 25, 35, 60);       //| mHist2D[6]
  mHist2D.emplace_back("hdEdxVsPhiElesAside", "dEdx (a.u.) vs #phi (rad) of electrons (A side); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 30, 70, 100); //| mHist2D[7]
  mHist2D.emplace_back("hdEdxVsPhiElesCside", "dEdx (a.u.) vs #phi (rad) of electrons (C side); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 30, 70, 100); //| mHist2D[8]
}

//______________________________________________________________________________
void PID::resetHistograms()
{
  for (auto& hist : mHist1D) {
    hist.Reset();
  }
  for (auto& hist : mHist2D) {
    hist.Reset();
  }
}

//______________________________________________________________________________
bool PID::processTrack(const o2::tpc::TrackTPC& track)
{
  // ===| variables required for cutting and filling |===
  const auto p = track.getP();
  const auto dEdxTot = track.getdEdx().dEdxTotTPC;
  const auto dEdxMax = track.getdEdx().dEdxMaxTPC;
  const auto phi = track.getPhi();
  const auto tgl = track.getTgl();
  const auto snp = track.getSnp();
  const auto nclusters = track.getNClusterReferences();
  const auto eta = track.getEta();

  double absEta = TMath::Abs(eta);

  // ===| histogram filling before cuts |===
  mHist1D[8].Fill(nclusters);
  mHist2D[4].Fill(p, dEdxTot);

  // ===| histogram filling including cuts |===
  if (absEta < 1. && nclusters > 60 && dEdxTot > 20) {
    mHist1D[0].Fill(nclusters);
    mHist1D[1].Fill(dEdxTot);
    mHist1D[2].Fill(dEdxMax);
    mHist1D[3].Fill(phi);
    mHist1D[4].Fill(tgl);
    mHist1D[5].Fill(snp);

    mHist2D[0].Fill(phi, dEdxTot);
    mHist2D[1].Fill(tgl, dEdxTot);
    mHist2D[2].Fill(nclusters, dEdxTot);
    mHist2D[3].Fill(p, dEdxTot);
  }

  // ===| cuts and  histogram filling for MIPs |===
  if (p > 0.4 && p < 0.55 && absEta < 1. && nclusters > 80 && dEdxTot > 35 && dEdxTot < 60) {
    mHist1D[6].Fill(dEdxTot);
    if (eta > 0.) {
      mHist2D[5].Fill(phi, dEdxTot);
    } else {
      mHist2D[6].Fill(phi, dEdxTot);
    }
  }

  // ===| cuts and  histogram filling for electrons |===
  if (p > 0.32 && p < 0.38 && absEta < 1. && nclusters > 80 && dEdxTot > 70 && dEdxTot < 100) {
    mHist1D[7].Fill(dEdxTot);
    if (eta > 0.) {
      mHist2D[7].Fill(phi, dEdxTot);
    } else {
      mHist2D[8].Fill(phi, dEdxTot);
    }
  }

  return true;
}

//______________________________________________________________________________
void PID::dumpToFile(const std::string filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.c_str(), "recreate"));
  for (auto& hist : mHist1D) {
    f->WriteObject(&hist, hist.GetName());
  }
  for (auto& hist : mHist2D) {
    f->WriteObject(&hist, hist.GetName());
  }
  f->Close();
}
