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

// root includes
#include "TFile.h"
#include "TAxis.h"

// o2 includes
#include "DataFormatsTPC/dEdxInfo.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "TPCQC/ExpertVis.h"
#include "TPCQC/Helpers.h"

ClassImp(o2::tpc::qc::ExpertVis);

using namespace o2::tpc::qc;

//______________________________________________________________________________
void ExpertVis::initializeHistograms()
{
  /*  => ND hist for PID:
  Dimensions: 8
  [0] p, 40 bins, [0.05,20] (log)
  [1] dEdxTot, 100 bins, [20,2000] (log)
  [2] dEdxMax, 100 bins, [1,1000] (log)
  [3] phi, 72 bins, [0,6.3]
  [4] tgl, 24 bins, [-1.2,1.2]
  [5] snp, 24 bins, [-1.2,1.2]
  [6] nclusters_pid, 40 bins, [52, 212]
  [7] sign, 3 bins, [-1,1]
  */
  const Int_t ndims_pid = 8;
  const std::string titles_pid[ndims_pid] = {"P", "dEdxTot", "dEdxMax", "Phi", "Tgl", "Snp", "NClusters", "Sign"};
  const Int_t bins_pid[ndims_pid] = {40, 100, 100, 72, 24, 24, 40, 3};
  const Double_t xmin_pid[ndims_pid] = {0.05, 20., 5., 0., -1.2, -1.2, 52., -1.5};
  const Double_t xmax_pid[ndims_pid] = {20., 2000., 1000., 6.3, 1.2, 1.2, 212., 1.5};
  mPIDND = std::make_unique<THnSparseF>("hNdPid", "Sprase Nd ExpertVis histogram for PID variables", ndims_pid, bins_pid, xmin_pid, xmax_pid);
  const auto logPBinning = helpers::makeLogBinning(40, 0.05, 20);
  const auto logdEdxTotBinning = helpers::makeLogBinning(50, 20, 2000);
  const auto logdEdxMaxBinning = helpers::makeLogBinning(50, 5, 1000);
  mPIDND->SetBinEdges(0, logPBinning.data());
  mPIDND->SetBinEdges(1, logdEdxTotBinning.data());
  mPIDND->SetBinEdges(2, logdEdxMaxBinning.data());
  for (int i = 0; i < ndims_pid; i++) {
    mPIDND->GetAxis(i)->SetTitle(titles_pid[i].c_str());
    mPIDND->GetAxis(i)->SetName(titles_pid[i].c_str());
  }

  /*  => ND hist for Tracks:
  Dimensions: 5
  [0] q/pt, 40 bins [-20,20]
  [1] dEdxTot, 50 bins, [20,2000] (log)
  [2] phi, 72 bins, [0,6.3]
  [3] tgl, 24 bins, [-1.2,1.2]
  [4] nclusters, 40 bins, [52, 212]
  */
  const Int_t ndims_tracks = 5;
  const std::string titles_tracks[ndims_tracks] = {"Q/Pt", "dEdxTot", "Phi", "Tgl", "NClusters"};
  const Int_t bins_tracks[ndims_tracks] = {40, 50, 72, 24, 40};
  const Double_t xmin_tracks[ndims_tracks] = {-20., 1., 0., -1.2, 52.};
  const Double_t xmax_tracks[ndims_tracks] = {20., 2000., 6.3, 1.2, 212.};
  mTracksND = std::make_unique<THnSparseF>("hNdTracks", "Sprase Nd ExpertVis histogram for Tracks variables", ndims_tracks, bins_tracks, xmin_tracks, xmax_tracks);
  const auto logdEdxTotBinningTracks = helpers::makeLogBinning(50, 20, 1000);
  mTracksND->SetBinEdges(1, logdEdxTotBinningTracks.data());
  for (int i = 0; i < ndims_tracks; i++) {
    mTracksND->GetAxis(i)->SetTitle(titles_tracks[i].c_str());
    mTracksND->GetAxis(i)->SetName(titles_tracks[i].c_str());
  }
}

//______________________________________________________________________________
void ExpertVis::resetHistograms()
{
  mPIDND->Reset();
  mTracksND->Reset();
}

//______________________________________________________________________________
bool ExpertVis::processTrack(const o2::tpc::TrackTPC& track)
{
  // ===| variables required for cutting and filling |===
  const Int_t ndims_pid = 8;
  const Int_t ndims_tracks = 5;
  const auto p = track.getP();
  const auto pt = track.getPt();
  const auto dEdxTot = track.getdEdx().dEdxTotTPC;
  const auto dEdxMax = track.getdEdx().dEdxMaxTPC;
  const auto phi = track.getPhi();
  const auto tgl = track.getTgl();
  const auto snp = track.getSnp();
  const auto nclusters = track.getNClusterReferences();
  const auto eta = track.getEta();
  const auto sign = track.getSign();
  const auto q2pt = track.getQ2Pt();
  const double absEta = std::abs(eta);
  const auto& dEdxAll = track.getdEdx();
  const int nclusters_pid = dEdxAll.NHitsSubThresholdIROC + dEdxAll.NHitsSubThresholdOROC1 + dEdxAll.NHitsSubThresholdOROC2 + dEdxAll.NHitsSubThresholdOROC3;

  // ===| Filling PID histogram including cuts |===
  if (absEta < 1. && nclusters_pid > 60. && dEdxTot > 20.) {
    const Double_t dataForPID[ndims_pid] = {p, dEdxTot, dEdxMax, phi, tgl, snp, nclusters_pid, sign};
    mPIDND->Fill(dataForPID);
  }

  // ===| Filling Tracks histogram including cuts |===
  if (absEta < 1. && nclusters > 60. && dEdxTot > 20.) {
    const Double_t dataForTracks[ndims_tracks] = {q2pt, dEdxTot, phi, tgl, nclusters};
    mTracksND->Fill(dataForTracks);
  }

  return true;
}

//______________________________________________________________________________
void ExpertVis::dumpToFile(std::string_view filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.data(), "recreate"));
  f->WriteObject(mPIDND.get(), mPIDND->GetName());
  f->WriteObject(mTracksND.get(), mTracksND->GetName());
  f->Close();
}
