// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define _USE_MATH_DEFINES

#include <cmath>
#include <memory>

// root includes
#include "TFile.h"

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "TPCQC/Tracks.h"

ClassImp(o2::tpc::qc::Tracks);

using namespace o2::tpc::qc;

//______________________________________________________________________________
void Tracks::initializeHistograms()
{
  mHist1D.emplace_back("hNClustersBeforeCuts", "Number of clusters before cuts;# TPC clusters", 160, -0.5, 159.5);
  mHist1D.emplace_back("hNClustersAfterCuts", "Number of clusters after cuts;# TPC clusters", 160, -0.5, 159.5);
  mHist1D.emplace_back("hEta", "Pseudorapidity;eta", 400, -2., 2.);
  mHist1D.emplace_back("hPhiAside", "Azimuthal angle, A side;phi", 360, -M_PI, M_PI);
  mHist1D.emplace_back("hPhiCside", "Azimuthal angle, C side;phi", 360, -M_PI, M_PI);
  mHist1D.emplace_back("hPt", "Transverse momentum;p_T", 200, 0., 10.);
  mHist1D.emplace_back("hSign", "Sign of electric charge;charge sign", 3, -1.5, 1.5);

  mHist2D.emplace_back("h2DNClustersEta", "Number of clusters vs. eta;eta;# TPC clusters", 400, -2., 2., 160, -0.5, 159.5);
  mHist2D.emplace_back("h2DNClustersPhiAside", "Number of clusters vs. phi, A side ;phi;# TPC clusters", 360, -M_PI, M_PI, 160, -0.5, 159.5);
  mHist2D.emplace_back("h2DNClustersPhiCside", "Number of clusters vs. phi, C side ;phi;# TPC clusters", 360, -M_PI, M_PI, 160, -0.5, 159.5);
  mHist2D.emplace_back("h2DNClustersPt", "Number of clusters vs. p_T;p_T;# TPC clusters", 200, 0., 10., 160, -0.5, 159.5);
  mHist2D.emplace_back("h2DEtaPhi", "Tracks in eta vs. phi;phi;eta", 360, -M_PI, M_PI, 400, -2., 2.);
  mHist2D.emplace_back("h2DEtaPhiNeg", "Negative tracks in eta vs. phi;phi;eta", 360, -M_PI, M_PI, 400, -2., 2.);
  mHist2D.emplace_back("h2DEtaPhiPos", "Positive tracks in eta vs. phi;phi;eta", 360, -M_PI, M_PI, 400, -2., 2.);
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

  // ===| fill one histogram before any cuts |===
  mHist1D[0].Fill(nCls);

  // ===| cuts |===
  // hard coded cuts. Should be more configural in future
  if (nCls < 20) {
    return false;
  }

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

  return true;
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
