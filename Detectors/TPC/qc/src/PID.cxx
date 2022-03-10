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

struct binning {
  int bins;
  double min;
  double max;
};
  constexpr std::array<float, 5> xks{90.f, 108.475f, 151.7f, 188.8f, 227.65f};
  const std::vector<std::string_view> rocNames{"TPC", "IROC", "OROC1", "OROC2", "OROC3"};
  const std::vector<std::string_view> histVecNames{"hdEdxTotVspPos","hdEdxTotVspNeg","hNClsPID","hNClsSubPID","hdEdxVsPhi","hdEdxVsTgl","hdEdxVsncls","hdEdxVspBeforeCuts","hdEdxMaxVspBeforeCuts","hdEdxTotVsp", "histdEdxMax", "hdEdxTotVspMIP", "histdEdxMaxMIP", "hdEdxTotVspMIPTgl", "histdEdxMaxMIPTgl", "hdEdxTotVspMIPSnp", "hdEdxMaxMIPVsSnp", "hdEdxTotMIPVsNcl", "histdEdxMaxMIPVsNcl", "hdEdxTotVspMIPVsSec", "hdEdxMaxMIPVsSec","hMIPNclVsTgl","hMIPNclVsTglSub"};
  const std::vector<int> nclCuts{60, 30, 15, 15, 15};
  const std::vector<int> nclMax{152, 63, 34, 30, 25};
  const binning binsdEdxTot{2000, 20., 6000.};
  const binning binsdEdxMax{2000, 20., 2000.};
  int mipTot = 50;
  int mipMax = 50;
  const binning binsdEdxMIPTot{100, mipTot / 3., mipTot * 3.};
  const binning binsdEdxMIPMax{100, mipMax / 3., mipMax * 3.};
  const binning binsSec{36, 0., 36.};
  const auto bins = o2::tpc::qc::helpers::makeLogBinning(200, 0.05, 20);
  
//______________________________________________________________________________
void PID::initializeHistograms()
{
  for (const auto& keys : histVecNames) {
    mapHist2D[keys] = std::vector<TH1*>();
  }
  const auto& name = rocNames[0];
  mapHist2D["hdEdxTotVspPos"].emplace_back(new TH2F(fmt::format("hdEdxTotVsP_Pos_{}", name).data(), (fmt::format("Q_{{Tot}} positive particles {}", name) + ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), 200, bins.data(), binsdEdxTot.bins, binsdEdxTot.min, binsdEdxTot.max));
  mapHist2D["hdEdxTotVspNeg"].emplace_back(new TH2F(fmt::format("hdEdxTotVsP_Neg_{}", name).data(), (fmt::format("Q_{{Tot}} negative particles {}", name) + ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), 200, bins.data(), binsdEdxTot.bins, binsdEdxTot.min, binsdEdxTot.max));
  mapHist2D["hNClsPID"].emplace_back(new TH1F("hNClsPID", "Number of clusters (after cuts); # of clusters; counts", 160, 0, 160));
  mapHist2D["hNClsSubPID"].emplace_back(new TH1F("hNClsSubPID", "Number of clusters (after cuts); # of clusters; counts", 160, 0, 160));
  
  mapHist2D["hdEdxVsPhi"].emplace_back(new TH2F("hdEdxVsPhi", "dEdx (a.u.) vs #phi (rad); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 300, 0, 300)); 
  mapHist2D["hdEdxVsTgl"].emplace_back(new TH2F("hdEdxVsTgl", "dEdx (a.u.) vs tan#lambda; tan#lambda; dEdx (a.u.)", 60, -2, 2, 300, 0, 300)); 
  mapHist2D["hdEdxVsncls"].emplace_back(new TH2F("hdEdxVsncls", "dEdx (a.u.) vs ncls; ncls; dEdx (a.u.)", 80, 0, 160, 300, 0, 300));

  const auto logPtBinning = helpers::makeLogBinning(200, 0.05, 20);
  if (logPtBinning.size() > 0) {
    mapHist2D["hdEdxVspBeforeCuts"].emplace_back(new TH2F("hdEdxVspBeforeCuts", "dEdx (a.u.) vs p (GeV/#it{c}) (before cuts); p (GeV/#it{c}); dEdx (a.u.)", logPtBinning.size() - 1, logPtBinning.data(), 500, 0, 1000));
    mapHist2D["hdEdxMaxVspBeforeCuts"].emplace_back(new TH2F("hdEdxMaxVspBeforeCuts", "dEdx_Max (a.u.) vs p (GeV/#it{c}) (before cuts); p (GeV/#it{c}); dEdx (a.u.)", logPtBinning.size() - 1, logPtBinning.data(), 500, 0, 1000));
  }

  mapHist2D["hdEdxVsPhiMipsAside"].emplace_back(new TH2F("hdEdxVsPhiMipsAside", "dEdx (a.u.) vs #phi (rad) of MIPs (A side); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 25, 35, 60)); 
  mapHist2D["hdEdxVsPhiMipsCside"].emplace_back(new TH2F("hdEdxVsPhiMipsCside", "dEdx (a.u.) vs #phi (rad) of MIPs (C side); #phi (rad); dEdx (a.u.)", 180, 0., 2 * M_PI, 25, 35, 60)); 
  
  for (size_t idEdxType = 0; idEdxType < rocNames.size(); ++idEdxType) {
    const auto& name = rocNames[idEdxType];
    mapHist2D["hdEdxTotVsp"].emplace_back(new TH2F(fmt::format("hdEdxTotVsP_{}", name).data(), (fmt::format("Q_{{Tot}} {}", name) + ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), 200, bins.data(), binsdEdxTot.bins, binsdEdxTot.min, binsdEdxTot.max));
    mapHist2D["hdEdxMaxVsp"].emplace_back(new TH2F(fmt::format("hdEdxMaxVsP_{}", name).data(), (fmt::format("Q_{{Max}} {}", name) + ";#it{p} (GeV/#it{c});d#it{E}/d#it{x}_{Max} (arb. unit)").data(), 200, bins.data(), binsdEdxMax.bins, binsdEdxMax.min, binsdEdxMax.max));
    mapHist2D["hdEdxTotMIP"].emplace_back(new TH1F(fmt::format("hdEdxTotMIP_{}", name).data(), (fmt::format("MIP Q_{{Tot}} {}", name) + ";d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), binsdEdxMIPTot.bins, binsdEdxMIPTot.min, binsdEdxMIPTot.max));
    mapHist2D["hdEdxMaxMIP"].emplace_back(new TH1F(fmt::format("hdEdxMaxMIP_{}", name).data(), (fmt::format("MIP Q_{{Max}} {}", name) + ";d#it{E}/d#it{x}_{Max} (arb. unit)").data(), binsdEdxMIPMax.bins, binsdEdxMIPMax.min, binsdEdxMIPMax.max));
    mapHist2D["hdEdxTotMIPVsTgl"].emplace_back(new TH2F(fmt::format("hdEdxTotMIPVsTgl_{}", name).data(), (fmt::format("MIP Q_{{Tot}} {}", name) + ";#tan(#lambda);d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), 50, -2, 2, binsdEdxMIPTot.bins, binsdEdxMIPTot.min, binsdEdxMIPTot.max));
    mapHist2D["hdEdxMaxMIPVsTgl"].emplace_back(new TH2F(fmt::format("hdEdxMaxMIPVsTgl_{}", name).data(), (fmt::format("MIP Q_{{Max}} {}", name) + ";#tan(#lambda);d#it{E}/d#it{x}_{Max} (arb. unit)").data(), 50, -2, 2, binsdEdxMIPMax.bins, binsdEdxMIPMax.min, binsdEdxMIPMax.max));
    mapHist2D["hdEdxTotMIPVsSnp"].emplace_back(new TH2F(fmt::format("hdEdxTotMIPVsSnp_{}", name).data(), (fmt::format("MIP Q_{{Tot}} {}", name) + ";#sin(#phi);d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), 50, -1, 1, binsdEdxMIPTot.bins, binsdEdxMIPTot.min, binsdEdxMIPTot.max));
    mapHist2D["hdEdxMaxMIPVsSnp"].emplace_back(new TH2F(fmt::format("hdEdxMaxMIPVsSnp_{}", name).data(), (fmt::format("MIP Q_{{Max}} {}", name) + ";#sin(#phi);d#it{E}/d#it{x}_{Max} (arb. unit)").data(), 50, -1, 1, binsdEdxMIPMax.bins, binsdEdxMIPMax.min, binsdEdxMIPMax.max));
    mapHist2D["hdEdxTotMIPVsNcl"].emplace_back(new TH2F(fmt::format("hdEdxTotMIPVsNcl_{}", name).data(), (fmt::format("MIP Q_{{Tot}} {}", name) + ";N_{clusters};d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), nclMax[idEdxType], 0, nclMax[idEdxType], binsdEdxMIPTot.bins, binsdEdxMIPTot.min, binsdEdxMIPTot.max));
    mapHist2D["hdEdxMaxMIPVsNcl"].emplace_back(new TH2F(fmt::format("hdEdxMaxMIPVsNcl_{}", name).data(), (fmt::format("MIP Q_{{Max}} {}", name) + ";N_{clusters};d#it{E}/d#it{x}_{Max} (arb. unit)").data(), nclMax[idEdxType], 0, nclMax[idEdxType], binsdEdxMIPMax.bins, binsdEdxMIPMax.min, binsdEdxMIPMax.max));
    mapHist2D["hdEdxTotMIPVsSec"].emplace_back(new TH2F(fmt::format("hdEdxTotMIPVsSec_{}", name).data(), (fmt::format("MIP Q_{{Tot}} {}", name) + ";sector;d#it{E}/d#it{x}_{Tot} (arb. unit)").data(), binsSec.bins, binsSec.min, binsSec.max, binsdEdxMIPTot.bins, binsdEdxMIPTot.min, binsdEdxMIPTot.max));
    mapHist2D["hdEdxMaxMIPVsSec"].emplace_back(new TH2F(fmt::format("hdEdxMaxMIPVsSec_{}", name).data(), (fmt::format("MIP Q_{{Max}} {}", name) + ";sector;d#it{E}/d#it{x}_{Max} (arb. unit)").data(), binsSec.bins, binsSec.min, binsSec.max, binsdEdxMIPMax.bins, binsdEdxMIPMax.min, binsdEdxMIPMax.max));
 
    mapHist2D["hMIPNclVsTgl"].emplace_back(new TH2F(fmt::format("hMIPNclVsTgl_{}", name).data(), (fmt::format("rec. clusters {}", name) + ";#tan(#lambda);d#it{E}/d#it{x}_{Max} (arb. unit)").data(), 50, -2, 2, nclMax[idEdxType], 0, nclMax[idEdxType]));
    mapHist2D["hMIPNclVsTglSub"].emplace_back(new TH2F(fmt::format("hMIPNclVsTglSub_{}", name).data(), (fmt::format("rec. + sub-thrs. clusters {}", name) + ";#tan(#lambda);d#it{E}/d#it{x}_{Max} (arb. unit)").data(), 50, -2, 2, nclMax[idEdxType], 0, nclMax[idEdxType]));
 

 }
}

//______________________________________________________________________________
void PID::resetHistograms()
{
  for (const auto&pair:mapHist2D){
    for(auto hist:pair.second){
      hist->Reset();
    }
  }
}

//______________________________________________________________________________
bool PID::processTrack(const o2::tpc::TrackTPC& track)
{
  // ===| variables required for cutting and filling |===
  const auto& dEdx = track.getdEdx();
  const auto pTPC = track.getP();
  const auto tgl = track.getTgl();
  const auto snp = track.getSnp();
  const auto phi = track.getPhi();
  const auto ncl = uint8_t(track.getNClusters());
  //ncls for PID
  //const auto nclPID = static_cast<uint8_t>(dEdx.NHitsIROC + dEdx.NHitsOROC1 + dEdx.NHitsOROC2 + dEdx.NHitsOROC3);
  const auto eta = track.getEta();
  
  mapHist2D["hdEdxTotVspBeforeCuts"][0]->Fill(phi, dEdx.dEdxTotTPC);
  mapHist2D["hdEdxMaxVspBeforeCuts"][0]->Fill(pTPC, dEdx.dEdxMaxTPC);


  if (pTPC < 0.05 || pTPC > 20 || ncl < 60) {
    return true;
  } 
    const std::vector<float> dEdxTot{dEdx.dEdxTotTPC, dEdx.dEdxTotIROC, dEdx.dEdxTotOROC1, dEdx.dEdxTotOROC2, dEdx.dEdxTotOROC3};
    const std::vector<float> dEdxMax{dEdx.dEdxMaxTPC, dEdx.dEdxMaxIROC, dEdx.dEdxMaxOROC1, dEdx.dEdxMaxOROC2, dEdx.dEdxMaxOROC3};
    const std::vector<uint8_t> dEdxNcl{static_cast<uint8_t>(dEdx.NHitsIROC + dEdx.NHitsOROC1 + dEdx.NHitsOROC2 + dEdx.NHitsOROC3), dEdx.NHitsIROC, dEdx.NHitsOROC1, dEdx.NHitsOROC2, dEdx.NHitsOROC3};
    const std::vector<uint8_t> dEdxNclSub{static_cast<uint8_t>(dEdx.NHitsSubThresholdIROC + dEdx.NHitsSubThresholdOROC1 + dEdx.NHitsSubThresholdOROC2 + dEdx.NHitsSubThresholdOROC3), dEdx.NHitsSubThresholdIROC, dEdx.NHitsSubThresholdOROC1, dEdx.NHitsSubThresholdOROC2, dEdx.NHitsSubThresholdOROC3};
    mapHist2D["hdEdxVsTgl"][0]->Fill(tgl, dEdxTot[0]);
    
    if (std::abs(tgl) < 1) {

      mapHist2D["hdEdxVsPhi"][0]->Fill(phi, dEdxTot[0]);
      mapHist2D["hdEdxVsncls"][0]->Fill(ncl, dEdxTot[0]);
      mapHist2D["hNClsSubPID"][0]->Fill(dEdxNcl[0]);
      mapHist2D["hNClsSubPID"][0]->Fill(dEdxNclSub[0]);
      //inlcude both with and without..sub.

      if (track.getCharge() > 0) {
        mapHist2D["hdEdxTotVspPos"][0]->Fill(pTPC, dEdxTot[0]);
      } else {
        mapHist2D["hdEdxTotVspNeg"][0]->Fill(pTPC, dEdxTot[0]);
      }
    }
    for (size_t idEdxType = 0; idEdxType < rocNames.size(); ++idEdxType) {
      bool ok = false;
      float sec = 18.f * o2::math_utils::to02PiGen(track.getXYZGloAt(xks[idEdxType], 2, ok).Phi()) / o2::constants::math::TwoPI;
      if (track.hasCSideClusters()) {
        sec += 18.f;
      }
      if (!ok) {
        sec = -1;
      }

      if (dEdxTot[idEdxType] < 10 || dEdxTot[idEdxType] > binsdEdxTot.max || dEdxNcl[idEdxType] < nclCuts[idEdxType]) {
        continue;
      }
      if (std::abs(tgl) < 1) {
        mapHist2D["hdEdxTotVsp"][idEdxType]->Fill(pTPC, dEdxTot[idEdxType]);
        mapHist2D["hdEdxMaxVsp"][idEdxType]->Fill(pTPC, dEdxMax[idEdxType]);
      }
      // ===| cuts and  histogram filling for MIPs |===
      if (pTPC > 0.45 && pTPC < 0.55) {
        mapHist2D["hdEdxTotMIPVsTgl"][idEdxType]->Fill(tgl, dEdxTot[idEdxType]);
        mapHist2D["hdEdxMaxMIPVsTgl"][idEdxType]->Fill(tgl, dEdxMax[idEdxType]);

          if (dEdxTot[idEdxType] < 70) {
            mapHist2D["hMIPNclVsTgl"][idEdxType]->Fill(tgl, dEdxNcl[idEdxType]);
            mapHist2D["hMIPNclVsTglSub"][idEdxType]->Fill(tgl, dEdxNclSub[idEdxType]);
          }

        if (std::abs(tgl) < 1) {
          if (track.hasASideClustersOnly()) {
            mapHist2D["hdEdxVsPhiMipsAside"][0]->Fill(phi, dEdxTot[0]);
          } else if (track.hasCSideClustersOnly()) {
            mapHist2D["hdEdxVsPhiMipsCside"][0]->Fill(phi, dEdxTot[0]);
          }

          mapHist2D["hdEdxTotMIP"][idEdxType]->Fill(dEdxTot[idEdxType]);
          mapHist2D["hdEdxMaxMIP"][idEdxType]->Fill(dEdxMax[idEdxType]);

          mapHist2D["hdEdxTotMIPVsNcl"][idEdxType]->Fill(dEdxNcl[idEdxType], dEdxTot[idEdxType]);
          mapHist2D["hdEdxMaxMIPVsNcl"][idEdxType]->Fill(dEdxNcl[idEdxType], dEdxMax[idEdxType]);

          mapHist2D["hdEdxTotMIPVsSec"][idEdxType]->Fill(sec, dEdxTot[idEdxType]);
          mapHist2D["hdEdxMaxMIPVsSec"][idEdxType]->Fill(sec, dEdxMax[idEdxType]);

          mapHist2D["hdEdxTotMIPVsSnp"][idEdxType]->Fill(snp, dEdxTot[idEdxType]);
          mapHist2D["hdEdxMaxMIPVsSnp"][idEdxType]->Fill(snp, dEdxMax[idEdxType]);
        }
      }
    }

    return true;
}

//______________________________________________________________________________
void PID::dumpToFile(const std::string filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.c_str(), "recreate"));
  for (const auto& pair:mapHist2D){
    for(auto hist:pair.second){
      f->WriteObject(hist, hist->GetName());;
    }
  }
  f->Close();
}
