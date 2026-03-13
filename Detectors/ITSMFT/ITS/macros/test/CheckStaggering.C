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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <cmath>
#include <vector>
#include <filesystem>
#include <vector>

#include <TGeoManager.h>
#include <TFile.h>
#include <TTree.h>
#include <TF1.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPad.h>
#include <TLatex.h>
#include <TProfile.h>
#include <TStyle.h>
#include <TLine.h>
#include <TLorentzVector.h>
#include <TPaveText.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooGaussian.h>
#include <RooAddPdf.h>
#include "RooVoigtian.h"
#include <RooChebychev.h>
#include <RooPlot.h>
#include <RooFitResult.h>
#include <RooHist.h>
#include <RooFit.h>

#include "DetectorsBase/Propagator.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITS/Vertex.h"
#include "DataFormatsITS/TimeEstBC.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonConstants/LHCConstants.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "Framework/Logger.h"
#include "DetectorsVertexing/SVertexHypothesis.h"
#include "DCAFitter/DCAFitterN.h"
#endif
using namespace RooFit;

constexpr const char* tracFile = "o2trac_its.root";
constexpr const char* clsFile = "o2clus_its.root";

struct PairCandidate {
  int posIdx;
  int negIdx;
  double dca;
  double mass;
};

std::vector<std::filesystem::path> findDirs(const std::string&);
void fitK0(TH1D*, int);
void fitPhiMeson(TH1D*, int);

void CheckStaggering(int runNumber, int max = -1, const std::string& dir = "")
{
  gStyle->SetOptStat(0);
  auto dirs = findDirs(dir);
  LOGP(info, "Will iterate over {} input dirs", dirs.size());
  if (dirs.empty()) {
    LOGP(error, "No input found");
    return;
  }
  if (max > 0 && (int)dirs.size() > max) {
    dirs.resize(max);
    LOGP(info, "restricting to {} dirs", max);
  }

  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setURL("https://alice-ccdb.cern.ch");
  auto runDuration = ccdbmgr.getRunDuration(runNumber);
  auto tRun = runDuration.first + (runDuration.second - runDuration.first) / 2; // time stamp for the middle of the run duration
  ccdbmgr.setTimestamp(tRun);
  LOGP(info, "Run {} has TS {}", runNumber, tRun);
  auto geoAligned = ccdbmgr.get<TGeoManager>("GLO/Config/GeometryAligned");
  auto magField = ccdbmgr.get<o2::parameters::GRPMagField>("GLO/Config/GRPMagField");
  auto grpLHC = ccdbmgr.get<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF");
  auto bcFill = grpLHC->getBunchFilling();
  bcFill.print(-1);
  const o2::base::MatLayerCylSet* matLut = o2::base::MatLayerCylSet::rectifyPtrFromFile(ccdbmgr.get<o2::base::MatLayerCylSet>("GLO/Param/MatLUT"));
  o2::base::Propagator::initFieldFromGRP(magField);
  auto prop = o2::base::Propagator::Instance();
  prop->setMatLUT(matLut);
  const float bz = prop->getNominalBz();

  auto hNTrkCls = new TH1D("hNTrkCls", "Number of cluster per track;nCls;entries", 4, 3.5, 7.5);
  std::array<TH1*, 5> hTrkTS;
  for (int i{0}; i < 5; ++i) {
    hTrkTS[i] = new TH1D(Form("hTrkTS_%d", i), Form("track time t0 (%s);t0 (BC)", i == 0 ? "all" : Form("NCls=%d", 3 + i)), o2::constants::lhc::LHCMaxBunches, 0, o2::constants::lhc::LHCMaxBunches);
  }
  auto hTrkTSE = new TH1D("hTrkTSE", "assigned time errors; tE (BC)", 198, -0.5, 198 - 0.5);

  // K0 && Phi-Meson
  const float mMinITSPt{0.3};
  //
  const int phiMinNCls{7};
  const float phiMaxDCAR2MVTX{0.1}; // max distance to mean vtx
  auto hPhiMeson = new TH1D("hPhiMeson", "#phi meson;mass (GeV/c^{2})", 100, 0.96, 1.1);

  const int mK0MinNCls{7};
  const float mK0minCosPAXYMeanVertex = 0.98;
  const float mK0MaxDCAXY2ToMeanVertex = 0.2;
  const float mK0MaxTgl2V0 = 1;
  const float mK0MinPt2V0 = 0.3;
  const float mK0MinCosPA = 0.999;
  o2::vertexing::DCAFitterN<2> k0Ft;
  k0Ft.setBz(bz);
  k0Ft.setPropagateToPCA(true); // After finding the vertex, propagate tracks to the DCA. This is default anyway
  k0Ft.setMaxR(30);
  k0Ft.setMaxDZIni(0.1);
  k0Ft.setMaxDXYIni(0.1);
  k0Ft.setMinParamChange(1e-3);
  k0Ft.setMinRelChi2Change(0.9);
  k0Ft.setMaxChi2(5);
  k0Ft.setUseAbsDCA(true);
  auto hK0 = new TH1D("hK0Meson", "K0;mass (GeV/c^{2})", 100, 0.4, 0.6);
  o2::vertexing::SVertexHypothesis k0Hyp;
  const float k0Par[] = {0., 20, 0., 5.0, 0.0, 1.09004e-03, 2.62291e-04, 8.93179e-03, 2.83121};
  k0Hyp.set(o2::track::PID::K0, o2::track::PID::Pion, o2::track::PID::Pion, k0Par, bz);

  auto hVtxXY = new TH2F("hVtxXY", "seeding vertices XY", 200, -0.3, 0.3, 200, -0.3, 0.3);
  auto hVtxZ = new TH1F("hVtxZ", "seeding vertices Z", 200, -16, 16);
  auto hVtxNCont = new TH1F("hVtxNCont", "seeding vertices contributors", 100, 0, 100);
  auto hVtxZNCont = new TProfile("hVtxZNCont", "seeding vertices z-contributors", 200, -16, 16);
  auto hVtxCls = new TProfile("hVtxCls", ";Cls/TF;Cls/Vtx", 400, 20000, 60000);
  auto hVtxTS = new TH1D("hVtxTS", "vtx time t0;t0 (BC)", o2::constants::lhc::LHCMaxBunches, 0, o2::constants::lhc::LHCMaxBunches);

  const float minVtxWeight{5};
  float meanVtxWeight{0};
  o2::dataformats::VertexBase meanVtx;
  auto accountVtx = [&](o2::its::Vertex const& vtx) {
    const float w = vtx.getNContributors();
    if (w >= minVtxWeight) {
      meanVtx.setX((meanVtx.getX() * meanVtxWeight + vtx.getX() * w) / (meanVtxWeight + w));
      meanVtx.setY((meanVtx.getY() * meanVtxWeight + vtx.getY() * w) / (meanVtxWeight + w));
      meanVtxWeight += w;
    }
  };

  std::vector<o2::its::TrackITS> trkArr, *trkArrPtr{&trkArr};
  std::vector<o2::its::Vertex> vtxArr, *vtxArrPtr{&vtxArr};
  std::array<std::vector<o2::itsmft::CompClusterExt>*, 7> clsArr{nullptr};
  for (const auto& d : dirs) {
    LOGP(info, "Entering {}", d.c_str());
    auto fTrks = TFile::Open((d / tracFile).c_str());
    auto fCls = TFile::Open((d / clsFile).c_str());
    if (!fTrks || !fCls || fTrks->IsZombie() || fCls->IsZombie()) {
      LOGP(error, "Could not open files");
      continue;
    }
    auto tTrks = fTrks->Get<TTree>("o2sim");
    auto tCls = fCls->Get<TTree>("o2sim");
    if (!tTrks || !tCls) {
      LOGP(error, "Could not retrieve trees");
      continue;
    }

    tTrks->SetBranchAddress("ITSTrack", &trkArrPtr);
    tTrks->SetBranchAddress("Vertices", &vtxArrPtr);
    for (int i{0}; i < 7; ++i) {
      tCls->SetBranchAddress(Form("ITSClusterComp_%d", i), &clsArr[i]);
    }

    for (int iTF{0}; tTrks->LoadTree(iTF) >= 0; ++iTF) {
      tTrks->GetEntry(iTF);
      tCls->GetEntry(iTF);

      size_t ncls = 0;
      for (int i{0}; i < 7; ++i) {
        ncls += clsArr[i]->size();
      }
      LOGP(info, "\tTF:{:03} {} trks {} vtx {} cls", iTF, trkArr.size(), vtxArr.size(), ncls);

      // for each TF built pool of positive and negaitve tracks
      std::vector<const o2::its::TrackITS*> posPool, negPool;

      for (const auto& trk : trkArr) {
        hNTrkCls->Fill(trk.getNClusters());
        hTrkTS[0]->Fill(trk.getTimeStamp().getTimeStamp() % o2::constants::lhc::LHCMaxBunches);
        hTrkTS[trk.getNClusters() - 3]->Fill(trk.getTimeStamp().getTimeStamp() % o2::constants::lhc::LHCMaxBunches);
        hTrkTSE->Fill(trk.getTimeStamp().getTimeStampError());

        if (trk.getPt() > mMinITSPt) {
          if (trk.getCharge() > 0) {
            posPool.push_back(&trk);
          } else {
            negPool.push_back(&trk);
          }
        }
      }

      for (const auto& vtx : vtxArr) {
        hVtxXY->Fill(vtx.getX(), vtx.getY());
        hVtxZ->Fill(vtx.getZ());
        hVtxNCont->Fill(vtx.getNContributors());
        hVtxZNCont->Fill(vtx.getZ(), vtx.getNContributors());
        hVtxTS->Fill(vtx.getTimeStamp().getTimeStamp());
        accountVtx(vtx);
      }
      hVtxCls->Fill(ncls, (float)ncls / (float)vtxArr.size());

      std::vector<PairCandidate> k0Cands;
      for (int iPos{0}; iPos < (int)posPool.size(); ++iPos) {
        const auto pos = posPool[iPos];
        auto posTS = o2::its::TimeEstBC(pos->getTimeStamp().getTimeStamp(), pos->getTimeStamp().getTimeStampError());
        for (int iNeg{0}; iNeg < (int)negPool.size(); ++iNeg) {
          const auto neg = negPool[iNeg];
          auto negTS = o2::its::TimeEstBC(neg->getTimeStamp().getTimeStamp(), neg->getTimeStamp().getTimeStampError());
          if (!posTS.isCompatible(negTS)) {
            continue;
            // } else if (posTS.getTimeStamp() + posTS.getTimeStampError() < negTS.getTimeStamp() - negTS.getTimeStampError()) {
            //   break;
          }

          // phi-meson
          if (pos->getNClusters() >= phiMinNCls && neg->getNClusters() >= phiMinNCls) {
            o2::dataformats::DCA posDCA, negDCA;
            o2::track::TrackParCov posPhi = *pos;
            posPhi.setPID(o2::track::PID::Kaon);
            o2::track::TrackParCov negPhi = *neg;
            negPhi.setPID(o2::track::PID::Kaon);
            if (prop->propagateToDCA(meanVtx, posPhi, bz, 2.0, o2::base::Propagator::MatCorrType::USEMatCorrLUT, &posDCA) && prop->propagateToDCA(meanVtx, negPhi, bz, 2.0, o2::base::Propagator::MatCorrType::USEMatCorrLUT, &negDCA)) {
              if (posDCA.getR2() < phiMaxDCAR2MVTX && negDCA.getR2() < phiMaxDCAR2MVTX) {
                std::array<float, 3> pP{}, pN{};
                posPhi.getPxPyPzGlo(pP);
                negPhi.getPxPyPzGlo(pN);
                TLorentzVector p1, p2;
                p1.SetXYZM(pP[0], pP[1], pP[2], posPhi.getPID().getMass());
                p2.SetXYZM(pN[0], pN[1], pN[2], negPhi.getPID().getMass());
                TLorentzVector mother = p1 + p2;
                double mass = mother.M();
                hPhiMeson->Fill(mass);
              }
            }
          }
          // K0
          if (pos->getNClusters() >= mK0MinNCls && neg->getNClusters() >= mK0MinNCls) {
            o2::track::TrackParCov posPion = *pos;
            posPion.setPID(o2::track::PID::Pion);
            o2::track::TrackParCov negPion = *neg;
            negPion.setPID(o2::track::PID::Pion);
            int ncand = k0Ft.process(posPion, negPion);
            const int cand = 0;
            if (ncand) {
              const auto& v0XYZ = k0Ft.getPCACandidate();
              float dxv0 = v0XYZ[0] - meanVtx.getX(), dyv0 = v0XYZ[1] - meanVtx.getY(), r2v0 = dxv0 * dxv0 + dyv0 * dyv0;
              if (!k0Ft.isPropagateTracksToVertexDone(cand) && !k0Ft.propagateTracksToVertex(cand)) {
                continue;
              }
              const auto& trPProp = k0Ft.getTrack(0, cand);
              const auto& trNProp = k0Ft.getTrack(1, cand);
              std::array<float, 3> pP{}, pN{};
              trPProp.getPxPyPzGlo(pP);
              trNProp.getPxPyPzGlo(pN);
              // estimate DCA of neutral V0 track to beamline: straight line with parametric equation
              // x = X0 + pV0[0]*t, y = Y0 + pV0[1]*t reaches DCA to beamline (Xv, Yv) at
              // t = -[ (x0-Xv)*pV0[0] + (y0-Yv)*pV0[1]) ] / ( pT(pV0)^2 )
              // Similar equation for 3D distance involving pV0[2]
              std::array<float, 3> pV0 = {pP[0] + pN[0], pP[1] + pN[1], pP[2] + pN[2]};
              float pt2V0 = pV0[0] * pV0[0] + pV0[1] * pV0[1], prodXYv0 = dxv0 * pV0[0] + dyv0 * pV0[1], tDCAXY = prodXYv0 / pt2V0;
              if (pt2V0 < mK0MinPt2V0) { // pt cut
                continue;
              }
              if (pV0[2] * pV0[2] / pt2V0 > mK0MaxTgl2V0) { // tgLambda cut
                continue;
              }
              float dcaX = dxv0 - pV0[0] * tDCAXY, dcaY = dyv0 - pV0[1] * tDCAXY, dca2 = dcaX * dcaX + dcaY * dcaY;
              float cosPAXY = prodXYv0 / std::sqrt(r2v0 * pt2V0);
              if (dca2 > mK0MaxDCAXY2ToMeanVertex || cosPAXY < mK0minCosPAXYMeanVertex) {
                continue;
              }
              float p2V0 = pt2V0 + pV0[2] * pV0[2], ptV0 = std::sqrt(pt2V0);
              float p2Pos = pP[0] * pP[0] + pP[1] * pP[1] + pP[2] * pP[2], p2Neg = pN[0] * pN[0] + pN[1] * pN[1] + pN[2] * pN[2];
              if (!k0Hyp.check(p2Pos, p2Neg, p2V0, ptV0)) {
                continue;
              }

              float bestCosPA = mK0MinCosPA;
              bool candFound = false;
              for (const auto& vtx : vtxArr) {
                if (vtx.getNContributors() > minVtxWeight) {
                  if (posTS.isCompatible(vtx.getTimeStamp()) && negTS.isCompatible(vtx.getTimeStamp())) {
                    float dx = v0XYZ[0] - vtx.getX(), dy = v0XYZ[1] - vtx.getY(), dz = v0XYZ[2] - vtx.getZ(), prodXYZv0 = dx * pV0[0] + dy * pV0[1] + dz * pV0[2];
                    float cosPA = prodXYZv0 / std::sqrt((dx * dx + dy * dy + dz * dz) * p2V0);
                    if (cosPA > bestCosPA) {
                      bestCosPA = cosPA;
                      candFound = true;
                    }
                  }
                }
              }
              if (candFound) {
                TLorentzVector p1, p2;
                p1.SetXYZM(pP[0], pP[1], pP[2], posPion.getPID().getMass());
                p2.SetXYZM(pN[0], pN[1], pN[2], negPion.getPID().getMass());
                TLorentzVector mother = p1 + p2;
                double mass = mother.M();
                k0Cands.emplace_back(iPos, iNeg, k0Ft.getChi2AtPCACandidate(cand), mass);
              }
            }
          }
        }
      }

      // disambiguiate candidates by using the smallest DCA one
      std::sort(k0Cands.begin(), k0Cands.end(), [](const auto& a, const auto& b) { return a.dca < b.dca; });
      std::vector<bool> posUsed(posPool.size(), false);
      std::vector<bool> negUsed(negPool.size(), false);
      for (const auto& c : k0Cands) {
        if (!posUsed[c.posIdx] && !negUsed[c.negIdx]) {
          posUsed[c.posIdx] = true;
          negUsed[c.negIdx] = true;
          hK0->Fill(c.mass);
        }
      }
    }

    fTrks->Close();
    fCls->Close();
  }

  auto drawBCPattern = [&]() {
    gPad->Update();
    // draw BC pattern
    double ymin = gPad->GetUymin();
    double ymax = gPad->GetUymax();
    auto interactingBC = bcFill.getPattern();
    TLine* lastLine{nullptr};
    for (int iBC{0}; iBC < (int)interactingBC.size(); ++iBC) {
      if (interactingBC.test(iBC)) {
        TLine* line = new TLine(iBC, ymin, iBC, ymax);
        line->SetLineColor(kRed);
        line->SetLineWidth(1);
        line->SetLineStyle(kDashed);
        line->Draw();
        lastLine = line;
      }
    }
    return lastLine;
  };

  fitK0(hK0, runNumber);
  fitPhiMeson(hPhiMeson, runNumber);
  {
    auto c = new TCanvas();
    hNTrkCls->Draw();
    c->Draw();
    c->SaveAs(Form("trk_%d.pdf", runNumber));
  }
  {
    auto c = new TCanvas();
    c->Divide(3, 2);
    for (int i{0}; i < 5; ++i) {
      c->cd(i + 1);
      hTrkTS[i]->Draw();
      auto lastLine = drawBCPattern();
      if (i == 0) {
        auto leg = new TLegend(0.6, 0.6, 0.9, 0.9);
        leg->AddEntry(lastLine, "Interacting BCs", "l");
        leg->AddEntry(hTrkTS[i], "Track timestamp");
        leg->Draw();
      }
    }
    c->cd(6);
    hTrkTSE->Draw();
    c->Draw();
    c->SaveAs(Form("time_%d.pdf", runNumber));
  }
  {
    auto c = new TCanvas();
    c->Divide(2, 1);
    {
      c->cd(1);
      hPhiMeson->Draw();
      gPad->Update();
      double ymin = gPad->GetUymin();
      double ymax = gPad->GetUymax();
      const float mass = 1.019461;
      TLine* line = new TLine(mass, ymin, mass, ymax);
      line->SetLineColor(kRed);
      line->SetLineWidth(1);
      line->SetLineStyle(kDashed);
      line->Draw();
    }
    {
      c->cd(2);
      hK0->Draw();
      gPad->Update();
      double ymin = gPad->GetUymin();
      double ymax = gPad->GetUymax();
      const float mass = 0.497611;
      TLine* line = new TLine(mass, ymin, mass, ymax);
      line->SetLineColor(kRed);
      line->SetLineWidth(1);
      line->SetLineStyle(kDashed);
      line->Draw();
    }
    c->Draw();
    c->SaveAs(Form("mass_%d.pdf", runNumber));
  }
  {
    auto c = new TCanvas();
    c->Divide(3, 2);
    {
      c->cd(1);
      hVtxXY->Draw("col");
    }
    {
      c->cd(2);
      hVtxZ->Draw();
    }
    {
      c->cd(3);
      hVtxNCont->Draw();
    }
    {
      c->cd(4);
      hVtxZNCont->Draw();
    }
    {
      c->cd(5);
      hVtxCls->Draw();
    }
    {
      c->cd(6);
      hVtxTS->Draw();
      auto lastLine = drawBCPattern();
      auto leg = new TLegend(0.6, 0.6, 0.9, 0.9);
      leg->AddEntry(lastLine, "Interacting BCs", "l");
      leg->AddEntry(hVtxTS, "Track timestamp");
      leg->Draw();
    }
    c->Draw();
    c->SaveAs(Form("vertex_%d.pdf", runNumber));
  }
}

std::vector<std::filesystem::path> findDirs(const std::string& roots)
{
  std::filesystem::path root;
  if (roots.empty()) {
    root = std::filesystem::current_path();
  } else {
    root = roots;
  }
  namespace fs = std::filesystem;
  std::vector<fs::path> result;
  auto has_files = [](const fs::path& dir) {
    auto s = dir / tracFile;
    return fs::exists(dir / tracFile) && fs::exists(dir / clsFile) &&
           fs::is_regular_file(dir / tracFile) && fs::is_regular_file(dir / clsFile);
  };
  if (fs::is_directory(root) && has_files(root)) {
    result.push_back(root);
    return result;
  }
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const fs::path dir = entry.path();
    if (has_files(dir)) {
      result.push_back(dir);
    }
  }
  return result;
}

void fitK0(TH1D* h, int runNumber)
{
  RooRealVar mass("mass", "K^{0} mass (GeV/c^{2})", 0.46, 0.54);
  RooDataHist data("data", "data", RooArgList(mass), Import(*h));
  RooRealVar mean("mean", "mean", 0.4976, 0.490, 0.503);
  RooRealVar sigma1("sigma1", "core sigma", 0.003, 0.0005, 0.02);
  RooRealVar sigma2("sigma2", "tail sigma", 0.006, 0.001, 0.03);
  RooGaussian gauss1("gauss1", "gauss1", mass, mean, sigma1);
  RooGaussian gauss2("gauss2", "gauss2", mass, mean, sigma2);
  RooRealVar frac2("frac2", "fraction in second gaussian", 0.15, 0.0, 1.0);
  RooAddPdf signal("signal", "signal PDF", RooArgList(gauss1, gauss2), RooArgList(frac2));
  RooRealVar c1("c1", "cheby coeff1", 0.0, -5.0, 5.0);
  RooChebychev bkg("bkg", "background", mass, RooArgList(c1));
  double totalIntegral = h->Integral();
  RooRealVar nsig("nsig", "signal yield", totalIntegral * 0.4, 0., totalIntegral * 10.0);
  RooRealVar nbkg("nbkg", "background yield", totalIntegral * 0.6, 0., totalIntegral * 10.0);
  RooAddPdf model("model", "sig + bkg", RooArgList(signal, bkg), RooArgList(nsig, nbkg));
  RooFitResult* fitRes = model.fitTo(data,
                                     Extended(true),
                                     Save(true),
                                     Strategy(1),
                                     Hesse(true),
                                     Minos(false),
                                     PrintLevel(-1));

  TCanvas* c = new TCanvas("c", "K0 fit", 800, 800);
  c->cd();
  TPad* pad1 = new TPad("pad1", "pad1", 0, 0.30, 1, 1.0);
  TPad* pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.30);
  pad1->SetBottomMargin(0.0);
  pad2->SetTopMargin(0.0);
  pad2->SetBottomMargin(0.30);
  pad1->Draw();
  pad2->Draw();

  pad1->cd();
  RooPlot* frame = mass.frame(Title("K^{0}_{S} mass fit"));
  data.plotOn(frame, Name("data"));
  model.plotOn(frame, Name("model"));
  // draw background component
  model.plotOn(frame, Components("bkg"), LineStyle(kDashed), LineColor(kRed), Name("bkg"));
  // draw total signal component
  model.plotOn(frame, Components("signal"), LineStyle(kDashed), LineColor(kBlue), Name("signal"));
  frame->GetYaxis()->SetTitle("Events / bin");
  frame->GetXaxis()->SetLabelSize(0.05);
  frame->GetXaxis()->SetTitleSize(0.06);
  frame->GetYaxis()->SetTitleSize(0.05);
  frame->Draw();

  TLegend leg(0.60, 0.65, 0.88, 0.88);
  leg.SetBorderSize(0);
  leg.AddEntry((TObject*)0, Form("mean = %.3f #pm %.3f GeV/c^{2}", mean.getVal(), mean.getError()), "");
  leg.AddEntry((TObject*)0, Form("sigma_{1} = %.3f #pm %.3f GeV/c^{2}", sigma1.getVal(), sigma1.getError()), "");
  leg.AddEntry((TObject*)0, Form("sigma_{2} = %.3f #pm %.3f GeV/c^{2}", sigma2.getVal(), sigma2.getError()), "");
  leg.AddEntry((TObject*)0, Form("signal yield = %.0f #pm %.0f", nsig.getVal(), nsig.getError()), "");
  leg.Draw();

  int npar = fitRes->floatParsFinal().getSize();
  double chi2ndf = frame->chiSquare(npar);
  TPaveText* pt = new TPaveText(0.15, 0.75, 0.45, 0.88, "NDC");
  pt->SetFillStyle(0);
  pt->SetBorderSize(0);
  pt->AddText(Form("#chi^{2}/ndf = %.3g", chi2ndf));
  pt->Draw();

  pad2->cd();
  RooPlot* frame_pull = mass.frame(Title("Pull (data - fit) / uncertainty"));
  RooHist* hpull = frame->pullHist();
  frame_pull->addPlotable(hpull, "P");
  frame_pull->GetXaxis()->SetTitle("mass (GeV/c^{2})");
  frame_pull->GetXaxis()->SetTitleSize(0.12);
  frame_pull->GetXaxis()->SetLabelSize(0.10);
  frame_pull->GetYaxis()->SetLabelSize(0.10);
  frame_pull->GetYaxis()->SetTitleSize(0.12);
  frame_pull->Draw();
  frame_pull->GetYaxis()->SetRangeUser(-5, 5);
  TLine l(frame_pull->GetXaxis()->GetXmin(), 0, frame_pull->GetXaxis()->GetXmax(), 0);
  l.SetLineStyle(kDashed);
  l.Draw();
  c->SaveAs(Form("k0_%d.pdf", runNumber));
}

void fitPhiMeson(TH1D* h, int runNumber)
{
  double hxmin = h->GetXaxis()->GetXmin();
  double hxmax = h->GetXaxis()->GetXmax();
  int nbins = h->GetNbinsX();
  double sig_low = 1.015;
  double sig_high = 1.025;
  TH1D* h_sb = (TH1D*)h->Clone(Form("%s_sidebands", h->GetName()));
  for (int ib = 1; ib <= nbins; ++ib) {
    double x = h_sb->GetBinCenter(ib);
    if (x > sig_low && x < sig_high) {
      h_sb->SetBinContent(ib, 0.0);
      h_sb->SetBinError(ib, 0.0);
    }
  }
  RooRealVar mass("mass", "m_{K^{+}K^{-}} (GeV/c^{2})", hxmin, hxmax);
  RooDataHist data("data", "data", RooArgList(mass), Import(*h));
  RooDataHist data_sb("data_sb", "sideband data", RooArgList(mass), Import(*h_sb));
  RooRealVar mean("mean", "phi mass", 1.019, 1.015, 1.025);
  RooRealVar gamma("gamma", "natural width (GeV)", 0.00426);
  gamma.setConstant(true);
  RooRealVar sigma("sigma", "detector resolution (GeV)", 0.001, 0.0002, 0.01);
  RooVoigtian signal("signal", "Voigtian(signal)", mass, mean, gamma, sigma);
  RooRealVar c1("c1", "cheby coeff1", 0.0, -5.0, 5.0);
  RooRealVar c2("c2", "cheby coeff2", 0.0, -5.0, 5.0);
  RooChebychev bkg("bkg", "background", mass, RooArgList(c1, c2));
  RooFitResult* fbkg = bkg.fitTo(data_sb,
                                 Range(hmin, hxmax),
                                 Save(true),
                                 Verbose(false),
                                 PrintLevel(-1));

  if (!fbkg) {
    LOGP(error, "Warning: background sideband fit failed to return RooFitResult.");
  }
  double totalIntegral = h->Integral();
  RooRealVar nsig("nsig", "signal yield", totalIntegral * 0.05, 0., totalIntegral * 10.0); // small initial guess
  RooRealVar nbkg("nbkg", "background yield", totalIntegral * 0.95, 0., totalIntegral * 10.0);
  RooAddPdf model("model", "sig + bkg", RooArgList(signal, bkg), RooArgList(nsig, nbkg));
  RooFitResult* fitRes = model.fitTo(data,
                                     Extended(true),
                                     Save(true),
                                     Strategy(1),
                                     Hesse(true),
                                     Minos(false),
                                     Verbose(true),
                                     PrintLevel(-1));

  if (!fitRes) {
    LOGP(error, "fitPhiMeson: fitTo returned null RooFitResult.");
  }
  TCanvas* c = new TCanvas("c_phi", "Phi mass fit", 800, 800);
  c->cd();
  TPad* pad1 = new TPad("pad1", "pad1", 0, 0.30, 1, 1.0);
  TPad* pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.30);
  pad1->SetBottomMargin(0.0);
  pad2->SetTopMargin(0.0);
  pad2->SetBottomMargin(0.30);
  pad1->Draw();
  pad2->Draw();
  pad1->cd();

  RooPlot* frame = mass.frame(Title("Phi (#phi) meson fit"));
  frame->GetXaxis()->SetTitle("m_{K^{+}K^{-}} (GeV/c^{2})");
  frame->GetYaxis()->SetTitle("Events / bin");
  data.plotOn(frame, Binning(nbins), Name("data"));
  model.plotOn(frame, Name("model"));
  model.plotOn(frame, Components("bkg"), LineStyle(kDashed), LineColor(kRed), Name("bkg"));
  model.plotOn(frame, Components("signal"), LineStyle(kSolid), LineColor(kBlue), Name("signal"));
  frame->Draw();

  TLegend leg(0.55, 0.60, 0.88, 0.88);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.AddEntry("data", "Data", "lep");
  leg.AddEntry("model", "Total fit", "l");
  leg.AddEntry("signal", "Signal (Voigtian)", "l");
  leg.AddEntry("bkg", "Background (Chebychev)", "l");
  leg.Draw();

  TPaveText* pv = new TPaveText(0.15, 0.60, 0.45, 0.88, "NDC");
  pv->SetFillStyle(0);
  pv->SetBorderSize(0);

  pv->AddText(Form("m = %.6g #pm %.6g GeV", mean.getVal(), mean.getError()));
  pv->AddText(Form("#sigma_{res} = %.6g #pm %.6g GeV", sigma.getVal(), sigma.getError()));
  pv->AddText(Form("width (fixed) = %.6g GeV", gamma.getVal()));
  pv->AddText(Form("Signal yield = %.1f #pm %.1f", nsig.getVal(), nsig.getError()));
  pv->AddText(Form("Background yield = %.1f #pm %.1f", nbkg.getVal(), nbkg.getError()));
  pv->Draw();

  // chi^2 / ndf (approx via RooPlot::chiSquare)
  int nfloat = fitRes ? fitRes->floatParsFinal().getSize() : 0;
  double chi2ndf = frame->chiSquare(nfloat);
  TPaveText* pv2 = new TPaveText(0.15, 0.52, 0.45, 0.58, "NDC");
  pv2->SetFillStyle(0);
  pv2->SetBorderSize(0);
  pv2->AddText(Form("#chi^{2}/ndf = %.3g", chi2ndf));
  pv2->Draw();
  pad2->cd();
  RooPlot* frame_pull = mass.frame(Title("Pull (data-fit)/#sigma"));
  RooHist* hpull = frame->pullHist(); // pulls computed from frame
  frame_pull->addPlotable(hpull, "P");
  frame_pull->GetXaxis()->SetTitle("m_{K^{+}K^{-}} (GeV/c^{2})");
  frame_pull->GetXaxis()->SetTitleSize(0.12);
  frame_pull->GetXaxis()->SetLabelSize(0.10);
  frame_pull->GetYaxis()->SetLabelSize(0.10);
  frame_pull->GetYaxis()->SetTitleSize(0.12);
  frame_pull->Draw();
  double xmin = frame_pull->GetXaxis()->GetXmin();
  double xmax = frame_pull->GetXaxis()->GetXmax();
  TLine l(xmin, 0.0, xmax, 0.0);
  l.SetLineStyle(kDashed);
  l.Draw();
  c->SaveAs(Form("phi_%d.pdf", runNumber));
}
