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

#include <map>

// root includes
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2Poly.h"
#include "TProfile.h"

#include "TPCQC/IDCsVsSACs.h"
#include "TPCCalibration/IDCCCDBHelper.h"
#include "TPCCalibration/SACCCDBHelper.h"
#include "TPCCalibration/IDCDrawHelper.h"
#include "TPCCalibration/SACDrawHelper.h"
#include "TPCCalibration/SACFactorization.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/Painter.h"

TCanvas* o2::tpc::qc::IDCsVsSACs::drawComparisionSACandIDCZero(TCanvas* outputCanvas, int nbins1D, float xMin1D, float xMax1D, int nbins1DSAC, float xMin1DSAC, float xMax1DSAC) const
{

  const Mapper& mapper = Mapper::instance();

  const auto& calDet = mCCDBHelper->getIDCZeroCalDet();
  std::function<float(const unsigned int, const unsigned int)> SACFunc;

  SACFunc = [&](const unsigned int sector, const unsigned int stack) {
    return mSacCCDBHelper->getSACZeroVal(sector, stack);
  };

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;

  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(o2::tpc::SACType::IDCZero);
  auto hSac0SideA = SACDrawHelper::drawSide(drawFun, o2::tpc::Side::A, zAxisTitle);
  auto hSac0SideC = SACDrawHelper::drawSide(drawFun, o2::tpc::Side::C, zAxisTitle);

  const std::string SACname = "SAC0";
  hSac0SideA->SetTitle(fmt::format("{} ({}-Side)", SACname.data(), "A").data());
  hSac0SideC->SetTitle(fmt::format("{} ({}-Side)", SACname.data(), "C").data());

  // ===| name and title |======================================================
  std::string title = calDet.getName();
  std::string name = calDet.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  std::replace(title.begin(), title.end(), '_', ' ');

  const std::string Rationame = "Ratio";

  const int bufferSize = TH1::GetDefaultBufferSize();
  TH1::SetDefaultBufferSize(Sector::MAXSECTOR * mapper.getPadsInSector());

  auto h2DIdc0SideA = new TH2F(fmt::format("hIDC0_Aside_2D_{}", name).data(), fmt::format("{} (A-Side);#it{{x}} (cm);#it{{y}} (cm)", title).data(),
                               330, -270, 270, 330, -270, 270);

  auto h2DIdc0SideC = new TH2F(fmt::format("hIDC0_Cside_2D_{}", name).data(), fmt::format("{} (C-Side);#it{{x}} (cm);#it{{y}} (cm)", title).data(),
                               330, -270, 270, 330, -270, 270);

  auto hAsideRatio2D = new TH2F(fmt::format("hRatio_Aside_2D_{}", Rationame).data(), fmt::format("{} (A-Side);#it{{x}} (cm);#it{{y}} (cm)", Rationame).data(),
                                330, -270, 270, 330, -270, 270);

  auto hCsideRatio2D = new TH2F(fmt::format("hRatio_CsideRatio_2D_{}", Rationame).data(), fmt::format("{} (C-Side);#it{{x}} (cm);#it{{y}} (cm)", Rationame).data(),
                                330, -270, 270, 330, -270, 270);

  auto hAsideRatio1D = new TH1F(fmt::format("h_AsideRatio_1D_{}", Rationame).data(), fmt::format("{} (A-Side);IDC0/SAC0 (arb unit.);Entries", Rationame).data(),
                                nbins1D, xMin1D, xMax1D); // TODO: modify ranges

  auto hCsideRatio1D = new TH1F(fmt::format("h_CsideRatio_1D_{}", Rationame).data(), fmt::format("{} (C-Side);IDC0/SAC0 (arb unit.);Entries", Rationame).data(),
                                nbins1D, xMin1D, xMax1D); // TODO: modify ranges

  for (ROC roc; !roc.looped(); ++roc) {
    auto hist2DIdc0 = h2DIdc0SideA;
    auto histRatio2D = hAsideRatio2D;
    auto histRatio1D = hAsideRatio1D;
    if (roc.side() == Side::C) {
      hist2DIdc0 = h2DIdc0SideC;
      histRatio2D = hCsideRatio2D;
      histRatio1D = hCsideRatio1D;
    }
    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      /// get SACs zero values
      const int part = o2::tpc::Mapper::REGION[irow] / 2;
      const int stackID = (part < 2) ? 0 : part - 1;
      const auto sacZero = mSacCCDBHelper->getSACZeroVal(roc.getSector(), stackID);
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto val = calDet.getValue(roc, irow, ipad);
        const GlobalPosition2D pos = mapper.getPadCentre(PadROCPos(roc, irow, ipad));
        const int bin = hist2DIdc0->FindBin(pos.X(), pos.Y());
        if (!hist2DIdc0->GetBinContent(bin)) {
          hist2DIdc0->SetBinContent(bin, val);
          if (val / sacZero != 0) {
            histRatio2D->SetBinContent(bin, val / sacZero);
          }
        }
        if (val / sacZero != 0) {
          histRatio2D->SetBinContent(bin, val / sacZero);
          histRatio1D->Fill(val / sacZero);
        }
      }
    }
  }

  if (xMax1D > xMin1D) {

    hSac0SideA->SetMinimum(xMin1DSAC);
    hSac0SideC->SetMinimum(xMin1DSAC);
    hSac0SideA->SetMaximum(xMax1DSAC);
    hSac0SideC->SetMaximum(xMax1DSAC);

    h2DIdc0SideA->SetMinimum(xMin1D);
    h2DIdc0SideA->SetMaximum(xMax1D);
    h2DIdc0SideC->SetMinimum(xMin1D);
    h2DIdc0SideC->SetMaximum(xMax1D);
    hAsideRatio2D->SetMinimum(xMin1D);
    hCsideRatio2D->SetMaximum(xMax1D);
  }

  // ===| Draw histograms |=====================================================
  gStyle->SetOptStat("mr");
  auto c = outputCanvas;
  if (!c) {
    c = new TCanvas(fmt::format("c_{}", name).data(), title.data(), 1000, 1000);
  }
  gStyle->SetStatX(1. - gPad->GetRightMargin());
  gStyle->SetStatY(1. - gPad->GetTopMargin());

  c->Clear();
  c->Divide(2, 4);

  c->cd(1);
  h2DIdc0SideA->Draw("colz");
  h2DIdc0SideA->SetStats(0);
  h2DIdc0SideA->SetTitleOffset(1.05, "XY");
  h2DIdc0SideA->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::A);

  c->cd(2);
  h2DIdc0SideC->Draw("colz");
  h2DIdc0SideC->SetStats(0);
  h2DIdc0SideC->SetTitleOffset(1.05, "XY");
  h2DIdc0SideC->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::C);
  c->cd(3);
  hSac0SideA->Draw("colz");
  hSac0SideA->SetStats(0);
  hSac0SideA->SetTitleOffset(1.05, "XY");
  hSac0SideA->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::A);
  c->cd(4);
  hSac0SideC->Draw("colz");
  hSac0SideC->SetStats(0);
  hSac0SideC->SetTitleOffset(1.05, "XY");
  hSac0SideC->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::A);

  c->cd(5);
  hAsideRatio2D->Draw("colz");
  hAsideRatio2D->SetStats(0);
  hAsideRatio2D->SetTitleOffset(1.05, "XY");
  hAsideRatio2D->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::A);

  c->cd(6);
  hCsideRatio2D->Draw("colz");
  hCsideRatio2D->SetStats(0);
  hCsideRatio2D->SetTitleOffset(1.05, "XY");
  hCsideRatio2D->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::C);
  c->cd(7);
  hAsideRatio1D->Draw();
  hAsideRatio1D->SetTitleOffset(1.05, "XY");
  hAsideRatio1D->SetTitleSize(0.05, "XY");
  c->cd(8);
  hCsideRatio1D->Draw();
  hAsideRatio1D->SetTitleOffset(1.05, "XY");
  hAsideRatio1D->SetTitleSize(0.05, "XY");

  // reset the buffer size
  TH1::SetDefaultBufferSize(bufferSize);

  // associate histograms to canvas
  hAsideRatio1D->SetBit(TObject::kCanDelete);
  hCsideRatio1D->SetBit(TObject::kCanDelete);
  h2DIdc0SideA->SetBit(TObject::kCanDelete);
  h2DIdc0SideC->SetBit(TObject::kCanDelete);
  hAsideRatio2D->SetBit(TObject::kCanDelete);
  hCsideRatio2D->SetBit(TObject::kCanDelete);

  hSac0SideA->SetBit(TObject::kCanDelete);
  hSac0SideC->SetBit(TObject::kCanDelete);
  return c;
}
