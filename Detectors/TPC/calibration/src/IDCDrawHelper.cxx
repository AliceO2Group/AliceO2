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

#include "TPCCalibration/IDCDrawHelper.h"
#include "TPCBase/Painter.h"
#include "TPCBase/Mapper.h"
#include "TH2Poly.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include <fmt/format.h>
#include "TROOT.h"

unsigned int o2::tpc::IDCDrawHelper::getPad(const unsigned int pad, const unsigned int region, const unsigned int row, const Side side)
{
  return (side == Side::A) ? pad : (Mapper::PADSPERROW[region][row] - pad - 1); // C-Side is mirrored
}

void o2::tpc::IDCDrawHelper::drawSector(const IDCDraw& idc, const unsigned int startRegion, const unsigned int endRegion, const unsigned int sector, const std::string zAxisTitle, const std::string filename, const float minZ, const float maxZ)
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSectorHist("hSector", "Sector;local #it{x} (cm);local #it{y} (cm); #it{IDC}");
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetYaxis()->SetTickSize(0.002f);
  poly->GetYaxis()->SetTitleOffset(0.7f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->SetStats(0);
  poly->GetZaxis()->SetTitle(zAxisTitle.data());
  poly->Sumw2();
  if (minZ < maxZ) {
    poly->SetMinimum(minZ);
    poly->SetMaximum(maxZ);
  }

  TCanvas* can = new TCanvas("can", "can", 2000, 1400);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.06f);
  can->SetTopMargin(0.04f);
  poly->Draw("colz");

  for (unsigned int region = startRegion; region < endRegion; ++region) {
    for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
      for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
        const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
        const auto coordinate = coords[padNum];
        const float yPos = -static_cast<float>(coordinate.yVals[0] + coordinate.yVals[2]) / 2; // local coordinate system is mirrored
        const float xPos = static_cast<float>(coordinate.xVals[0] + coordinate.xVals[2]) / 2;
        poly->Fill(xPos, yPos, idc.getIDC(sector, region, irow, ipad));
      }
    }
  }

  painter::drawSectorLocalPadNumberPoly(kBlack);
  painter::drawSectorInformationPoly(kRed, kRed);

  TLatex latex;
  latex.DrawLatexNDC(.07, .9, fmt::format("Sector {}", sector).data());
  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

void o2::tpc::IDCDrawHelper::drawSide(const IDCDraw& idc, const o2::tpc::Side side, const std::string zAxisTitle, const std::string filename, const float minZ, const float maxZ)
{
  TH2Poly* poly = o2::tpc::IDCDrawHelper::drawSide(idc, side, zAxisTitle);
  if (minZ < maxZ) {
    poly->SetMinimum(minZ);
    poly->SetMaximum(maxZ);
  }

  TCanvas* can = ((TVirtualPad*)gROOT->GetSelectedPad()) ? ((TCanvas*)((TVirtualPad*)gROOT->GetSelectedPad()->GetCanvas())) : new TCanvas("can", "can", 650, 600);
  can->SetTopMargin(0.04f);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.1f);
  poly->Draw("colz");
  o2::tpc::painter::drawSectorsXY(side);

  std::string sideName = (side == Side::A) ? "A-Side" : "C-Side";
  TLatex latex;
  latex.DrawLatexNDC(.13, .9, sideName.data());

  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

TH2Poly* o2::tpc::IDCDrawHelper::drawSide(const IDCDraw& idc, const o2::tpc::Side side, const std::string zAxisTitle)
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSideHist(side);
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetXaxis()->SetTitleOffset(1.2f);
  poly->GetYaxis()->SetTitleOffset(1.3f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->GetZaxis()->SetTitle(zAxisTitle.data());
  poly->GetZaxis()->SetMaxDigits(3); // force exponential axis
  poly->SetStats(0);
  poly->Sumw2();

  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : Mapper::NSECTORS;
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
          const float angDeg = 10.f + sector * 20;
          auto coordinate = coords[padNum];
          coordinate.rotate(angDeg);
          const float yPos = static_cast<float>(coordinate.yVals[0] + coordinate.yVals[1] + coordinate.yVals[2] + coordinate.yVals[3]) / 4;
          const float xPos = static_cast<float>(coordinate.xVals[0] + coordinate.xVals[1] + coordinate.xVals[2] + coordinate.xVals[3]) / 4;
          const auto padTmp = getPad(ipad, region, irow, side);
          poly->Fill(xPos, yPos, idc.getIDC(sector, region, irow, padTmp));
        }
      }
    }
  }

  return poly;
}

TH1F* o2::tpc::IDCDrawHelper::drawSide(const IDCDraw& idc, std::string_view type, const o2::tpc::Side side, int nbins1D, float xMin1D, float xMax1D)
{
  static const Mapper& mapper = Mapper::instance();
  const int bufferSize = TH1::GetDefaultBufferSize();
  TH1::SetDefaultBufferSize(Sector::MAXSECTOR * mapper.getPadsInSector());
  std::string sideName = (side == Side::A) ? "A" : "C";
  TH1F* h = new TH1F(fmt::format("h_{}_{}side", type.data(), sideName).data(), fmt::format("{} ({}-Side)", type.data(), sideName).data(), nbins1D, xMin1D, xMax1D);

  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : Mapper::NSECTORS;
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          const auto padTmp = getPad(ipad, region, irow, side);
          h->Fill(idc.getIDC(sector, region, irow, padTmp));
        }
      }
    }
  }
  TH1::SetDefaultBufferSize(bufferSize);
  return h;
}

void o2::tpc::IDCDrawHelper::drawRadialProfile(const IDCDraw& idc, TH2F& hist, const o2::tpc::Side side)
{
  const auto& mapper = Mapper::instance();

  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : Mapper::NSECTORS;
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          const auto padTmp = getPad(ipad, region, irow, side);
          const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
          const float padX = mapper.padCentre(padNum).x();
          hist.Fill(padX, idc.getIDC(sector, region, irow, padTmp));
        }
      }
    }
  }
}

void o2::tpc::IDCDrawHelper::drawIDCZeroStackCanvas(const IDCDraw& idc, const o2::tpc::Side side, const std::string_view type, int nbins1D, float xMin1D, float xMax1D, TCanvas& outputCanvas, int integrationInterval)
{
  size_t pad = 1;

  outputCanvas.Divide(4, 18);
  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : Mapper::NSECTORS;
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    auto hIROC = new TH1F(fmt::format("h1_{}_IROC_{:02}", type.data(), sector).data(), fmt::format("{} distribution IROC {:02} {}-Side", type.data(), sector, (side == Side::A) ? "A" : "C").data(), nbins1D, xMin1D, xMax1D);
    auto hOROC1 = new TH1F(fmt::format("h1_{}_OROC1_{:02}", type.data(), sector).data(), fmt::format("{} distribution OROC1 {:02} {}-Side", type.data(), sector, (side == Side::A) ? "A" : "C").data(), nbins1D, xMin1D, xMax1D);
    auto hOROC2 = new TH1F(fmt::format("h1_{}_OROC2_{:02}", type.data(), sector).data(), fmt::format("{} distribution OROC2 {:02} {}-Side", type.data(), sector, (side == Side::A) ? "A" : "C").data(), nbins1D, xMin1D, xMax1D);
    auto hOROC3 = new TH1F(fmt::format("h1_{}_OROC3_{:02}", type.data(), sector).data(), fmt::format("{} distribution OROC3 {:02} {}-Side", type.data(), sector, (side == Side::A) ? "A" : "C").data(), nbins1D, xMin1D, xMax1D);
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          const auto padTmp = getPad(ipad, region, irow, side);
          if (region < 4) {
            hIROC->Fill(idc.getIDC(sector, region, irow, padTmp));
          } else if (region < 6) {
            hOROC1->Fill(idc.getIDC(sector, region, irow, padTmp));
          } else if (region < 8) {
            hOROC2->Fill(idc.getIDC(sector, region, irow, padTmp));
          } else {
            hOROC3->Fill(idc.getIDC(sector, region, irow, padTmp));
          }
        }
      }
    }
    outputCanvas.cd(pad);
    hIROC->Draw();
    hIROC->GetXaxis()->SetTitle(fmt::format("{}", type.data()).data());
    hIROC->SetTitleOffset(1.05, "XY");
    hIROC->SetTitleSize(0.05, "XY");
    pad++;
    outputCanvas.cd(pad);
    hOROC1->Draw();
    hOROC1->GetXaxis()->SetTitle(fmt::format("{}", type.data()).data());
    hOROC1->SetTitleOffset(1.05, "XY");
    hOROC1->SetTitleSize(0.05, "XY");
    pad++;
    outputCanvas.cd(pad);
    hOROC2->Draw();
    hOROC2->GetXaxis()->SetTitle(fmt::format("{}", type.data()).data());
    hOROC2->SetTitleOffset(1.05, "XY");
    hOROC2->SetTitleSize(0.05, "XY");
    pad++;
    outputCanvas.cd(pad);
    hOROC3->Draw();
    hOROC3->GetXaxis()->SetTitle(fmt::format("{}", type.data()).data());
    hOROC3->SetTitleOffset(1.05, "XY");
    hOROC3->SetTitleSize(0.05, "XY");
    pad++;

    // associate histograms to canvas
    hIROC->SetBit(TObject::kCanDelete);
    hOROC1->SetBit(TObject::kCanDelete);
    hOROC2->SetBit(TObject::kCanDelete);
    hOROC3->SetBit(TObject::kCanDelete);
  }
}

std::string o2::tpc::IDCDrawHelper::getZAxisTitle(const IDCType type, const IDCDeltaCompression compression)
{
  std::string stype = "IDC";
  switch (type) {
    case IDCType::IDC:
    default: {
      return fmt::format("#it{{{}}} (ADC)", stype);
      break;
    }
    case IDCType::IDCZero: {
      return fmt::format("#it{{{}_{{0}}}} (ADC)", stype);
      break;
    }
    case IDCType::IDCDelta:
      switch (compression) {
        case IDCDeltaCompression::NO:
        default: {
          return fmt::format("#Delta#it{{{}}}", stype);
          break;
        }
        case IDCDeltaCompression::MEDIUM: {
          return fmt::format("#Delta#it{{{}}}_{{medium compressed}}", stype);
          break;
        }
        case IDCDeltaCompression::HIGH: {
          return fmt::format("#Delta#it{{{}}}_{{high compressed}}", stype);
          break;
        }
      }
    case IDCType::IDCOne: {
      return fmt::format("#it{{{}}}_{{1}}", stype);
      break;
    }
  }
}

void o2::tpc::IDCDrawHelper::drawSideGIF(const IDCDrawGIF& idcs, const unsigned int slices, const std::string zAxisTitle, const std::string filename, const float minZ, const float maxZ, const int run)
{
  const int gifSpeed = 40;

  TCanvas can("canvas", "canvas", 3350, 2000);
  TPad padIDCA("padIDCA", "padIDCA", 0, 0.25, 0.5, 1);
  TPad padIDCC("padIDCC", "padIDCC", 0.5, 0.25, 1, 1);
  TPad padIDCOne("padIDCOne", "padIDCOne", 0, 0, 1, 0.25);

  const float tm = 0.04f;
  const float rm = 0.14f;
  const float lm = 0.1f;
  const float bm = 0.2f;
  padIDCA.SetTopMargin(tm);
  padIDCA.SetRightMargin(rm);
  padIDCA.SetLeftMargin(lm);
  padIDCC.SetTopMargin(tm);
  padIDCC.SetRightMargin(rm);
  padIDCC.SetLeftMargin(lm);
  padIDCOne.SetTopMargin(tm);
  padIDCOne.SetRightMargin(rm / 2);
  padIDCOne.SetLeftMargin(lm / 2);
  padIDCOne.SetBottomMargin(bm);

  padIDCA.Draw();
  padIDCC.Draw();
  padIDCOne.Draw();

  std::array<TGraphErrors, SIDES> graphIDCOne;
  std::array<TGraphErrors, SIDES> graphIDCOneSlice;
  TMultiGraph multiGraph;
  float minY = 1000;
  float maxY = -1;
  const float widthLine = 0.005f;
  const float idcWH = 0.5; // IDC width half: 1ms/2
  for (unsigned int sideT = 0; sideT < SIDES; ++sideT) {
    const Side side = (sideT == 0) ? Side::A : Side::C;
    const auto col = (sideT == 0) ? (kGreen + 2) : kBlue;
    graphIDCOne[sideT].SetFillColorAlpha(col, 0.3);
    graphIDCOne[sideT].SetLineWidth(5);
    graphIDCOneSlice[sideT] = graphIDCOne[sideT];
    graphIDCOneSlice[sideT].Set(1);
    graphIDCOneSlice[sideT].SetFillColorAlpha(col, 1);
    for (unsigned int slice = 0; slice < slices; ++slice) {
      const auto idc = idcs.mIDCOneFunc(side, slice);
      if (idc < minY) {
        minY = idc;
      }
      if (idc > maxY) {
        maxY = idc;
      }
      graphIDCOne[sideT].AddPoint(slice + idcWH, idc);
      graphIDCOne[sideT].SetPointError(slice, idcWH, widthLine);
    }
    multiGraph.Add(&graphIDCOne[sideT]);
  }

  const int font = 63;
  const int fontsize = 50;
  multiGraph.GetXaxis()->SetTitleFont(font);
  multiGraph.GetXaxis()->SetLabelFont(font);
  multiGraph.GetYaxis()->SetTitleFont(font);
  multiGraph.GetYaxis()->SetLabelFont(font);
  multiGraph.GetXaxis()->SetLabelSize(fontsize);
  multiGraph.GetXaxis()->SetTitleSize(fontsize);
  multiGraph.GetYaxis()->SetLabelSize(fontsize);
  multiGraph.GetYaxis()->SetTitleSize(fontsize);
  multiGraph.GetXaxis()->SetTitle("#it{t} (ms)");
  multiGraph.GetXaxis()->SetTitleOffset(0.9f);
  multiGraph.GetYaxis()->SetTitleOffset(1.5f);
  multiGraph.GetYaxis()->SetTitle(IDCDrawHelper::getZAxisTitle(IDCType::IDCOne).data());
  multiGraph.SetMinimum(0.9 * minY);
  multiGraph.SetMaximum(1.1 * maxY);
  multiGraph.GetXaxis()->SetLimits(0, slices);

  for (unsigned int slice = 0; slice < slices; ++slice) {
    LOGP(info, "Drawing slice {} from {}", slice, slices - 1);
    IDCDrawHelper::IDCDraw drawFun;
    std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [slice, idcs](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
      return idcs.mIDCFunc(sector, region, irow, pad, slice);
    };
    drawFun.mIDCFunc = idcFunc;

    TH2Poly* poly[SIDES]{nullptr, nullptr};
    for (int sideT = 0; sideT < SIDES; ++sideT) {
      const Side side = (sideT == 0) ? Side::A : Side::C;
      poly[sideT] = o2::tpc::IDCDrawHelper::drawSide(drawFun, side, zAxisTitle);
      poly[sideT]->GetXaxis()->SetTitleFont(font);
      poly[sideT]->GetXaxis()->SetLabelFont(font);
      poly[sideT]->GetYaxis()->SetTitleFont(font);
      poly[sideT]->GetYaxis()->SetLabelFont(font);
      poly[sideT]->GetXaxis()->SetLabelSize(fontsize);
      poly[sideT]->GetXaxis()->SetTitleSize(fontsize);
      poly[sideT]->GetYaxis()->SetLabelSize(fontsize);
      poly[sideT]->GetYaxis()->SetTitleSize(fontsize);

      if (minZ < maxZ) {
        poly[sideT]->SetMinimum(minZ);
        poly[sideT]->SetMaximum(maxZ);
      }

      if (sideT == 0) {
        padIDCA.cd();
      } else {
        padIDCC.cd();
      }

      poly[sideT]->Draw("colz");
      const std::string sideName = (side == Side::A) ? "A-Side" : "C-Side";
      TLatex latex;
      const auto col = (side == Side::A) ? (kGreen + 2) : kBlue;
      latex.SetTextColor(col);
      latex.DrawLatexNDC(0.13, 0.9, sideName.data());
      if (run > 0) {
        latex.DrawLatexNDC(0.62, 0.9, fmt::format("Run {}", run).data());
      }
      graphIDCOneSlice[sideT].SetPoint(0, slice + idcWH, idcs.mIDCOneFunc(side, slice));
      graphIDCOneSlice[sideT].SetPointError(0, idcWH, widthLine);
    }

    padIDCOne.cd();
    multiGraph.Draw("ZA E2");
    graphIDCOneSlice[0].Draw("Z E2 SAME");
    graphIDCOneSlice[1].Draw("Z E2 SAME");

    can.Print(Form("%s.gif+%i", filename.data(), gifSpeed));
    if (slice == (slices - 1)) {
      can.Print(Form("%s.gif++", filename.data()));
    }

    delete poly[0];
    delete poly[1];
  }
}
