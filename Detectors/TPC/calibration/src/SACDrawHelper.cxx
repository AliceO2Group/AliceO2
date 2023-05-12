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

#include "TPCCalibration/SACDrawHelper.h"
#include "TPCBase/Painter.h"
#include "TH2Poly.h"
#include "TCanvas.h"
#include "TLatex.h"
#include <fmt/format.h>
#include <numeric>

void o2::tpc::SACDrawHelper::drawSector(const SACDraw& SAC, const unsigned int sector, const std::string zAxisTitle, const std::string filename, const float minZ, const float maxZ)
{
  const auto coords = o2::tpc::painter::getStackCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSectorHist("hSector", "Sector;local #it{x} (cm);local #it{y} (cm); #it{SAC}", 83.65f, 247.7f, -43.7f, 43.7f, painter::Type::Stack);

  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetYaxis()->SetTickSize(0.002f);
  poly->GetYaxis()->SetTitleOffset(0.7f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->SetStats(0);
  poly->GetZaxis()->SetTitle(zAxisTitle.data());
  if (minZ < maxZ) {
    poly->SetMinimum(minZ);
    poly->SetMaximum(maxZ);
  }

  TCanvas* can = new TCanvas("can", "can", 2000, 1400);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.06f);
  can->SetTopMargin(0.04f);
  poly->Draw("colz");

  for (unsigned int stack = 0; stack < GEMSTACKSPERSECTOR; ++stack) {
    const auto coordinate = coords[stack];
    const float yPos = -static_cast<float>(coordinate.yVals[0] + coordinate.yVals[2]) / 2; // local coordinate system is mirrored
    const float xPos = static_cast<float>(coordinate.xVals[0] + coordinate.xVals[2]) / 2;
    poly->Fill(xPos, yPos, SAC.getSAC(sector, stack));
  }

  TLatex latex;
  latex.DrawLatexNDC(.07, .9, fmt::format("Sector {}", sector).data());
  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

void o2::tpc::SACDrawHelper::drawSide(const SACDraw& SAC, const o2::tpc::Side side, const std::string zAxisTitle, const std::string filename, const float minZ, const float maxZ)
{
  TH2Poly* poly = o2::tpc::SACDrawHelper::drawSide(SAC, side, zAxisTitle);
  if (minZ < maxZ) {
    poly->SetMinimum(minZ);
    poly->SetMaximum(maxZ);
  }

  TCanvas* can = new TCanvas("can", "can", 650, 600);
  can->SetTopMargin(0.04f);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.1f);
  poly->Draw("colz");

  std::string sideName = (side == Side::A) ? "A-Side" : "C-Side";
  TLatex latex;
  latex.DrawLatexNDC(.13, .9, sideName.data());

  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

TH2Poly* o2::tpc::SACDrawHelper::drawSide(const SACDraw& SAC, const o2::tpc::Side side, const std::string zAxisTitle)
{
  const auto coords = o2::tpc::painter::getStackCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSideHist(side, painter::Type::Stack);
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetXaxis()->SetTitleOffset(1.2f);
  poly->GetYaxis()->SetTitleOffset(1.3f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->GetZaxis()->SetTitle(zAxisTitle.data());
  poly->GetZaxis()->SetMaxDigits(3); // force exponential axis
  poly->SetStats(0);

  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : (sectorStart * SIDES);
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    for (unsigned int stack = 0; stack < GEMSTACKSPERSECTOR; ++stack) {
      const float angDeg = 10.f + sector * 20;
      auto coordinate = coords[stack];
      coordinate.rotate(angDeg);
      auto const count = static_cast<float>(coordinate.yVals.size());
      const float yPos = std::reduce(coordinate.yVals.begin(), coordinate.yVals.end()) / count;
      const float xPos = std::reduce(coordinate.xVals.begin(), coordinate.xVals.end()) / count;
      poly->Fill(xPos, yPos, SAC.getSAC(sector, stack));
    }
  }
  return poly;
}

std::string o2::tpc::SACDrawHelper::getZAxisTitle(const SACType type)
{
  std::string stype = "SAC";
  switch (type) {
    case SACType::IDC:
    default: {
      return fmt::format("#it{{{}}} (ADC)", stype);
      break;
    }
    case SACType::IDCZero: {
      return fmt::format("#it{{{}_{{0}}}} (ADC)", stype);
      break;
    }
    case SACType::IDCDelta:
      return fmt::format("#Delta#it{{{}}}", stype);
      break;
    case SACType::IDCOne: {
      return fmt::format("#Delta#it{{{}}}_{{1}}", stype);
      break;
    }
  }
}
