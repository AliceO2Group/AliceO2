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

  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

void o2::tpc::IDCDrawHelper::drawSide(const IDCDraw& idc, const o2::tpc::Side side, const std::string zAxisTitle, const std::string filename, const float minZ, const float maxZ)
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSideHist(side);
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetXaxis()->SetTitleOffset(1.2f);
  poly->GetYaxis()->SetTitleOffset(1.3f);
  poly->GetZaxis()->SetTitleOffset(1.4f);
  poly->GetZaxis()->SetTitle(zAxisTitle.data());
  poly->GetZaxis()->SetMaxDigits(3); // force exponential axis
  poly->SetStats(0);
  if (minZ < maxZ) {
    poly->SetMinimum(minZ);
    poly->SetMaximum(maxZ);
  }

  TCanvas* can = new TCanvas("can", "can", 650, 600);
  can->SetTopMargin(0.04f);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.1f);
  poly->Draw("colz");

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
          const auto padTmp = (side == Side::A) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad); // C-Side is mirrored
          poly->Fill(xPos, yPos, idc.getIDC(sector, region, irow, padTmp));
        }
      }
    }
  }
  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

std::string o2::tpc::IDCDrawHelper::getZAxisTitle(const IDCType type, const IDCDeltaCompression compression)
{
  switch (type) {
    case IDCType::IDC:
    default: {
      return "#it{IDC} (ADC)";
      break;
    }
    case IDCType::IDCZero: {
      return "#it{IDC_{0}} (ADC)";
      break;
    }
    case IDCType::IDCDelta:
      switch (compression) {
        case IDCDeltaCompression::NO:
        default: {
          return "#Delta#it{IDC}";
          break;
        }
        case IDCDeltaCompression::MEDIUM: {
          return "#Delta#it{IDC}_{medium compressed}";
          break;
        }
        case IDCDeltaCompression::HIGH: {
          return "#Delta#it{IDC}_{high compressed}";
          break;
        }
      }
    case IDCType::IDCOne: {
      return "#Delta#it{IDC}_{1}";
      break;
    }
  }
}
