// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCCalibration/IDCCCDBHelper.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/Painter.h"
#include "TH2Poly.h"
#include "TCanvas.h"
#include "TLatex.h"

template <typename DataT>
std::string o2::tpc::IDCCCDBHelper<DataT>::getZAxisTitle(const o2::tpc::IDCType type) const
{
  switch (type) {
    case IDCType::IDCZero:
    default:
      return "#it{IDC_{0}}";
      break;
    case IDCType::IDCDelta:
      return "#Delta#it{IDC}";
      break;
    case IDCType::IDCOne:
    case IDCType::IDC:
      return "Wrong Type";
      break;
  }
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawSide(const o2::tpc::IDCType type, const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename) const
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSideHist(side);
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetXaxis()->SetTitleOffset(1.2f);
  poly->GetYaxis()->SetTitleOffset(1.3f);
  poly->GetZaxis()->SetTitleOffset(1.2f);
  poly->GetZaxis()->SetTitle(getZAxisTitle(type).data());
  poly->GetZaxis()->SetMaxDigits(3); // force exponential axis
  poly->SetStats(0);

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
          const float yPos = 0.25f * static_cast<float>(coordinate.yVals[0] + coordinate.yVals[1] + coordinate.yVals[2] + coordinate.yVals[3]);
          const float xPos = 0.25f * static_cast<float>(coordinate.xVals[0] + coordinate.xVals[1] + coordinate.xVals[2] + coordinate.xVals[3]);
          const auto padTmp = (side == Side::A) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad); // C-Side is mirrored
          switch (type) {
            case IDCType::IDCZero:
            default:
              poly->Fill(xPos, yPos, getIDCZeroVal(sector, region, irow, padTmp));
              break;
            case IDCType::IDCDelta:
              poly->Fill(xPos, yPos, getIDCDeltaVal(sector, region, irow, padTmp, integrationInterval));
              break;
            case IDCType::IDC:
              break;
            case IDCType::IDCOne:
              break;
          }
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

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawSector(const IDCType type, const unsigned int sector, const unsigned int integrationInterval, const std::string filename) const
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSectorHist("hSector", "Sector;local #it{x} (cm);local #it{y} (cm); #it{IDC}");
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetYaxis()->SetTickSize(0.002f);
  poly->GetYaxis()->SetTitleOffset(0.7f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->SetStats(0);
  poly->GetZaxis()->SetTitle(getZAxisTitle(type).data());

  TCanvas* can = new TCanvas("can", "can", 2000, 1400);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.06f);
  can->SetTopMargin(0.04f);

  TLatex lat;
  lat.SetTextFont(63);
  lat.SetTextSize(2);
  poly->Draw("colz");

  for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
    for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
      for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
        const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
        const auto coordinate = coords[padNum];
        const float yPos = -0.5f * static_cast<float>(coordinate.yVals[0] + coordinate.yVals[2]); // local coordinate system is mirrored
        const float xPos = 0.5f * static_cast<float>(coordinate.xVals[0] + coordinate.xVals[2]);
        switch (type) {
          case IDCType::IDCZero:
          default:
            poly->Fill(xPos, yPos, getIDCZeroVal(sector, region, irow, ipad));
            break;
          case IDCType::IDCDelta:
            poly->Fill(xPos, yPos, getIDCDeltaVal(sector, region, irow, ipad, integrationInterval));
            break;
          case IDCType::IDC:
            break;
          case IDCType::IDCOne:
            break;
        }
        // draw global pad number
        lat.SetTextAlign(12);
        lat.DrawLatex(xPos, yPos, Form("%i", ipad));
      }
    }
  }
  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

/// load IDC-Delta, 0D-IDCs, grouping parameter
template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::loadAll()
{
  loadIDCDelta();
  loadIDCZero();
  loadGroupingParameter();
}

template class o2::tpc::IDCCCDBHelper<float>;
template class o2::tpc::IDCCCDBHelper<short>;
template class o2::tpc::IDCCCDBHelper<char>;
