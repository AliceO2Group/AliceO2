// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCCalibration/IDCGroup.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCBase/Painter.h"
#include "TPCBase/Mapper.h"
#include "TH2Poly.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLatex.h"
#include <numeric>

void o2::tpc::IDCGroup::dumpToTree(const char* outname) const
{
  o2::utils::TreeStreamRedirector pcstream(outname, "RECREATE");
  pcstream.GetFile()->cd();
  for (unsigned int integrationInterval = 0; integrationInterval < getNIntegrationIntervals(); ++integrationInterval) {
    for (unsigned int irow = 0; irow < mRows; ++irow) {
      for (unsigned int ipad = 0; ipad < mPadsPerRow[irow]; ++ipad) {
        float idc = (*this)(irow, ipad, integrationInterval);
        pcstream << "idcs"
                 << "row=" << irow
                 << "pad=" << ipad
                 << "IDC=" << idc
                 << "\n";
      }
    }
  }
  pcstream.Close();
}

void o2::tpc::IDCGroup::draw(const unsigned int integrationInterval, const std::string filename) const
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSectorHist("hSector", "Sector;local #it{x} (cm);local #it{y} (cm); #it{IDC}");
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetYaxis()->SetTickSize(0.002f);
  poly->GetYaxis()->SetTitleOffset(0.7f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->SetStats(0);

  TCanvas* can = new TCanvas("can", "can", 2000, 1400);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.06f);
  can->SetTopMargin(0.04f);

  TLatex lat;
  lat.SetTextFont(63);
  lat.SetTextSize(2);

  poly->Draw("colz");
  for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[mRegion]; ++irow) {
    for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[mRegion][irow]; ++ipad) {
      const auto padNum = getGlobalPadNumber(irow, ipad);
      const auto coordinate = coords[padNum];
      const float yPos = -0.5f * (coordinate.yVals[0] + coordinate.yVals[2]); // local coordinate system is mirrored
      const float xPos = 0.5f * (coordinate.xVals[0] + coordinate.xVals[2]);
      poly->Fill(xPos, yPos, (*this)(getGroupedRow(irow), getGroupedPad(ipad, irow), integrationInterval));
      lat.SetTextAlign(12);
      lat.DrawLatex(xPos, yPos, Form("%i", ipad));
    }
  }
  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

void o2::tpc::IDCGroup::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "UPDATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

float o2::tpc::IDCGroup::getValUngroupedGlobal(unsigned int ugrow, unsigned int upad, unsigned int integrationInterval) const
{
  return mIDCsGrouped[getIndexUngrouped(Mapper::getLocalRowFromGlobalRow(ugrow), upad, integrationInterval)];
}

std::vector<float> o2::tpc::IDCGroup::get1DIDCs() const
{
  // integrate IDCs for each interval
  std::vector<float> idc;
  const unsigned int nIntervals = getNIntegrationIntervals();
  idc.reserve(nIntervals);
  for (unsigned int i = 0; i < nIntervals; ++i) {
    // set integration range for one integration interval
    const auto start = mIDCsGrouped.begin() + i * getNIDCsPerIntegrationInterval();
    const auto end = start + getNIDCsPerIntegrationInterval();
    idc.emplace_back(std::accumulate(start, end, decltype(mIDCsGrouped)::value_type(0)));
  }
  // normalize 1D-IDCs to absolute space charge
  const float norm = Mapper::REGIONAREA[mRegion] / getNIDCsPerIntegrationInterval();
  std::transform(idc.begin(), idc.end(), idc.begin(), [norm](auto& val) { return val * norm; });

  return idc;
}
