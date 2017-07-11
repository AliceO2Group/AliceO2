// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TCanvas.h"
#include "TString.h"
#include "TAxis.h"
#include "TH1.h"
#include "TH2.h"

#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"

using namespace o2::TPC;

//template <class T>
void Painter::draw(const CalDet<T>& calDet)
{
  using DetType = CalDet<T>;
  using CalType = CalArray<T>;

  const auto& name = calDet.getName().c_str();

  // ===| define histograms |===================================================
  // TODO: auto scaling of ranges based on mean and variance?
  //       for the moment use roots auto scaling
  auto hAside1D = new TH1F(Form("h_Aside_1D_%s", name), Form("%s (A-Side)", name),
                         300, 0, 0); //TODO: modify ranges

  auto hCside1D = new TH1F(Form("h_Cside_1D_%s", name), Form("%s (C-Side)", name),
                         300, 0, 0); //TODO: modify ranges

  auto hAside2D = new TH2F(Form("h_Aside_2D_%s;x (cm);y (cm)", name), Form("%s (A-Side)", name),
                         300, -300, 300, 300, -300, 300);

  auto hCside2D = new TH2F(Form("h_Cside_2D_%s;x (cm);y (cm)", name), Form("%s (C-Side)", name),
                         300, -300, 300, 300, -300, 300);


  const Mapper& mapper = Mapper::instance();

  for (auto& cal : calDet.getData()) {

    int calPadSubsetNumber = cal.getPadSubsetNumber();
    int row = -1;
    int pad = -1;
    int offset=0;
    Sector sector;
    switch (cal.getPadSubset()) {
      case PadSubset::ROC: {
        ROC roc(calPadSubsetNumber);
        offset = (roc.rocType() == RocType::IROC)?0:mapper.getPadsInIROC();
        sector = roc.getSector();
        break;
      }
      case PadSubset::Partition: {
        const int npartitions = mapper.getNumberOfPartitions();
        const int partition = calPadSubsetNumber%npartitions;
        const int rowOffset = mapper.getPartitionInfo(partition).getGlobalRowOffset();
        offset = mapper.globalPadNumber(PadPos(rowOffset, 0));
        sector = calPadSubsetNumber/npartitions;
        break;
      }
      case PadSubset::Region: {
        const int nregions = mapper.getNumberOfPadRegions();
        const int region = calPadSubsetNumber%mapper.getNumberOfPadRegions();
        const int rowOffset = mapper.getPadRegionInfo(region).getGlobalRowOffset();
        offset = mapper.globalPadNumber(PadPos(rowOffset, 0));
        sector = calPadSubsetNumber/nregions;
        break;
      }
    }

    auto hist2D = hAside2D;
    auto hist1D = hAside1D;
    if (sector.side() == Side::C) {
      hist2D = hCside2D;
      hist1D = hCside1D;
    }

    int padNumber = offset;
    for (const auto& val : cal.getData()) {
      const PadCentre& localCoord = mapper.padCentre(padNumber);
      const GlobalPosition3D pos = mapper.LocalToGlobal(LocalPosition3D(localCoord.getX(), localCoord.getY(), 0), sector);

      Int_t bin = hist2D->FindBin(pos.getX(), pos.getY());

      //hist2D->SetBinContent(bin, val);
      hist2D->Fill(pos.getX(), pos.getY(), val);
      hist1D->Fill(val);

      ++padNumber;
    }
  }

  // ===| Draw histograms |=====================================================
  auto c = new TCanvas(Form("c_%s", name));
  c->Divide(2,2);

  c->cd(1);
  hAside2D->Draw("colz");

  c->cd(2);
  hCside2D->Draw("colz");

  c->cd(3);
  hAside1D->Draw();

  c->cd(4);
  hCside1D->Draw();
}

//______________________________________________________________________________
//template <class T>
void Painter::draw(const CalArray<T>& calArray)
{
}

//______________________________________________________________________________
//template <class T>
TH2* Painter::getHistogram2D(const CalDet<T>& calDet, Side side)
{
  static const Mapper& mapper = Mapper::instance();

  const auto name = calDet.getName().c_str();
  const char side_name = (side == Side::A)?'A':'C';
  auto h2D = new TH2F(Form("h_%cside_2D_%s;x (cm);y (cm)", side_name, name), Form("%s (%c-Side)", name, side_name),
                         300, -300, 300, 300, -300, 300);

  for (ROC roc; !roc.looped(); ++roc) {
    if (roc.side() != side) continue;
    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow=0; irow<nrows; ++irow) {
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad=0; ipad<npads; ++ipad) {
        const auto val = calDet.getValue(roc, irow, ipad);
        const GlobalPosition2D pos = mapper.getPadCentre(PadROCPos(roc, irow, ipad));
        const int binX = h2D->GetXaxis()->FindBin(pos.getX());
        const int binY = h2D->GetXaxis()->FindBin(pos.getY());
        h2D->SetBinContent(binX, binY, val);
      }
    }
  }

  return h2D;
}

//______________________________________________________________________________
//template <class T>
TH2* Painter::getHistogram2D(const CalArray<T>& calArray)
{
  static const Mapper& mapper = Mapper::instance();

  const size_t position = calArray.getPadSubsetNumber();

  const PadSubset padSubset = calArray.getPadSubset();
  // ===| maximum number of rows and pads |=====================================
  const int nrows = mapper.getNumberOfPadRows(padSubset, position);
  const int npads = mapper.getNumberOfPadsInRow(padSubset, position, nrows-1);

  // ===| create histogram |====================================================
  const auto name = calArray.getName().c_str();
  auto hist = new TH2D(Form("h_%s", name), Form("%s;pad row;pad", name), nrows, 0., nrows, npads, -npads/2, npads/2);

  // ===| fill hist |===========================================================
  for (int irow = 0; irow<nrows; ++irow) {
    const int padsInRow = mapper.getNumberOfPadsInRow(padSubset, position, irow);
    for (int ipad = 0; ipad<padsInRow; ++ipad) {
       const auto val = calArray.getValue(irow, ipad);
       const int cpad = ipad - padsInRow/2;
       hist->Fill(irow, cpad, val);
    }
  }
  return hist;
}


