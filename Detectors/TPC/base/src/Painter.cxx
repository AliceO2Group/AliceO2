// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <string>
#include <algorithm>

#include "TCanvas.h"
#include "TString.h"
#include "TAxis.h"
#include "TH1.h"
#include "TH2.h"

#include "TPCBase/ROC.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"

using namespace o2::tpc;

template <class T>
void painter::draw(const CalDet<T>& calDet)
{
  using DetType = CalDet<T>;
  using CalType = CalArray<T>;

  static const Mapper& mapper = Mapper::instance();

  // ===| name and title |======================================================
  const auto title = calDet.getName().c_str();
  std::string name = calDet.getName();
  std::replace(name.begin(), name.end(), ' ', '_');

  // ===| define histograms |===================================================
  // TODO: auto scaling of ranges based on mean and variance?
  //       for the moment use roots auto scaling

  // set buffer size such that autoscaling uses the full range. This is about 2MB per histogram!
  const int bufferSize = TH1::GetDefaultBufferSize();
  TH1::SetDefaultBufferSize(Sector::MAXSECTOR * mapper.getPadsInSector());

  auto hAside1D = new TH1F(Form("h_Aside_1D_%s", name.c_str()), Form("%s (A-Side)", title),
                           300, 0, 0); //TODO: modify ranges

  auto hCside1D = new TH1F(Form("h_Cside_1D_%s", name.c_str()), Form("%s (C-Side)", title),
                           300, 0, 0); //TODO: modify ranges

  auto hAside2D = new TH2F(Form("h_Aside_2D_%s;x (cm);y (cm)", name.c_str()), Form("%s (A-Side)", title),
                           300, -300, 300, 300, -300, 300);

  auto hCside2D = new TH2F(Form("h_Cside_2D_%s;x (cm);y (cm)", name.c_str()), Form("%s (C-Side)", title),
                           300, -300, 300, 300, -300, 300);

  for (ROC roc; !roc.looped(); ++roc) {

    auto hist2D = hAside2D;
    auto hist1D = hAside1D;
    if (roc.side() == Side::C) {
      hist2D = hCside2D;
      hist1D = hCside1D;
    }

    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto val = calDet.getValue(roc, irow, ipad);
        const GlobalPosition2D pos = mapper.getPadCentre(PadROCPos(roc, irow, ipad));
        const int bin = hist2D->FindBin(pos.X(), pos.Y());
        if (!hist2D->GetBinContent(bin))
          hist2D->SetBinContent(bin, val);
        hist1D->Fill(val);
      }
    }
  }

  // ===| Draw histograms |=====================================================
  auto c = new TCanvas(Form("c_%s", name.c_str()), title);
  c->Divide(2, 2);

  c->cd(1);
  hAside2D->Draw("colz");

  c->cd(2);
  hCside2D->Draw("colz");

  c->cd(3);
  hAside1D->Draw();

  c->cd(4);
  hCside1D->Draw();

  // reset the buffer size
  TH1::SetDefaultBufferSize(bufferSize);
}

//______________________________________________________________________________
template <class T>
void painter::draw(const CalArray<T>& calArray)
{
  const auto title = calArray.getName().c_str();
  std::string name = calArray.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  auto c = new TCanvas(Form("c_%s", name.c_str()), title);

  auto hist = getHistogram2D(calArray);
  hist->Draw("colz");
}

//______________________________________________________________________________
template <class T>
TH2* painter::getHistogram2D(const CalDet<T>& calDet, Side side)
{
  static const Mapper& mapper = Mapper::instance();

  const auto title = calDet.getName().c_str();
  std::string name = calDet.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  const char side_name = (side == Side::A) ? 'A' : 'C';

  auto h2D = new TH2F(Form("h_%cside_2D_%s;x (cm);y (cm)", side_name, name.c_str()),
                      Form("%s (%c-Side)", title, side_name),
                      300, -300, 300, 300, -300, 300);

  for (ROC roc; !roc.looped(); ++roc) {
    if (roc.side() != side)
      continue;
    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto val = calDet.getValue(roc, irow, ipad);
        const GlobalPosition2D pos = mapper.getPadCentre(PadROCPos(roc, irow, ipad));
        const int bin = h2D->FindBin(pos.X(), pos.Y());
        if (!h2D->GetBinContent(bin))
          h2D->SetBinContent(bin, val);
      }
    }
  }

  return h2D;
}

//______________________________________________________________________________
template <class T>
TH2* painter::getHistogram2D(const CalArray<T>& calArray)
{
  static const Mapper& mapper = Mapper::instance();

  const size_t position = calArray.getPadSubsetNumber();

  const PadSubset padSubset = calArray.getPadSubset();
  // ===| maximum number of rows and pads |=====================================
  const int nrows = mapper.getNumberOfPadRows(padSubset, position);
  const int npads = mapper.getNumberOfPadsInRow(padSubset, position, nrows - 1);

  // ===| create histogram |====================================================
  const auto title = calArray.getName().c_str();
  std::string name = calArray.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  auto hist = new TH2F(Form("h_%s", name.c_str()),
                       Form("%s;pad row;pad", title),
                       nrows, 0., nrows,
                       npads, -npads / 2, npads / 2);

  // ===| fill hist |===========================================================
  for (int irow = 0; irow < nrows; ++irow) {
    const int padsInRow = mapper.getNumberOfPadsInRow(padSubset, position, irow);
    for (int ipad = 0; ipad < padsInRow; ++ipad) {
      const GlobalPadNumber pad = mapper.getPadNumber(padSubset, position, irow, ipad);
      const auto val = calArray.getValue(pad);
      const int cpad = ipad - padsInRow / 2;
      hist->Fill(irow, cpad, val);
      //printf("%d %d: %f\n", irow, cpad, (double)val);
    }
  }
  return hist;
}

// ===| explicit instantiations |===============================================
// this is required to force the compiler to create instances with the types
// we usually would like to deal with
template void painter::draw<float>(const CalDet<float>& calDet);
template void painter::draw<float>(const CalArray<float>& calArray);
template TH2* painter::getHistogram2D<float>(const CalDet<float>& calDet, Side side);
template TH2* painter::getHistogram2D<float>(const CalArray<float>& calArray);

template void painter::draw<double>(const CalDet<double>& calDet);
template void painter::draw<double>(const CalArray<double>& calArray);
template TH2* painter::getHistogram2D<double>(const CalDet<double>& calDet, Side side);
template TH2* painter::getHistogram2D<double>(const CalArray<double>& calArray);

template void painter::draw<int>(const CalDet<int>& calDet);
template void painter::draw<int>(const CalArray<int>& calArray);
template TH2* painter::getHistogram2D<int>(const CalDet<int>& calDet, Side side);
template TH2* painter::getHistogram2D<int>(const CalArray<int>& calArray);

template void painter::draw<short>(const CalDet<short>& calDet);
template void painter::draw<short>(const CalArray<short>& calArray);
template TH2* painter::getHistogram2D<short>(const CalDet<short>& calDet, Side side);
template TH2* painter::getHistogram2D<short>(const CalArray<short>& calArray);

template void painter::draw<bool>(const CalDet<bool>& calDet);
template void painter::draw<bool>(const CalArray<bool>& calArray);
template TH2* painter::getHistogram2D<bool>(const CalDet<bool>& calDet, Side side);
template TH2* painter::getHistogram2D<bool>(const CalArray<bool>& calArray);
