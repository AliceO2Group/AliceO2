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
#include <fmt/format.h>

#include "TString.h"
#include "TAxis.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"

#include "TPCBase/ROC.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"
#include "TPCBase/Utils.h"

using namespace o2::tpc;

template <class T>
TCanvas* painter::draw(const CalDet<T>& calDet, int nbins1D, float xMin1D, float xMax1D, TCanvas* outputCanvas)
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
                           nbins1D, xMin1D, xMax1D); //TODO: modify ranges

  auto hCside1D = new TH1F(Form("h_Cside_1D_%s", name.c_str()), Form("%s (C-Side)", title),
                           nbins1D, xMin1D, xMax1D); //TODO: modify ranges

  auto hAside2D = new TH2F(Form("h_Aside_2D_%s", name.c_str()), Form("%s (A-Side);x (cm);y (cm)", title),
                           300, -300, 300, 300, -300, 300);

  auto hCside2D = new TH2F(Form("h_Cside_2D_%s", name.c_str()), Form("%s (C-Side);x (cm);y (cm)", title),
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
        if (!hist2D->GetBinContent(bin)) {
          hist2D->SetBinContent(bin, val);
        }
        hist1D->Fill(val);
      }
    }
  }

  if (xMax1D > xMin1D) {
    hAside2D->SetMinimum(xMin1D);
    hAside2D->SetMaximum(xMax1D);
    hCside2D->SetMinimum(xMin1D);
    hCside2D->SetMaximum(xMax1D);
  }

  // ===| Draw histograms |=====================================================
  auto c = outputCanvas;
  if (!c) {
    c = new TCanvas(Form("c_%s", name.c_str()), title, 1000, 1000);
  }
  c->Clear();
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

  return c;
}

//______________________________________________________________________________
template <class T>
TCanvas* painter::draw(const CalArray<T>& calArray)
{
  const auto title = calArray.getName().c_str();
  std::string name = calArray.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  auto c = new TCanvas(Form("c_%s", name.c_str()), title);

  auto hist = getHistogram2D(calArray);
  hist->Draw("colz");

  return c;
}

//______________________________________________________________________________
template <class T>
void painter::fillHistogram2D(TH2& h2D, const CalDet<T>& calDet, Side side)
{
  static const Mapper& mapper = Mapper::instance();

  for (ROC roc; !roc.looped(); ++roc) {
    if (roc.side() != side) {
      continue;
    }

    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto val = calDet.getValue(roc, irow, ipad);
        const GlobalPosition2D pos = mapper.getPadCentre(PadROCPos(roc, irow, ipad));
        const int bin = h2D.FindBin(pos.X(), pos.Y());
        if (!h2D.GetBinContent(bin)) {
          h2D.SetBinContent(bin, val);
        }
      }
    }
  }
}

//______________________________________________________________________________
template <class T>
void painter::fillHistogram2D(TH2& h2D, const CalArray<T>& calArray)
{
  static const Mapper& mapper = Mapper::instance();

  const size_t position = calArray.getPadSubsetNumber();
  const PadSubset padSubset = calArray.getPadSubset();
  const int nrows = mapper.getNumberOfPadRows(padSubset, position);

  // ===| fill hist |===========================================================
  for (int irow = 0; irow < nrows; ++irow) {
    const int padsInRow = mapper.getNumberOfPadsInRow(padSubset, position, irow);
    for (int ipad = 0; ipad < padsInRow; ++ipad) {
      const GlobalPadNumber pad = mapper.getPadNumber(padSubset, position, irow, ipad);
      const auto val = calArray.getValue(pad);
      const int cpad = ipad - padsInRow / 2;
      h2D.Fill(irow, cpad, val);
    }
  }
}

//______________________________________________________________________________
template <class T>
TH2* painter::getHistogram2D(const CalDet<T>& calDet, Side side)
{
  const auto title = calDet.getName().c_str();
  std::string name = calDet.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  const char side_name = (side == Side::A) ? 'A' : 'C';

  auto h2D = new TH2F(Form("h_%cside_2D_%s", side_name, name.c_str()),
                      Form("%s (%c-Side);x (cm);y (cm)", title, side_name),
                      300, -300, 300, 300, -300, 300);

  fillHistogram2D(*h2D, calDet, side);

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
  const int npads = mapper.getNumberOfPadsInRow(padSubset, position, nrows - 1) + 6;

  // ===| create histogram |====================================================
  const auto title = calArray.getName().c_str();
  std::string name = calArray.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  auto hist = new TH2F(Form("h_%s", name.c_str()),
                       Form("%s;pad row;pad", title),
                       nrows, 0., nrows,
                       npads, -npads / 2, npads / 2);

  fillHistogram2D(*hist, calArray);

  return hist;
}

template <typename T>
std::enable_if_t<std::is_signed<T>::value, bool> hasData(const CalArray<T>& cal)
{
  return std::abs(cal.getSum()) > T{0};
}

template <typename T>
std::enable_if_t<std::is_unsigned<T>::value, bool> hasData(const CalArray<T>& cal)
{
  return cal.getSum() > T{0};
}

template <class T>
std::vector<TCanvas*> painter::makeSummaryCanvases(const CalDet<T>& calDet, int nbins1D, float xMin1D, float xMax1D, bool onlyFilled, std::vector<TCanvas*>* outputCanvases)
{

  std::vector<TCanvas*> vecCanvases;

  auto nROCs = calDet.getData().size();

  if (onlyFilled) {
    nROCs = 0;
    for (size_t iroc = 0; iroc < calDet.getData().size(); ++iroc) {
      const auto& roc = calDet.getCalArray(iroc);

      if (hasData(roc)) {
        ++nROCs;
      }
    }

    if (!nROCs) {
      return vecCanvases;
    }
  }

  // ===| set up canvases |===
  TCanvas* cSides = nullptr;
  TCanvas* cROCs1D = nullptr;
  TCanvas* cROCs2D = nullptr;
  const std::string_view calName = calDet.getName();

  if (outputCanvases) {
    if (outputCanvases->size() < 3) {
      LOGP(error, "At least 3 canvases are needed to fill the output, only {} given", outputCanvases->size());
      return vecCanvases;
    }

    cSides = outputCanvases->at(0);
    cROCs1D = outputCanvases->at(1);
    cROCs2D = outputCanvases->at(2);
    cSides->Clear();
    cROCs1D->Clear();
    cROCs2D->Clear();
  } else {

    cROCs1D = new TCanvas(fmt::format("c_ROCs_{}_1D", calName).data(), fmt::format("{} values for each ROC", calName).data(), 1400, 1000);
    cROCs2D = new TCanvas(fmt::format("c_ROCs_{}_2D", calName).data(), fmt::format("{} values for each ROC", calName).data(), 1400, 1000);
  }
  vecCanvases.emplace_back(cSides);
  vecCanvases.emplace_back(cROCs1D);
  vecCanvases.emplace_back(cROCs2D);

  cSides = draw(calDet, nbins1D, xMin1D, xMax1D, cSides);
  cROCs1D->DivideSquare(nROCs);
  cROCs2D->DivideSquare(nROCs);

  // ===| produce plots for each ROC |===
  size_t pad = 1;
  for (size_t iroc = 0; iroc < calDet.getData().size(); ++iroc) {
    const auto& roc = calDet.getCalArray(iroc);

    if (onlyFilled && !hasData(roc)) {
      continue;
    }

    // ===| 1D histogram |===
    auto h1D = new TH1F(fmt::format("h_{}_{:02d}", calName, iroc).data(), fmt::format("{} distribution ROC {:02d};ADC value", calName, iroc).data(), nbins1D, xMin1D, xMax1D);
    for (const auto& val : roc.getData()) {
      h1D->Fill(val);
    }

    // ===| 2D histogram |===
    auto h2D = painter::getHistogram2D(roc);
    h2D->SetStats(0);
    if (xMax1D > xMin1D) {
      h2D->SetMinimum(xMin1D);
      h2D->SetMaximum(xMax1D);
    }
    h2D->SetUniqueID(iroc);

    cROCs1D->cd(pad);
    h1D->Draw();

    cROCs2D->cd(pad);
    h2D->Draw("colz");

    ++pad;
  }

  return vecCanvases;
}

//==============================================================================
std::vector<TCanvas*> painter::makeSummaryCanvases(const std::string_view fileName, const std::string_view calPadNames, int nbins1D, float xMin1D, float xMax1D, bool onlyFilled)
{
  using namespace o2::tpc;

  const auto calPads = utils::readCalPads(fileName, calPadNames);

  std::vector<TCanvas*> vecCanvases;

  for (const auto calPad : calPads) {
    auto canvases = makeSummaryCanvases(*calPad, nbins1D, xMin1D, xMax1D, onlyFilled);
    for (auto c : canvases) {
      vecCanvases.emplace_back(c);
    }
  }

  return vecCanvases;
}

// ===| explicit instantiations |===============================================
// this is required to force the compiler to create instances with the types
// we usually would like to deal with
template TCanvas* painter::draw<float>(const CalDet<float>& calDet, int, float, float, TCanvas*);
template std::vector<TCanvas*> painter::makeSummaryCanvases<float>(const CalDet<float>& calDet, int, float, float, bool, std::vector<TCanvas*>*);
template TCanvas* painter::draw<float>(const CalArray<float>& calArray);
template void painter::fillHistogram2D<float>(TH2& h2D, const CalDet<float>& calDet, Side side);
template void painter::fillHistogram2D<float>(TH2& h2D, const CalArray<float>& calArray);
template TH2* painter::getHistogram2D<float>(const CalDet<float>& calDet, Side side);
template TH2* painter::getHistogram2D<float>(const CalArray<float>& calArray);

template TCanvas* painter::draw<double>(const CalDet<double>& calDet, int, float, float, TCanvas*);
template std::vector<TCanvas*> painter::makeSummaryCanvases<double>(const CalDet<double>& calDet, int, float, float, bool, std::vector<TCanvas*>*);
template TCanvas* painter::draw<double>(const CalArray<double>& calArray);
template TH2* painter::getHistogram2D<double>(const CalDet<double>& calDet, Side side);
template TH2* painter::getHistogram2D<double>(const CalArray<double>& calArray);

template TCanvas* painter::draw<int>(const CalDet<int>& calDet, int, float, float, TCanvas*);
template std::vector<TCanvas*> painter::makeSummaryCanvases<int>(const CalDet<int>& calDet, int, float, float, bool, std::vector<TCanvas*>*);
template TCanvas* painter::draw<int>(const CalArray<int>& calArray);
template TH2* painter::getHistogram2D<int>(const CalDet<int>& calDet, Side side);
template TH2* painter::getHistogram2D<int>(const CalArray<int>& calArray);

template TCanvas* painter::draw<short>(const CalDet<short>& calDet, int, float, float, TCanvas*);
template std::vector<TCanvas*> painter::makeSummaryCanvases<short>(const CalDet<short>& calDet, int, float, float, bool, std::vector<TCanvas*>*);
template TCanvas* painter::draw<short>(const CalArray<short>& calArray);
template TH2* painter::getHistogram2D<short>(const CalDet<short>& calDet, Side side);
template TH2* painter::getHistogram2D<short>(const CalArray<short>& calArray);

template TCanvas* painter::draw<bool>(const CalDet<bool>& calDet, int, float, float, TCanvas*);
template std::vector<TCanvas*> painter::makeSummaryCanvases<bool>(const CalDet<bool>& calDet, int, float, float, bool, std::vector<TCanvas*>*);
template TCanvas* painter::draw<bool>(const CalArray<bool>& calArray);
template TH2* painter::getHistogram2D<bool>(const CalDet<bool>& calDet, Side side);
template TH2* painter::getHistogram2D<bool>(const CalArray<bool>& calArray);
