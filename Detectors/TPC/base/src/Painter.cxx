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
#include <cmath>

#include "TString.h"
#include "TAxis.h"
#include "TH1.h"
#include "TH2.h"
#include "TH2Poly.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"
#include "TPCBase/Utils.h"

using namespace o2::tpc;

std::vector<painter::PadCoordinates> painter::getPadCoordinatesSector()
{
  std::vector<painter::PadCoordinates> padCoords;

  const auto& regInf = Mapper::instance().getMapPadRegionInfo();

  for (const auto& padReg : regInf) {
    const auto npr = padReg.getNumberOfPadRows();
    const auto ro = padReg.getRowOffset();
    const auto xm = padReg.getXhelper();
    const auto ph = padReg.getPadHeight();
    const auto pw = padReg.getPadWidth();
    const auto yro = padReg.getRadiusFirstRow();
    const auto ks = ph / pw * std::tan(1.74532925199432948e-01);

    for (int irow = 0; irow < npr; ++irow) {
      const auto npads = std::floor(ks * (irow + ro) + xm);
      for (int ipad = -npads; ipad < npads; ++ipad) {
        const auto xPadBottomRight = yro + ph * irow;
        const auto xPadTopRight = yro + ph * (irow + 1);
        const auto ri = xPadBottomRight;
        const auto yPadBottomRight = pw * ipad * xPadBottomRight / (ri + ph / 2);
        const auto yPadTopRight = pw * ipad * xPadTopRight / (ri + ph / 2);
        const auto yPadBottomLeft = pw * (ipad + 1) * xPadBottomRight / (ri + ph / 2);
        const auto yPadTopLeft = pw * (ipad + 1) * xPadTopRight / (ri + ph / 2);
        auto& padCoord = padCoords.emplace_back();
        padCoord.xVals = {xPadBottomRight, xPadTopRight, xPadTopRight, xPadBottomRight};
        padCoord.yVals = {yPadBottomRight, yPadTopRight, yPadTopLeft, yPadBottomLeft};
      }
    }
  }

  return padCoords;
}

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
  hAside2D->SetStats(0);
  drawSectorsXY(Side::A);

  c->cd(2);
  hCside2D->Draw("colz");
  hCside2D->SetStats(0);
  drawSectorsXY(Side::C);

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

//______________________________________________________________________________
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

//______________________________________________________________________________
TH2Poly* painter::makeSectorHist(const std::string_view name, const std::string_view title)
{
  auto poly = new TH2Poly(name.data(), title.data(), 83.65, 247.7, -43.7, 43.7);

  auto coords = painter::getPadCoordinatesSector();
  for (const auto& coord : coords) {
    poly->AddBin(coord.xVals.size(), coord.xVals.data(), coord.yVals.data());
  }

  return poly;
}

//______________________________________________________________________________
TH2Poly* painter::makeSideHist(Side side)
{
  const auto s = (side == Side::A) ? "A" : "C";
  auto poly = new TH2Poly(fmt::format("hSide_{}", s).data(), fmt::format("{}-Side;#it{{x}} (cm);#it{{y}} (cm)", s).data(), -270., 270., -270., 270.);

  auto coords = painter::getPadCoordinatesSector();
  for (int isec = 0; isec < 18; ++isec) {
    const float angDeg = 10.f + isec * 20;
    for (auto coord : coords) {
      coord.rotate(angDeg);
      poly->AddBin(coord.xVals.size(), coord.xVals.data(), coord.yVals.data());
    }
  }

  return poly;
}

//______________________________________________________________________________
template <class T>
void painter::fillPoly2D(TH2Poly& h2D, const CalDet<T>& calDet, Side side)
{
  static const Mapper& mapper = Mapper::instance();

  int bin = 1;
  for (const auto& calROC : calDet.getData()) {
    ROC roc(calROC.getPadSubsetNumber());
    if (roc.side() != side) {
      continue;
    }

    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      const int padMax = mapper.getNumberOfPadsInRowROC(roc, irow) - 1;
      for (int ipad = 0; ipad <= padMax; ++ipad) {
        const auto val = calDet.getValue(roc, irow, (side == Side::A) ? ipad : padMax - ipad); // C-Side is mirrored
        h2D.SetBinContent(bin++, val);
      }
    }
  }
}

//______________________________________________________________________________
void painter::drawSectorsXY(Side side, int sectorLineColor, int sectorTextColor)
{
  TLine l;
  l.SetLineColor(sectorLineColor);

  TLine l2;
  l2.SetLineColor(sectorLineColor);
  l2.SetLineStyle(kDotted);

  TLatex latSide;
  latSide.SetTextColor(sectorTextColor);
  latSide.SetTextAlign(22);
  latSide.SetTextSize(0.08);
  latSide.DrawLatex(0, 0, (side == Side::C) ? "C" : "A");

  TLatex lat;
  lat.SetTextAlign(22);
  lat.SetTextSize(0.03);
  lat.SetTextColor(sectorLineColor);

  constexpr float phiWidth = float(SECPHIWIDTH);
  const float rFactor = std::cos(phiWidth / 2.);
  const float rLow = 83.65 / rFactor;
  const float rIROCup = 133.3 / rFactor;
  const float rOROClow = 133.5 / rFactor;
  const float rOROC12 = 169.75 / rFactor;
  const float rOROC23 = 207.85 / rFactor;
  const float rOut = 247.7 / rFactor;
  const float rText = rLow * rFactor * 3. / 4.;

  for (Int_t isector = 0; isector < 18; ++isector) {
    const float sinR = std::sin(phiWidth * isector);
    const float cosR = std::cos(phiWidth * isector);

    const float sinL = std::sin(phiWidth * ((isector + 1) % 18));
    const float cosL = std::cos(phiWidth * ((isector + 1) % 18));

    const float sinText = std::sin(phiWidth * (isector + 0.5));
    const float cosText = std::cos(phiWidth * (isector + 0.5));

    const float xR1 = rLow * cosR;
    const float yR1 = rLow * sinR;
    const float xR2 = rOut * cosR;
    const float yR2 = rOut * sinR;

    const float xL1 = rLow * cosL;
    const float yL1 = rLow * sinL;
    const float xL2 = rOut * cosL;
    const float yL2 = rOut * sinL;

    const float xOROCmup1 = rOROClow * cosR;
    const float yOROCmup1 = rOROClow * sinR;
    const float xOROCmup2 = rOROClow * cosL;
    const float yOROCmup2 = rOROClow * sinL;

    const float xIROCmup1 = rIROCup * cosR;
    const float yIROCmup1 = rIROCup * sinR;
    const float xIROCmup2 = rIROCup * cosL;
    const float yIROCmup2 = rIROCup * sinL;

    const float xO121 = rOROC12 * cosR;
    const float yO121 = rOROC12 * sinR;
    const float xO122 = rOROC12 * cosL;
    const float yO122 = rOROC12 * sinL;

    const float xO231 = rOROC23 * cosR;
    const float yO231 = rOROC23 * sinR;
    const float xO232 = rOROC23 * cosL;
    const float yO232 = rOROC23 * sinL;

    const float xText = rText * cosText;
    const float yText = rText * sinText;

    // left side line
    l.DrawLine(xR1, yR1, xR2, yR2);

    // IROC inner line
    l.DrawLine(xR1, yR1, xL1, yL1);

    // IROC end line
    l.DrawLine(xIROCmup1, yIROCmup1, xIROCmup2, yIROCmup2);

    // OROC start line
    l.DrawLine(xOROCmup1, yOROCmup1, xOROCmup2, yOROCmup2);

    // OROC1 - OROC2 line
    l.DrawLine(xO121, yO121, xO122, yO122);

    // OROC2 - OROC3 line
    l.DrawLine(xO231, yO231, xO232, yO232);

    // IROC inner line
    l.DrawLine(xR2, yR2, xL2, yL2);

    // sector numbers
    lat.DrawLatex(xText, yText, fmt::format("{}", isector).data());
  }
}
// ===| explicit instantiations |===============================================
// this is required to force the compiler to create instances with the types
// we usually would like to deal with
template TCanvas* painter::draw<float>(const CalDet<float>& calDet, int, float, float, TCanvas*);
template std::vector<TCanvas*> painter::makeSummaryCanvases<float>(const CalDet<float>& calDet, int, float, float, bool, std::vector<TCanvas*>*);
template TCanvas* painter::draw<float>(const CalArray<float>& calArray);
template void painter::fillHistogram2D<float>(TH2& h2D, const CalDet<float>& calDet, Side side);
template void painter::fillPoly2D<float>(TH2Poly& h2D, const CalDet<float>& calDet, Side side);
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
