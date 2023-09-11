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

#include <string>
#include <algorithm>
#include <fmt/format.h>
#include <cmath>

#include "TAxis.h"
#include "TMultiGraph.h"
#include "TGraphErrors.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TH2Poly.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TPaletteAxis.h"
#include "TObjArray.h"

#include "CommonUtils/StringUtils.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Painter.h"
#include "TPCBase/Utils.h"
#include "DataFormatsTPC/LaserTrack.h"

// for conversion CalDet to TH3
#include <deque>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <string_view>
#include "MathUtils/Utils.h"

using namespace o2::tpc;

std::array<int, 6> painter::colors = {EColor::kBlack, EColor::kRed + 1, EColor::kOrange + 2, EColor::kGreen + 2, EColor::kBlue + 1, EColor::kMagenta + 1};
std::array<int, 10> painter::markers = {20, 21, 33, 34, 47, 24, 25, 27, 28, 46};

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

std::vector<painter::PadCoordinates> painter::getStackCoordinatesSector()
{
  std::vector<painter::PadCoordinates> padCoords;
  const auto& regInf = Mapper::instance().getMapPadRegionInfo();

  std::vector<GEMstack> stacks;
  stacks.reserve(CRU::CRUperSector);
  for (int cru = 0; cru < CRU::CRUperSector; ++cru) {
    stacks.emplace_back(CRU(cru).gemStack());
  }

  for (int stack = 0; stack < GEMSTACKSPERSECTOR; ++stack) {
    auto& padCoord = padCoords.emplace_back();
    padCoord.xVals.resize(0);
    padCoord.yVals.resize(0);

    const GEMstack currentStack = static_cast<GEMstack>(stack);
    const auto first = std::find(stacks.cbegin(), stacks.cend(), currentStack);
    const auto last = std::find(stacks.crbegin(), stacks.crend(), currentStack);
    const int firstRegion = std::distance(stacks.cbegin(), first);
    const int lastRegion = (stacks.size() - std::distance(stacks.crbegin(), last) - 1);

    for (int region = firstRegion; region <= lastRegion; ++region) {
      const auto& padReg = regInf[region];
      const auto npr = padReg.getNumberOfPadRows();
      const auto ro = padReg.getRowOffset();
      const auto xm = padReg.getXhelper();
      const auto ph = padReg.getPadHeight();
      const auto pw = padReg.getPadWidth();
      const auto yro = padReg.getRadiusFirstRow();
      const auto ks = ph / pw * std::tan(1.74532925199432948e-01);

      for (int irow = 0; irow < npr; ++irow) {
        const auto npads = std::floor(ks * (irow + ro) + xm);
        const int ipad = -npads;
        const auto xPadBottomRight = yro + ph * irow;
        const auto xPadTopRight = yro + ph * (irow + 1);
        const auto ri = xPadBottomRight;
        const auto yPadBottomRight = pw * ipad * xPadBottomRight / (ri + ph / 2);
        const auto yPadTopRight = pw * ipad * xPadTopRight / (ri + ph / 2);
        const auto yPadBottomLeft = pw * (ipad + 1) * xPadBottomRight / (ri + ph / 2);
        const auto yPadTopLeft = pw * (ipad + 1) * xPadTopRight / (ri + ph / 2);
        padCoord.xVals.emplace_back(xPadBottomRight);
        padCoord.yVals.emplace_back(yPadBottomRight);
        padCoord.xVals.emplace_back(xPadTopRight);
        padCoord.yVals.emplace_back(yPadTopRight);
      }
    }
    // mirror coordinates
    for (int i = padCoord.xVals.size() - 1; i >= 0; i--) {
      padCoord.xVals.emplace_back(padCoord.xVals[i]);
      padCoord.yVals.emplace_back(std::abs(padCoord.yVals[i]));
    }
  }
  return padCoords;
}

std::vector<painter::PadCoordinates> painter::getFECCoordinatesSector()
{
  std::vector<painter::PadCoordinates> padCoords;
  const Mapper& mapper = Mapper::instance();
  const auto& regInf = mapper.getMapPadRegionInfo();

  for (int stack = 0; stack < 5; ++stack) {
    const int regionStart = 2 * stack;
    const int regionEnd = regionStart + 2;

    struct fecInf {
      void addPad(int pad, int row) { pad_map[row].emplace_back(pad); }
      const std::vector<int>& getPads(const int row) { return pad_map[row]; }
      std::unordered_map<int, std::vector<int>> pad_map;
    };

    std::unordered_map<size_t, std::array<fecInf, 2>> fecs;
    for (int region = regionStart; region < regionEnd; ++region) {
      for (unsigned int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {
        for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
          const FECInfo& fec = mapper.fecInfo(Mapper::getGlobalPadNumber(lrow, pad, region));
          const size_t fecIndex = fec.getIndex();
          fecs[fecIndex][region - regionStart].addPad(pad, lrow);
        }
      }
    }

    for (auto& fec : fecs) {
      auto& padCoord = padCoords.emplace_back();
      padCoord.xVals.resize(0);
      padCoord.yVals.resize(0);
      for (int j = 0; j < 2; ++j) {
        for (int regionTmp = regionStart; regionTmp < regionEnd; ++regionTmp) {
          const int region = (j == 0) ? regionTmp : (regionStart + regionEnd - regionTmp - 1);
          const auto& padReg = regInf[region];
          const auto npr = padReg.getNumberOfPadRows();
          const auto ro = padReg.getRowOffset();
          const auto xm = padReg.getXhelper();
          const auto ph = padReg.getPadHeight();
          const auto pw = padReg.getPadWidth();
          const auto yro = padReg.getRadiusFirstRow();
          const auto ks = ph / pw * std::tan(1.74532925199432948e-01);

          for (int irowTmp = 0; irowTmp < npr; ++irowTmp) {
            const int irow = (j == 0) ? irowTmp : (npr - irowTmp - 1);
            const auto npads = std::floor(ks * (irow + ro) + xm);
            const std::vector<int>& padsFEC = fec.second[region - regionStart].getPads(irow);
            const int padOff = (j == 0) ? padsFEC.front() : (padsFEC.back() + 1);
            const int ipad = -npads + padOff;
            const auto xPadBottomRight = yro + ph * irow;
            const auto xPadTopRight = yro + ph * (irow + 1);
            const auto ri = xPadBottomRight;
            const auto yPadBottomRight = pw * ipad * xPadBottomRight / (ri + ph / 2);
            const auto yPadTopRight = pw * ipad * xPadTopRight / (ri + ph / 2);
            const auto yPadBottomLeft = pw * (ipad + 1) * xPadBottomRight / (ri + ph / 2);
            const auto yPadTopLeft = pw * (ipad + 1) * xPadTopRight / (ri + ph / 2);
            if (j == 0) {
              padCoord.xVals.emplace_back(xPadBottomRight);
              padCoord.yVals.emplace_back(yPadBottomRight);
              padCoord.xVals.emplace_back(xPadTopRight);
              padCoord.yVals.emplace_back(yPadTopRight);
            } else {
              padCoord.yVals.emplace_back(yPadTopRight);
              padCoord.xVals.emplace_back(xPadTopRight);
              padCoord.yVals.emplace_back(yPadBottomRight);
              padCoord.xVals.emplace_back(xPadBottomRight);
            }
          }
        }
      }
    }
  }
  return padCoords;
}

std::vector<o2::tpc::painter::PadCoordinates> painter::getCoordinates(const Type type)
{
  if (type == Type::Pad) {
    return painter::getPadCoordinatesSector();
  } else if (type == Type::Stack) {
    return painter::getStackCoordinatesSector();
  } else if (type == Type::FEC) {
    return painter::getFECCoordinatesSector();
  } else {
    LOGP(warning, "Wrong Type provided!");
    return std::vector<o2::tpc::painter::PadCoordinates>();
  }
}

std::vector<double> painter::getRowBinningCM(uint32_t roc)
{
  const Mapper& mapper = Mapper::instance();

  int firstRegion = 0, lastRegion = 10;
  if (roc < 36) {
    firstRegion = 0;
    lastRegion = 4;
  } else if (roc < 72) {
    firstRegion = 4;
    lastRegion = 10;
  }

  std::vector<double> binning;

  float lastPadHeight = mapper.getPadRegionInfo(firstRegion).getPadHeight();
  float localX = mapper.getPadRegionInfo(firstRegion).getRadiusFirstRow();
  binning.emplace_back(localX - 3);
  binning.emplace_back(localX);
  for (int iregion = firstRegion; iregion < lastRegion; ++iregion) {
    const auto& regionInfo = mapper.getPadRegionInfo(iregion);
    const auto padHeight = regionInfo.getPadHeight();

    if (std::abs(padHeight - lastPadHeight) > 1e-5) {
      lastPadHeight = padHeight;
      localX = regionInfo.getRadiusFirstRow();
      binning.emplace_back(localX);
    }

    for (int irow = 0; irow < regionInfo.getNumberOfPadRows(); ++irow) {
      localX += lastPadHeight;
      binning.emplace_back(localX);
    }
  }
  binning.emplace_back(localX + 3);

  return binning;
}

std::string painter::getROCTitle(const int rocNumber)
{
  const std::string_view type = (rocNumber < 36) ? "IROC" : "OROC";
  const std::string_view side = ((rocNumber % 36) < 18) ? "A" : "C";
  return fmt::format("{} {}{:02}", type, side, rocNumber % 18);
}

template <class T>
TCanvas* painter::draw(const CalDet<T>& calDet, int nbins1D, float xMin1D, float xMax1D, TCanvas* outputCanvas)
{
  using DetType = CalDet<T>;
  using CalType = CalArray<T>;

  const Mapper& mapper = Mapper::instance();

  // ===| name and title |======================================================
  std::string title = calDet.getName();
  std::string name = calDet.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  std::replace(title.begin(), title.end(), '_', ' ');

  // ===| define histograms |===================================================
  // TODO: auto scaling of ranges based on mean and variance?
  //       for the moment use roots auto scaling

  // set buffer size such that autoscaling uses the full range. This is about 2MB per histogram!
  const int bufferSize = TH1::GetDefaultBufferSize();
  TH1::SetDefaultBufferSize(Sector::MAXSECTOR * mapper.getPadsInSector());

  auto hAside1D = new TH1F(fmt::format("h_Aside_1D_{}", name).data(), fmt::format("{} (A-Side)", title).data(),
                           nbins1D, xMin1D, xMax1D); // TODO: modify ranges

  auto hCside1D = new TH1F(fmt::format("h_Cside_1D_{}", name).data(), fmt::format("{} (C-Side)", title).data(),
                           nbins1D, xMin1D, xMax1D); // TODO: modify ranges

  auto hAside2D = new TH2F(fmt::format("h_Aside_2D_{}", name).data(), fmt::format("{} (A-Side);#it{{x}} (cm);#it{{y}} (cm)", title).data(),
                           330, -270, 270, 330, -270, 270);

  auto hCside2D = new TH2F(fmt::format("h_Cside_2D_{}", name).data(), fmt::format("{} (C-Side);#it{{x}} (cm);#it{{y}} (cm)", title).data(),
                           330, -270, 270, 330, -270, 270);

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
  gStyle->SetOptStat("mr");
  auto c = outputCanvas;
  if (!c) {
    c = new TCanvas(fmt::format("c_{}", name).data(), title.data(), 1000, 1000);
  }
  gStyle->SetStatX(1. - gPad->GetRightMargin());
  gStyle->SetStatY(1. - gPad->GetTopMargin());

  c->Clear();
  c->Divide(2, 2);

  c->cd(1);
  hAside2D->Draw("colz");
  hAside2D->SetStats(0);
  hAside2D->SetTitleOffset(1.05, "XY");
  hAside2D->SetTitleSize(0.05, "XY");
  drawSectorsXY(Side::A);

  c->cd(2);
  hCside2D->Draw("colz");
  hCside2D->SetStats(0);
  hCside2D->SetTitleOffset(1.05, "XY");
  hCside2D->SetTitleSize(0.05, "XY");
  drawSectorsXY(Side::C);

  c->cd(3);
  hAside1D->Draw();

  c->cd(4);
  hCside1D->Draw();

  // reset the buffer size
  TH1::SetDefaultBufferSize(bufferSize);

  // associate histograms to canvas
  hAside1D->SetBit(TObject::kCanDelete);
  hCside1D->SetBit(TObject::kCanDelete);
  hAside2D->SetBit(TObject::kCanDelete);
  hCside2D->SetBit(TObject::kCanDelete);

  return c;
}

//______________________________________________________________________________
template <class T>
TCanvas* painter::draw(const CalArray<T>& calArray)
{
  auto hist = getHistogram2D(calArray);
  std::string name = hist->GetName();
  name[0] = 'c';
  auto c = new TCanvas(fmt::format("c_{}", name).data(), hist->GetTitle());

  hist->Draw("colz");

  // associate histograms to canvas
  hist->SetBit(TObject::kCanDelete);

  return c;
}

//______________________________________________________________________________
template <class T>
void painter::fillHistogram2D(TH2& h2D, const CalDet<T>& calDet, Side side)
{
  const Mapper& mapper = Mapper::instance();

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
  const Mapper& mapper = Mapper::instance();

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
  std::string title = calDet.getName();
  std::string name = calDet.getName();
  std::replace(name.begin(), name.end(), ' ', '_');
  std::replace(title.begin(), title.end(), '_', ' ');
  const char side_name = (side == Side::A) ? 'A' : 'C';

  auto h2D = new TH2F(fmt::format("h_{}side_2D_{}", side_name, name).data(),
                      fmt::format("{} ({}-Side);x (cm);y (cm)", title, side_name).data(),
                      300, -300, 300, 300, -300, 300);

  fillHistogram2D(*h2D, calDet, side);

  return h2D;
}

//______________________________________________________________________________
template <class T>
TH2* painter::getHistogram2D(const CalArray<T>& calArray)
{
  const Mapper& mapper = Mapper::instance();

  const size_t position = calArray.getPadSubsetNumber();
  const PadSubset padSubset = calArray.getPadSubset();

  // ===| maximum number of rows and pads |=====================================
  const int nrows = mapper.getNumberOfPadRows(padSubset, position);
  const int npads = mapper.getNumberOfPadsInRow(padSubset, position, nrows - 1) + 6;

  // ===| create histogram |====================================================
  std::string title = calArray.getName();
  std::string name = calArray.getName();
  std::replace(title.begin(), title.end(), '_', ' ');
  std::replace(name.begin(), name.end(), ' ', '_');

  if (padSubset == PadSubset::ROC) {
    title += fmt::format(" ({})", getROCTitle(position));
  }

  auto hist = new TH2F(fmt::format("h_{}", name).data(),
                       fmt::format("{};pad row;pad", title).data(),
                       nrows, 0., nrows,
                       npads, -npads / 2., npads / 2.);

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
  cSides = draw(calDet, nbins1D, xMin1D, xMax1D, cSides);
  cROCs1D->DivideSquare(nROCs);
  cROCs2D->DivideSquare(nROCs);

  vecCanvases.emplace_back(cSides);
  vecCanvases.emplace_back(cROCs1D);
  vecCanvases.emplace_back(cROCs2D);

  // ===| produce plots for each ROC |===
  size_t pad = 1;
  for (size_t iroc = 0; iroc < calDet.getData().size(); ++iroc) {
    const auto& roc = calDet.getCalArray(iroc);

    if (onlyFilled && !hasData(roc)) {
      continue;
    }

    // ===| 1D histogram |===
    auto h1D = new TH1F(fmt::format("h1_{}_{:02d}", calName, iroc).data(), fmt::format("{} distribution ROC {:02d} ({});ADC value", calName, iroc, getROCTitle(iroc)).data(), nbins1D, xMin1D, xMax1D);
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

    // associate histograms to canvas
    h1D->SetBit(TObject::kCanDelete);
    h2D->SetBit(TObject::kCanDelete);
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
TH2Poly* painter::makeSectorHist(const std::string_view name, const std::string_view title, const float xMin, const float xMax, const float yMin, const float yMax, const Type type)
{
  auto poly = new TH2Poly(name.data(), title.data(), xMin, xMax, yMin, yMax);

  auto coords = painter::getCoordinates(type);
  for (const auto& coord : coords) {
    poly->AddBin(coord.xVals.size(), coord.xVals.data(), coord.yVals.data());
  }

  return poly;
}

//______________________________________________________________________________
TH2Poly* painter::makeSideHist(Side side, const Type type)
{
  const auto s = (side == Side::A) ? "A" : "C";
  auto poly = new TH2Poly(fmt::format("hSide_{}", s).data(), fmt::format("{}-Side;#it{{x}} (cm);#it{{y}} (cm)", s).data(), -270., 270., -270., 270.);

  auto coords = painter::getCoordinates(type);
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
  const Mapper& mapper = Mapper::instance();

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
        const auto pos = mapper.getPadCentre(PadROCPos(calROC.getPadSubsetNumber(), irow, ipad));
        const auto val = calDet.getValue(roc, irow, ipad);
        h2D.Fill(pos.X(), pos.Y(), val);
      }
    }
  }
}

//______________________________________________________________________________
void painter::drawSectorsXY(Side side, int sectorLineColor, int sectorTextColor)
{
  TLine l;
  l.SetLineColor(std::abs(sectorLineColor));

  TLatex latSide;
  latSide.SetTextColor(sectorTextColor);
  latSide.SetTextAlign(22);
  latSide.SetTextSize(0.08);
  if (sectorTextColor >= 0) {
    latSide.DrawLatex(0, 0, (side == Side::C) ? "C" : "A");
  }

  TLatex lat;
  lat.SetTextAlign(22);
  lat.SetTextSize(0.03);
  lat.SetTextColor(std::abs(sectorLineColor));

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
    if (sectorLineColor >= 0) {
      lat.DrawLatex(xText, yText, fmt::format("{}", isector).data());
    }
  }
}

void painter::drawSectorLocalPadNumberPoly(short padTextColor, float lineScalePS)
{
  const Mapper& mapper = Mapper::instance();
  const auto coords = getPadCoordinatesSector();
  TLatex lat;
  lat.SetTextAlign(12);
  lat.SetTextSize(0.002f);
  lat.SetTextColor(padTextColor);
  gStyle->SetLineScalePS(lineScalePS);

  for (unsigned int iregion = 0; iregion < Mapper::NREGIONS; ++iregion) {
    const auto padInf = mapper.getPadRegionInfo(iregion);
    for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[iregion]; ++irow) {
      for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[iregion][irow]; ++ipad) {
        const GlobalPadNumber padNum = o2::tpc::Mapper::getGlobalPadNumber(irow, ipad, iregion);
        const auto coordinate = coords[padNum];
        const float yPos = (coordinate.yVals[0] + coordinate.yVals[2]) / 2;
        const float xPos = (coordinate.xVals[0] + coordinate.xVals[2]) / 2;
        lat.DrawLatex(xPos, yPos, Form("%i", ipad));
      }
    }
  }
}

void painter::drawSectorInformationPoly(short regionLineColor, short rowTextColor)
{
  const Mapper& mapper = Mapper::instance();

  TLatex lat;
  lat.SetTextColor(rowTextColor);
  lat.SetTextSize(0.02f);
  lat.SetTextAlign(12);

  TLine line;
  line.SetLineColor(regionLineColor);

  std::vector<float> radii(Mapper::NREGIONS + 1);
  radii.back() = 247;
  for (unsigned int ireg = 0; ireg < Mapper::NREGIONS; ++ireg) {
    const auto reg = mapper.getPadRegionInfo(ireg);
    const float rad = reg.getRadiusFirstRow();
    radii[ireg] = rad;
    line.DrawLine(rad, -43, rad, 43);
  }

  // draw top region information
  for (unsigned int ireg = 0; ireg < Mapper::NREGIONS; ++ireg) {
    lat.DrawLatex((radii[ireg] + radii[ireg + 1]) / 2, 45, Form("%i", ireg));
  }

  lat.SetTextSize(0.002f);
  lat.SetTextAlign(13);
  // draw local and global rows
  const std::array<float, Mapper::NREGIONS> posRow{16.2f, 18.2f, 20.2f, 22.3f, 26.f, 29.f, 33.f, 35.f, 39.f, 42.5f};
  int globalRow = 0;
  for (unsigned int ireg = 0; ireg < Mapper::NREGIONS; ++ireg) {
    const auto reg = mapper.getPadRegionInfo(ireg);
    const float nRows = reg.getNumberOfPadRows();
    for (int i = 0; i < nRows; ++i) {
      const float padHeight = reg.getPadHeight();
      const float radiusFirstRow = reg.getRadiusFirstRow();
      const float xPos = radiusFirstRow + (i + 0.5f) * padHeight;
      const float yPos = posRow[ireg];
      // row in region
      lat.DrawLatex(xPos, yPos, Form("%i", i));
      lat.DrawLatex(xPos, -yPos, Form("%i", i));
      // row in sector
      const float offs = 0.5f;
      lat.DrawLatex(xPos, yPos + offs, Form("%i", globalRow));
      lat.DrawLatex(xPos, -yPos - offs, Form("%i", globalRow++));
    }
  }
}

template <typename DataT>
TH3F painter::convertCalDetToTH3(const std::vector<CalDet<DataT>>& calDet, const bool norm, const int nRBins, const float rMin, const float rMax, const int nPhiBins, const float zMax)
{
  const int nZBins = calDet.size();
  TH3F histConvSum("hisCalDet", "hisCalDet", nPhiBins, 0, o2::constants::math::TwoPI, nRBins, rMin, rMax, 2 * nZBins, -zMax, zMax); // final converted histogram
  TH3F histConvWeight("histConvWeight", "histConvWeight", nPhiBins, 0, o2::constants::math::TwoPI, nRBins, rMin, rMax, 2 * nZBins, -zMax, zMax);

  typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<float>> polygon;
  const auto coords = o2::tpc::painter::getPadCoordinatesSector(); // coordinates of the pads in one sector

  // create polygons per pad of the input histogram
  std::vector<polygon> geoBin;
  const int nGeoBins = nPhiBins * nRBins;
  geoBin.reserve(nGeoBins);
  for (int iphi = 1; iphi <= nPhiBins; ++iphi) {
    const double phiLow = histConvSum.GetXaxis()->GetBinLowEdge(iphi);
    const double phiUp = histConvSum.GetXaxis()->GetBinUpEdge(iphi);
    for (int ir = 1; ir <= nRBins; ++ir) {
      const double rLow = histConvSum.GetYaxis()->GetBinLowEdge(ir);
      const double rUp = histConvSum.GetYaxis()->GetBinUpEdge(ir);
      const double xPos1 = rLow * std::cos(phiLow);
      const double yPos1 = rLow * std::sin(phiLow);
      const double xPos2 = rLow * std::cos(phiUp);
      const double yPos2 = rLow * std::sin(phiUp);
      const double xPos4 = rUp * std::cos(phiLow);
      const double yPos4 = rUp * std::sin(phiLow);
      const double xPos3 = rUp * std::cos(phiUp);
      const double yPos3 = rUp * std::sin(phiUp);
      // round the values due to problems in intersection in boost with polygons close to each other
      boost::geometry::read_wkt(Form("POLYGON((%.4f %.4f, %.4f %.4f, %.4f %.4f, %.4f %.4f, %.4f %.4f))", xPos1, yPos1, xPos2, yPos2, xPos3, yPos3, xPos4, yPos4, xPos1, yPos1), geoBin.emplace_back());
      boost::geometry::correct(geoBin.back());
    }
  }

  for (unsigned int sector = 0; sector < Mapper::NSECTORS / 2; ++sector) {
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      const unsigned int rowsRegion = o2::tpc::Mapper::ROWSPERREGION[region];
      for (unsigned int iRow = 0; iRow < rowsRegion; ++iRow) {
        const unsigned int padsInRow = o2::tpc::Mapper::PADSPERROW[region][iRow] - 1;
        for (unsigned int iPad = 0; iPad <= padsInRow; ++iPad) {
          const GlobalPadNumber padNum = Mapper::getGlobalPadNumber(iRow, iPad, region);

          const float angDeg = 10.f + sector * 20;
          auto coordinate = coords[padNum];
          coordinate.rotate(angDeg);

          const std::array<double, 2> radiusPadCoord{
            std::sqrt(coordinate.xVals[0] * coordinate.xVals[0] + coordinate.yVals[0] * coordinate.yVals[0]),
            std::sqrt(coordinate.xVals[2] * coordinate.xVals[2] + coordinate.yVals[2] * coordinate.yVals[2]),
          };

          std::array<float, 4> phiPadCoord{
            static_cast<float>(std::atan2(coordinate.yVals[0], coordinate.xVals[0])),
            static_cast<float>(std::atan2(coordinate.yVals[1], coordinate.xVals[1])),
            static_cast<float>(std::atan2(coordinate.yVals[2], coordinate.xVals[2])),
            static_cast<float>(std::atan2(coordinate.yVals[3], coordinate.xVals[3]))};

          for (auto& phi : phiPadCoord) {
            o2::math_utils::bringTo02PiGen(phi);
          }

          // bins of the histogram of the edges of the pad
          const int binRBottomStart = std::clamp(histConvSum.GetYaxis()->FindBin(radiusPadCoord[0]) - 1, 1, nRBins);
          const int binRTopEnd = std::clamp(histConvSum.GetYaxis()->FindBin(radiusPadCoord[1]) + 1, 1, nRBins);
          int binPhiStart = std::min(histConvSum.GetXaxis()->FindBin(phiPadCoord[0]), histConvSum.GetXaxis()->FindBin(phiPadCoord[1]));
          std::clamp(binPhiStart - 1, 1, nPhiBins);
          int binPhiEnd = std::max(histConvSum.GetXaxis()->FindBin(phiPadCoord[2]), histConvSum.GetXaxis()->FindBin(phiPadCoord[3]));
          std::clamp(binPhiEnd + 1, 1, nPhiBins);

          // define boost geoemtry object
          polygon geoPad;
          boost::geometry::read_wkt(Form("POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))", coordinate.xVals[0], coordinate.yVals[0], coordinate.xVals[1], coordinate.yVals[1], coordinate.xVals[2], coordinate.yVals[2], coordinate.xVals[3], coordinate.yVals[3], coordinate.xVals[0], coordinate.yVals[0]), geoPad);
          boost::geometry::correct(geoPad);

          for (int binR = binRBottomStart; binR <= binRTopEnd; ++binR) {
            for (int binPhi = binPhiStart; binPhi <= binPhiEnd; ++binPhi) {
              const int ind = (binPhi - 1) * nRBins + binR - 1;

              std::deque<polygon> output;
              boost::geometry::intersection(geoPad, geoBin[ind], output);
              if (output.empty()) {
                continue;
              }
              const double area = boost::geometry::area(output.front());
              const double fac = area * Mapper::INVPADAREA[region];

              for (int iSide = 0; iSide < 2; ++iSide) {
                const Side side = iSide == 0 ? Side::C : Side::A;
                const unsigned int iCRU = side == Side::A ? (sector * Mapper::NREGIONS + region) : ((sector + Mapper::NSECTORS / 2) * Mapper::NREGIONS + region);
                const CRU cru(iCRU);

                for (int iz = 0; iz < nZBins; ++iz) {
                  const auto val = calDet[iz].getValue(cru, iRow, (side == Side::A) ? iPad : padsInRow - iPad);
                  const int zBin = side == Side::A ? (nZBins + iz + 1) : (nZBins - iz);
                  const auto globBin = histConvSum.GetBin(binPhi, binR, zBin);
                  histConvSum.AddBinContent(globBin, val * fac);
                  if (norm) {
                    histConvWeight.AddBinContent(globBin, fac);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (norm) {
    histConvSum.Divide(&histConvWeight);
  }

  return histConvSum;
}

std::vector<TCanvas*> painter::makeSummaryCanvases(const LtrCalibData& ltr, std::vector<TCanvas*>* outputCanvases)
{
  std::vector<TCanvas*> vecCanvases;

  // ===| set up canvases |===
  TCanvas* cLtrCoverage = nullptr;
  TCanvas* cLtrdEdx = nullptr;
  TCanvas* cCalibValues = nullptr;

  const auto size = 1400;
  if (outputCanvases) {
    if (outputCanvases->size() < 3) {
      LOGP(error, "At least 3 canvases are needed to fill the output, only {} given", outputCanvases->size());
      return vecCanvases;
    }

    cLtrCoverage = outputCanvases->at(0);
    cCalibValues = outputCanvases->at(1);
    cLtrdEdx = outputCanvases->at(2);
    cLtrCoverage->Clear();
    cCalibValues->Clear();
    cLtrdEdx->Clear();
    cLtrCoverage->SetCanvasSize(size, 2. * size * 7 / 24 * 1.1);
    cCalibValues->SetCanvasSize(size, 2. * size * 7 / 24 * 1.1);
    cLtrdEdx->SetCanvasSize(size, 2. * size * 7 / 24 * 1.1);
  } else {
    cLtrCoverage = new TCanvas("cLtrCoverage", "laser track coverage", size, 2. * size * 7 / 24 * 1.1);
    cLtrdEdx = new TCanvas("cLtrdEdx", "laser track average dEdx", size, 2. * size * 7 / 24 * 1.1);
    cCalibValues = new TCanvas("cCalibValues", "calibration values", size, 2. * size * 7 / 24 * 1.1);

    // TODO: add cCalibValues
  }

  auto getLtrStatHist = [](Side side, std::string_view type = "Coverage") -> TH2F* {
    auto sideName = (side == Side::A) ? "A" : "C";
    auto hltr = new TH2F(fmt::format("hltr{}_{}", type, sideName).data(), ";Bundle ID;Track in bundle", 24, 0, 24, 7, 0, 7);
    hltr->SetBit(TObject::kCanDelete);
    hltr->SetStats(0);
    hltr->GetXaxis()->SetNdivisions(406, false);
    hltr->GetYaxis()->SetNdivisions(107, false);
    hltr->SetLabelSize(0.05, "XY");
    hltr->SetTitleSize(0.06, "X");
    hltr->SetTitleSize(0.07, "Y");
    hltr->SetTitleOffset(0.8, "X");
    hltr->SetTitleOffset(0.4, "Y");
    return hltr;
  };

  auto drawNames = [](Side side) {
    const std::array<const std::string_view, 6> namesA{"A01/02", "A04/05", "A07/08", "A10/11", "A13/14", "A16/17"};
    const std::array<const std::string_view, 6> namesC{"C00/01", "C03/04", "C06/07", "C09/10", "C12/13", "C15/16"};
    const auto& names = (side == Side::A) ? namesA : namesC;

    TLatex lat;
    lat.SetTextAlign(22);
    lat.SetTextSize(0.06);

    TLine line;
    for (int i = 0; i < 6; ++i) {
      lat.DrawLatex(2.f + i * 4.f, 7.5, names[i].data());
      if (i < 5) {
        line.DrawLine(4.f + i * 4.f, 0, 4.f + i * 4.f, 8);
      }
    }
  };

  auto hltrCoverageA = getLtrStatHist(Side::A);
  auto hltrCoverageC = getLtrStatHist(Side::C);

  auto hltrdEdxA = getLtrStatHist(Side::A, "dEdx");
  auto hltrdEdxC = getLtrStatHist(Side::C, "dEdx");

  float dEdxSumA = 0.f;
  float dEdxSumC = 0.f;
  int nTrackA = 0;
  int nTrackC = 0;

  for (size_t itrack = 0; itrack < ltr.matchedLtrIDs.size(); ++itrack) {
    const auto id = ltr.matchedLtrIDs.at(itrack);
    const auto dEdx = ltr.dEdx.at(itrack);
    const auto trackID = id % LaserTrack::TracksPerBundle;
    const auto bundleID = (id / LaserTrack::TracksPerBundle) % (LaserTrack::RodsPerSide * LaserTrack::BundlesPerRod);
    const auto sideA = id < (LaserTrack::NumberOfTracks / 2);

    auto hltrCoverage = (sideA) ? hltrCoverageA : hltrCoverageC;
    auto hltrdEdx = (sideA) ? hltrdEdxA : hltrdEdxC;

    hltrCoverage->Fill(bundleID, trackID);
    hltrdEdx->Fill(bundleID, trackID, dEdx);

    if (sideA) {
      dEdxSumA += dEdx;
      ++nTrackA;
    } else {
      dEdxSumC += dEdx;
      ++nTrackC;
    }
  }
  // hltrCoverage->Scale(1.f/float(ltr->processedTFs));

  if (nTrackA > 1) {
    dEdxSumA /= nTrackA;
  }

  if (nTrackC > 1) {
    dEdxSumC /= nTrackC;
  }

  // ===| coverage canvases |===
  cLtrCoverage->Divide(1, 2);

  // A-Side
  cLtrCoverage->cd(1);
  gPad->SetGridx(1);
  gPad->SetGridy(1);
  gPad->SetRightMargin(0.01);
  hltrCoverageA->Draw("col text");
  drawNames(Side::A);

  // C-Side
  cLtrCoverage->cd(2);
  gPad->SetGridx(1);
  gPad->SetGridy(1);
  gPad->SetRightMargin(0.01);
  hltrCoverageC->Draw("col text");
  drawNames(Side::C);

  // ===| dEdx canvases |===
  cLtrdEdx->Divide(1, 2);

  // A-Side
  cLtrdEdx->cd(1);
  gPad->SetGridx(1);
  gPad->SetGridy(1);
  gPad->SetRightMargin(0.01);
  hltrdEdxA->Divide(hltrCoverageA);
  hltrdEdxA->Draw("col text");
  drawNames(Side::A);

  // C-Side
  cLtrdEdx->cd(2);
  gPad->SetGridx(1);
  gPad->SetGridy(1);
  gPad->SetRightMargin(0.01);
  hltrdEdxC->Divide(hltrCoverageC);
  hltrdEdxC->Draw("col text");
  drawNames(Side::C);

  // ===| calibration value canvas |===
  // TODO: Implement
  auto calibValMsg = new TPaveText(0.1, 0.1, 0.9, 0.9, "NDC");
  calibValMsg->SetFillColor(0);
  calibValMsg->SetBorderSize(0);
  calibValMsg->AddText(fmt::format("processedTFs: {}", ltr.processedTFs).data());
  calibValMsg->AddText(fmt::format("dvCorrectionA: {}", ltr.dvCorrectionA).data());
  calibValMsg->AddText(fmt::format("dvCorrectionC: {}", ltr.dvCorrectionC).data());
  calibValMsg->AddText(fmt::format("dvCorrection: {}", ltr.getDriftVCorrection()).data());
  calibValMsg->AddText(fmt::format("dvAbsolute: {}", ltr.refVDrift / ltr.getDriftVCorrection()).data());
  calibValMsg->AddText(fmt::format("dvOffsetA: {}", ltr.dvOffsetA).data());
  calibValMsg->AddText(fmt::format("dvOffsetC: {}", ltr.dvOffsetC).data());
  calibValMsg->AddText(fmt::format("t0A: {}", ltr.getT0A()).data());
  calibValMsg->AddText(fmt::format("t0C: {}", ltr.getT0C()).data());
  calibValMsg->AddText(fmt::format("nTracksA: {}", ltr.nTracksA).data());
  calibValMsg->AddText(fmt::format("nTracksC: {}", ltr.nTracksC).data());
  calibValMsg->AddText(fmt::format("#LTdEdx A#GT: {}", dEdxSumA).data());
  calibValMsg->AddText(fmt::format("#LTdEdx C#GT: {}", dEdxSumC).data());

  cCalibValues->cd();
  calibValMsg->Draw();

  vecCanvases.emplace_back(cLtrCoverage);
  vecCanvases.emplace_back(cCalibValues);
  vecCanvases.emplace_back(cLtrdEdx);
  return vecCanvases;
}

TCanvas* painter::makeJunkDetectionCanvas(const TObjArray* data, TCanvas* outputCanvas)
{
  auto c = outputCanvas;
  if (!c) {
    c = new TCanvas("junk_detection", "Junk Detection", 1000, 1000);
  }

  c->Clear();

  auto strA = (TH2F*)data->At(4);
  auto strB = (TH2F*)data->At(5);

  double statsA[7];
  double statsB[7];

  strA->GetStats(statsA);
  strB->GetStats(statsB);

  auto junkDetectionMsg = new TPaveText(0.1, 0.1, 0.9, 0.9, "NDC");
  junkDetectionMsg->SetFillColor(0);
  junkDetectionMsg->SetBorderSize(0);
  junkDetectionMsg->AddText("Removal Strategy A");
  junkDetectionMsg->AddText(fmt::format("Number of Clusters before Removal: {}", statsA[2]).data());
  junkDetectionMsg->AddText(fmt::format("Removed Fraction: {:.2f}%", statsA[4]).data());
  junkDetectionMsg->AddLine(.0, .5, 1., .5);
  junkDetectionMsg->AddText("Removal Strategy B");
  junkDetectionMsg->AddText(fmt::format("Number of Clusters before Removal: {}", statsB[2]).data());
  junkDetectionMsg->AddText(fmt::format("Removed Fraction: {:.2f}%", statsB[4]).data());

  c->cd();
  junkDetectionMsg->Draw();

  return c;
}

void painter::adjustPalette(TH1* h, float x2ndc, float tickLength)
{
  gPad->Modified();
  gPad->Update();
  auto palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");
  palette->SetX2NDC(x2ndc);
  auto ax = h->GetZaxis();
  ax->SetTickLength(tickLength);
}

TMultiGraph* painter::makeMultiGraph(TTree& tree, std::string_view varX, std::string_view varsY, std::string_view errVarsY, std::string_view cut, bool makeSparse)
{
  bool hasErrors = errVarsY.size() > 0 && (std::count(varsY.begin(), varsY.end(), ':') == std::count(errVarsY.begin(), errVarsY.end(), ':'));

  tree.Draw(fmt::format("{} : {} {} {}", varX, varsY, hasErrors ? " : " : "", hasErrors ? errVarsY : "").data(), cut.data(), "goff");
  const auto nRows = tree.GetSelectedRows();

  // get sort index
  std::vector<size_t> idx(tree.GetSelectedRows());
  std::iota(idx.begin(), idx.end(), static_cast<size_t>(0));
  std::sort(idx.begin(), idx.end(), [&tree](auto a, auto b) { return tree.GetVal(0)[a] < tree.GetVal(0)[b]; });

  auto mgr = new TMultiGraph();
  const auto params = o2::utils::Str::tokenize(varsY.data(), ':');

  for (size_t ivarY = 0; ivarY < params.size(); ++ivarY) {
    auto gr = new TGraphErrors(nRows);
    gr->SetMarkerSize(1);
    gr->SetMarkerStyle(markers[ivarY % markers.size()]);
    gr->SetMarkerColor(colors[ivarY % colors.size()]);
    gr->SetLineColor(colors[ivarY % colors.size()]);
    gr->SetNameTitle(params[ivarY].data(), params[ivarY].data());
    for (Long64_t iEntry = 0; iEntry < nRows; ++iEntry) {
      if (makeSparse) {
        gr->SetPoint(iEntry, iEntry + 0.5, tree.GetVal(ivarY + 1)[idx[iEntry]]);
      } else {
        gr->SetPoint(iEntry, tree.GetVal(0)[idx[iEntry]], tree.GetVal(ivarY + 1)[idx[iEntry]]);
      }
      if (hasErrors) {
        gr->SetPointError(iEntry, 0, tree.GetVal(ivarY + 1 + params.size())[idx[iEntry]]);
      }
    }

    mgr->Add(gr, "lp");
  }

  if (makeSparse) {
    auto xax = mgr->GetXaxis();
    xax->Set(nRows, 0., static_cast<Double_t>(nRows));
    for (Long64_t iEntry = 0; iEntry < nRows; ++iEntry) {
      xax->SetBinLabel(iEntry + 1, fmt::format("{}", tree.GetVal(0)[idx[iEntry]]).data());
    }
    xax->LabelsOption("v");
  }

  return mgr;
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

template TH3F painter::convertCalDetToTH3<float>(const std::vector<CalDet<float>>&, const bool, const int, const float, const float, const int, const float);
