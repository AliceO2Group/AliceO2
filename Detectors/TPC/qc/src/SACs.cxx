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

#include <Framework/Logger.h>
#include "TPCQC/SACs.h"
#include "TPCCalibration/SACDrawHelper.h"
#include "TH2Poly.h"
#include "fmt/format.h"
#include "TText.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TColor.h"

ClassImp(o2::tpc::qc::SACs);
using namespace o2::tpc::qc;

float SACs::getSACOneVal(const Side side, unsigned int integrationInterval) const
{
  return !mSACOne[side] ? -1 : mSACOne[side]->getValue(side, integrationInterval);
}

TCanvas* SACs::drawSACTypeSides(const SACType type, const unsigned int integrationInterval, const int minZ, const int maxZ, TCanvas* canv)
{
  std::string name;
  std::function<float(const unsigned int, const unsigned int)> SACFunc;
  if (type == o2::tpc::SACType::IDC) {
    SACFunc = [this, integrationInterval](const unsigned int sector, const unsigned int stack) {
      return this->getSACValue(getStack(sector, stack), integrationInterval);
    };
    name = "SAC";
  } else if (type == o2::tpc::SACType::IDCZero) {
    SACFunc = [this](const unsigned int sector, const unsigned int stack) {
      return this->getSACZeroVal(getStack(sector, stack));
    };
    name = "SACZero";
  } else if (type == o2::tpc::SACType::IDCDelta) {
    SACFunc = [this, integrationInterval](const unsigned int sector, const unsigned int stack) {
      return this->getSACDeltaVal(getStack(sector, stack), integrationInterval);
    };
    name = "SACDelta";
  } else if (type == o2::tpc::SACType::IDCOutlier) {
    SACFunc = [this](const unsigned int sector, const unsigned int stack) {
      return this->getSACRejection(getStack(sector, stack));
    };
    name = "SACZero Outliers (Yellow)";
  }

  auto c = canv;
  if (!c) {
    c = new TCanvas(fmt::format("c_sides_{}", name).data(), fmt::format("sides_{}", name).data(), 500, 1000);
  }

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;
  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(type);

  auto hSideA = SACDrawHelper::drawSide(drawFun, o2::tpc::Side::A, zAxisTitle);
  auto hSideC = SACDrawHelper::drawSide(drawFun, o2::tpc::Side::C, zAxisTitle);

  hSideA->SetTitle(fmt::format("{} ({}-Side)", name.data(), "A").data());
  hSideC->SetTitle(fmt::format("{} ({}-Side)", name.data(), "C").data());

  if (minZ < maxZ) {
    hSideA->SetMinimum(minZ);
    hSideC->SetMinimum(minZ);
    hSideA->SetMaximum(maxZ);
    hSideC->SetMaximum(maxZ);
  }

  if (type == o2::tpc::SACType::IDCOutlier) {
    Double_t levels[] = {-3., 0., 3.};
    hSideA->SetContour(3, levels);
    hSideC->SetContour(3, levels);
  }

  c->Divide(1, 2);
  c->cd(1);
  hSideA->Draw("colz PFC");
  c->cd(2);
  hSideC->Draw("colz PFC");

  hSideA->SetBit(TObject::kCanDelete);
  hSideC->SetBit(TObject::kCanDelete);

  return c;
}

TCanvas* SACs::drawSACOneCanvas(int nbins1D, float xMin1D, float xMax1D, int integrationIntervals, TCanvas* outputCanvas) const
{
  auto* canv = outputCanvas;

  if (!canv) {
    canv = new TCanvas("c_sides_SAC1_1D", "SAC1 1D distribution for each side", 1000, 1000);
  }

  auto hAside1D = new TH1F("h_SAC1_1D_ASide", "SAC1 distribution over integration intervals A-Side", nbins1D, xMin1D, xMax1D);
  auto hCside1D = new TH1F("h_SAC1_1D_CSide", "SAC1 distribution over integration intervals C-Side", nbins1D, xMin1D, xMax1D);

  hAside1D->GetXaxis()->SetTitle("SAC1");
  hAside1D->SetTitleOffset(1.05, "XY");
  hAside1D->SetTitleSize(0.05, "XY");
  hCside1D->GetXaxis()->SetTitle("SAC1");
  hCside1D->SetTitleOffset(1.05, "XY");
  hCside1D->SetTitleSize(0.05, "XY");
  if (integrationIntervals <= 0) {
    integrationIntervals = std::min(mSACOne[Side::A]->mSACOne[Side::A].getNIDCs(), mSACOne[Side::C]->mSACOne[Side::C].getNIDCs());
  }
  for (unsigned int integrationInterval = 0; integrationInterval < integrationIntervals; ++integrationInterval) {
    hAside1D->Fill(getSACOneVal(Side::A, integrationInterval));
    hCside1D->Fill(getSACOneVal(Side::C, integrationInterval));
  }

  canv->Divide(1, 2);
  canv->cd(1);
  hAside1D->Draw();
  canv->cd(2);
  hCside1D->Draw();

  hAside1D->SetBit(TObject::kCanDelete);
  hCside1D->SetBit(TObject::kCanDelete);

  return canv;
}

TCanvas* SACs::drawFourierCoeffSAC(Side side, int nbins1D, float xMin1D, float xMax1D, TCanvas* outputCanvas) const
{
  auto* canv = outputCanvas;
  if (!canv) {
    canv = new TCanvas(fmt::format("c_FourierCoefficients_1D_{}Side", (side == Side::A) ? "A" : "C").data(), fmt::format("1D distributions of Fourier Coefficients ({}-Side)", (side == Side::A) ? "A" : "C").data(), 1000, 1000);
  }

  std::vector<TH1F*> histos;

  for (int i = 0; i < mFourierSAC->mCoeff[side].getNCoefficientsPerTF(); i++) {
    histos.emplace_back(new TH1F(fmt::format("h_FourierCoeff{}_{}Side", i, (side == Side::A) ? "A" : "C").data(), fmt::format("1D distribution of Fourier Coefficient {} ({}-Side)", i, (side == Side::A) ? "A" : "C").data(), nbins1D, xMin1D, xMax1D));
    histos.back()->GetXaxis()->SetTitle(fmt::format("Fourier Coefficient {}", i).data());
  }

  const auto& coeffs = mFourierSAC->mCoeff[side].getFourierCoefficients();
  const auto nCoeffPerTF = mFourierSAC->mCoeff[side].getNCoefficientsPerTF();

  for (int i = 0; i < mFourierSAC->mCoeff[side].getNCoefficients(); i++) {
    histos.at(i % nCoeffPerTF)->Fill(coeffs.at(i));
  }

  canv->DivideSquare(mFourierSAC->mCoeff[side].getNCoefficientsPerTF());

  size_t pad = 1;

  for (const auto& hist : histos) {
    canv->cd(pad);
    hist->SetTitleOffset(1.05, "XY");
    hist->SetTitleSize(0.05, "XY");
    hist->Draw();
    hist->SetBit(TObject::kCanDelete);
    pad++;
  }

  return canv;
}

TCanvas* SACs::drawSACZeroScale(TCanvas* outputCanvas) const
{
  TCanvas* canv = nullptr;

  if (outputCanvas) {
    canv = outputCanvas;
  } else {
    canv = new TCanvas("c_sides_SACZero_ScaleFactor", "SACZero", 1000, 1000);
  }
  canv->cd();

  double xsides[] = {0, 1};
  double yTotSAC0[] = {mScaleSACZeroAside, mScaleSACZeroCside};
  auto gSACZeroScale = new TGraph(2, xsides, yTotSAC0);

  gSACZeroScale->SetName("g_SAC0ScaleFactor");
  gSACZeroScale->SetTitle(Form("Scaling Factor (SACZero), Rejection Factor %.2f;Sides;SACZero Total (arb. unit)", mSACZeroMaxDeviation));
  gSACZeroScale->SetMarkerColor(kBlue);
  gSACZeroScale->SetMarkerStyle(21);
  gSACZeroScale->SetMarkerSize(1);
  gSACZeroScale->GetXaxis()->SetLabelColor(0);
  gSACZeroScale->GetXaxis()->CenterTitle();

  gSACZeroScale->Draw("ap");
  // Draw labels on the x axis
  TText* t = new TText();
  t->SetTextSize(0.035);
  const char* labels[2] = {"A-Side", "C-Side"};
  for (Int_t iSide = 0; iSide < 2; iSide++) {
    t->DrawText((double)iSide + 0.1, yTotSAC0[iSide], labels[iSide]);
  }
  gSACZeroScale->GetXaxis()->SetLimits(-0.5, 1.5);
  canv->Update();
  canv->Modified();
  gSACZeroScale->SetBit(TObject::kCanDelete);
  t->SetBit(TObject::kCanDelete);
  return canv;
}

void SACs::scaleSAC0(const float factor, const Side side)
{
  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : (sectorStart * SIDES);
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    for (unsigned int stack = 0; stack < GEMSTACKSPERSECTOR; ++stack) {
      float SACZeroValue = getSACZeroVal(getStack(sector, stack));
      if (factor != 0.) {
        setSACZeroVal(SACZeroValue / factor, getStack(sector, stack));
      }
    }
  }
}

void SACs::setSACZeroScale(const bool rejectOutlier)
{
  if (mSACZero) {
    mScaleSACZeroAside = getMeanSACZero(Side::A, rejectOutlier);
    scaleSAC0(abs(mScaleSACZeroAside), Side::A);

    mScaleSACZeroCside = getMeanSACZero(Side::C, rejectOutlier);
    scaleSAC0(abs(mScaleSACZeroCside), Side::C);
  } else {
    mScaleSACZeroAside = 0;
    mScaleSACZeroCside = 0;
  }

  /// check if SACZero total is not zero, in that case no scaling
  if (mScaleSACZeroAside == 0.0 || mScaleSACZeroCside == 0.0) {
    LOGP(error, "Please check the SAC0 total A side {} and C side {}, is zero, therefore no scaling applied!", mScaleSACZeroAside, mScaleSACZeroCside);
    mScaleSACZeroAside = 1.0;
    mScaleSACZeroCside = 1.0;
  }
}

float SACs::getMeanSACZero(const o2::tpc::Side side, bool rejectOutliers)
{
  float medianSACZero[GEMSTACKSPERSECTOR] = {0.};

  if (rejectOutliers) {
    // Calculate Median for each row for rejection of outliers
    std::vector<float> SACZeroValues[GEMSTACKSPERSECTOR];

    unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
    unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : (sectorStart * SIDES);
    for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
      for (unsigned int stack = 0; stack < GEMSTACKSPERSECTOR; ++stack) {
        SACZeroValues[stack].push_back(getSACZeroVal(getStack(sector, stack)));
      }
    }

    for (unsigned int stack = 0; stack < GEMSTACKSPERSECTOR; stack++) {
      size_t n = SACZeroValues[stack].size() / 2;
      sort(SACZeroValues[stack].begin(), SACZeroValues[stack].end());
      if (SACZeroValues[stack].size() % 2 != 0) {
        medianSACZero[stack] = SACZeroValues[stack].at(n);
      } else {
        medianSACZero[stack] = (SACZeroValues[stack].at(n) + SACZeroValues[stack].at(n - 1)) / 2.;
      }
    }
  }

  // Calculate Mean
  float meanSACZero = 0;
  float SACZeroChannelCounter = 0.;

  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : (sectorStart * SIDES);
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    for (unsigned int stack = 0; stack < GEMSTACKSPERSECTOR; ++stack) {

      float SACZeroVal = getSACZeroVal(getStack(sector, stack));

      if (rejectOutliers) {
        if (abs((SACZeroVal - medianSACZero[stack]) / medianSACZero[stack]) >= mSACZeroMaxDeviation) {
          mIsSACZeroOutlier[getStack(sector, stack)] = 1;
          continue;
        }
      }
      mIsSACZeroOutlier[getStack(sector, stack)] = -1;
      meanSACZero += SACZeroVal;
      SACZeroChannelCounter += 1.;
    }
  }
  if (SACZeroChannelCounter > 0.) {
    meanSACZero /= SACZeroChannelCounter;
  }
  return meanSACZero;
}
