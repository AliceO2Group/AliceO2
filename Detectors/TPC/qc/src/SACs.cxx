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

#include "TPCQC/SACs.h"
#include "TPCCalibration/SACDrawHelper.h"
#include "TH2Poly.h"
#include "fmt/format.h"
#include "TText.h"
#include "TGraph.h"

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

  c->Divide(1, 2);
  c->cd(1);
  hSideA->Draw("colz");
  c->cd(2);
  hSideC->Draw("colz");

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

TCanvas* SACs::drawIDCZeroScale(TCanvas* outputCanvas) const
{
  TCanvas* canv = nullptr;

  if (outputCanvas) {
    canv = outputCanvas;
  } else {
    canv = new TCanvas("c_sides_SACZero_scale", "SACZero", 1000, 1000);
  }
  canv->cd();
  TGraph* gSACZeroScale = new TGraph(2);
  gSACZeroScale->SetPoint(1,0,mScaleSACZeroAside);
  gSACZeroScale->SetPoint(2,1,mScaleSACZeroCside);

  gSACZeroScale->SetName("gSACZeroScale");
  gSACZeroScale->SetTitle("Scaling Factor (SACZero);Sides;SACZero Total (arb. unit)");
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
    t->DrawText((double)iSide, 0.5, labels[iSide]);
  }
  gSACZeroScale->GetXaxis()->SetLimits(-0.5, 1.5);
  canv->Update();
  canv->Modified();
  gSACZeroScale->SetBit(TObject::kCanDelete);
  t->SetBit(TObject::kCanDelete);
  return canv;
}

/* void SACs::setSACZeroScale(const bool rejectOutlier)
{
  if (rejectOutlier) {
    createOutlierMap(); //GANESHA what's this 
  }

  if (mIDCZero[Side::A]) { //GANESHA check
    mScaleSACZeroAside = getMeanIDC0(Side::A, *mIDCZero[Side::A], rejectOutlier ? mPadFlagsMap.get() : nullptr); //GANESHA check
    scaleIDC0(mScaleSACZeroAside, Side::A);
  } else {
    mScaleSACZeroAside = 0;
  }

  if (mIDCZero[Side::C]) { //GANESHA check
    mScaleSACZeroCside = getMeanIDC0(Side::C, *mIDCZero[Side::C], rejectOutlier ? mPadFlagsMap.get() : nullptr); //GANESHA check
    scaleIDC0(mScaleSACZeroCside, Side::C);
  } else {
    mScaleSACZeroCside = 0;
  }

  /// check if IDC0 total is not zero, in that case no scalling is applied
  if (mScaleSACZeroAside == 0.0 || mScaleSACZeroCside == 0.0) {
    LOGP(error, "Please check the SAC0 total A side {} and C side {}, is zero, therefore no scaling applied!", mScaleSACZeroAside, mScaleSACZeroCside);
    mScaleSACZeroAside = 1.0;
    mScaleSACZeroCside = 1.0;
  }
}*/
