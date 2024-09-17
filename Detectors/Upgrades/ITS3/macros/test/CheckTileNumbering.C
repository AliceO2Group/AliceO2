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

/// \file CheckTileNumberingITS3.C
/// \brief Macro to quickly check the tile numbering scheme
/// \autor Felix Schlepper felix.schlepper@cern.ch

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TCanvas.h>
#include <TH2F.h>
#include <TPad.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLine.h>
#include <TError.h>

#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SpecsV2.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "DataFormatsITSMFT/NoiseMap.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#endif

constexpr float halfTiles = o2::its3::constants::segment::nTilesPerSegment / 2;
constexpr float halfZ = o2::its3::constants::segment::lengthSensitive / 2.f;

o2::itsmft::NoiseMap* mNoiseMap{nullptr};

void Draw(TH2* in, float phiOff1, float phiOff2, int n = 3)
{
  const int nBinsX = in->GetXaxis()->GetNbins();
  const int nBinsY = in->GetYaxis()->GetNbins();

  // Create a 2D histogram
  TH2F* hChess = new TH2F(Form("%s_chess", in->GetName()), in->GetTitle(),
                          nBinsX, in->GetXaxis()->GetXmin(), in->GetXaxis()->GetXmax(),
                          nBinsY, in->GetYaxis()->GetXmin(), in->GetYaxis()->GetXmax());

  // Loop over all bins and set the content in a chess pattern
  for (int i = 1; i <= nBinsX; ++i) {
    for (int j = 1; j <= nBinsY; ++j) {
      // Alternating pattern: (i+j) % 2 gives 0 or 1
      int content = (((i + j) % 2) == 1) ? 2 : 0;
      if (mNoiseMap != nullptr) {
        int id = in->GetBinContent(i, j);
        if (mNoiseMap->isFullChipMasked(id)) {
          content = 1;
        }
      }
      hChess->SetBinContent(i, j, content);
    }
  }
  hChess->GetXaxis()->SetTitle(in->GetXaxis()->GetTitle());
  hChess->GetYaxis()->SetTitle(in->GetYaxis()->GetTitle());
  hChess->GetXaxis()->SetTickLength(0.003);
  hChess->GetYaxis()->SetTickLength(0.003);
  hChess->GetYaxis()->SetTitleOffset(0.5);
  hChess->SetFillColorAlpha(18, 0.6);
  hChess->Draw("col");

  // vertical lines
  for (int i{1}; i <= 11; ++i) {
    float z = -halfZ + i * o2::its3::constants::rsu::length;
    auto l = new TLine(z, phiOff1, z, phiOff2);
    l->SetLineStyle(kDashed);
    l->Draw("same");
  }
  // horizontal lines
  float dPhi = (phiOff2 - phiOff1) / (float)n;
  for (int i{1}; i < n; ++i) {
    float phi = phiOff1 + i * dPhi;
    auto l = new TLine(-halfZ, phi, halfZ, phi);
    l->SetLineStyle(kDashed);
    l->Draw("same");
  }

  in->Draw("TEXT;same");
  gPad->Update();
}

void CheckTileNumbering(const std::string& inputGeom = "", const std::string& deadmap = "", bool write = false)
{
  gStyle->SetOptStat(0);
  gStyle->SetHistMinimumZero();
  const Int_t NRGBs = 3;
  Int_t colors[NRGBs] = {kWhite, kRed, kGray};
  TColor::SetPalette(NRGBs, colors, 1.0);

  const float phiOffsetL0 = std::asin(o2::its3::constants::equatorialGap / 2.f / o2::its3::constants::radii[0]);
  const float phiOffsetL1 = std::asin(o2::its3::constants::equatorialGap / 2.f / o2::its3::constants::radii[1]);
  const float phiOffsetL2 = std::asin(o2::its3::constants::equatorialGap / 2.f / o2::its3::constants::radii[2]);
  const float markerSize = 0.5;
  auto hL0Up = new TH2F("hL0Up", "L0 Upper;z (cm); #varphi (rad)", halfTiles, -halfZ, halfZ, 2 * 3, phiOffsetL0, TMath::Pi() - phiOffsetL0);
  hL0Up->SetMarkerSize(markerSize);
  auto hL0Bot = new TH2F("hL0Bot", "L0 Bottom;z (cm); #varphi (rad)", halfTiles, -halfZ, halfZ, 2 * 3, TMath::Pi() + phiOffsetL0, 2 * TMath::Pi() - phiOffsetL0);
  hL0Bot->SetMarkerSize(markerSize);
  auto hL1Up = new TH2F("hL1Up", "L1 Upper;z (cm); #varphi (rad)", halfTiles, -halfZ, halfZ, 2 * 4, phiOffsetL1, TMath::Pi() - phiOffsetL1);
  hL1Up->SetMarkerSize(markerSize);
  auto hL1Bot = new TH2F("hL1Bot", "L1 Bottom;z (cm); #varphi (rad)", halfTiles, -halfZ, halfZ, 2 * 4, TMath::Pi() + phiOffsetL1, 2 * TMath::Pi() - phiOffsetL1);
  hL1Bot->SetMarkerSize(markerSize);
  auto hL2Up = new TH2F("hL2Up", "L2 Upper;z (cm); #varphi (rad)", halfTiles, -halfZ, halfZ, 2 * 5, phiOffsetL2, TMath::Pi() - phiOffsetL2);
  hL2Up->SetMarkerSize(markerSize);
  auto hL2Bot = new TH2F("hL2Bot", "L2 Bottom;z (cm); #varphi (rad)", halfTiles, -halfZ, halfZ, 2 * 5, TMath::Pi() + phiOffsetL2, 2 * TMath::Pi() - phiOffsetL2);
  hL2Bot->SetMarkerSize(markerSize);
  std::array<TH2F*, 6> hSensors{hL0Up, hL0Bot, hL1Up, hL1Bot, hL2Up, hL2Bot};

  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  if (!deadmap.empty()) {
    std::unique_ptr<TFile> f{TFile::Open(deadmap.c_str(), "READ")};
    mNoiseMap = f->Get<o2::itsmft::NoiseMap>("ccdb_object");
  }

  std::ofstream fs;
  if (write) {
    fs.open("its3Numbering.txt", std::ios::out);
    if (!fs.is_open()) {
      Error("", "Cannot open file for writing!");
      write = false;
    }
  }

  float xFlat{0}, yFlat{0}, x{0}, y{0}, z{0};
  for (unsigned int iDet{0}; iDet <= o2::its3::constants::detID::l2IDEnd; ++iDet) {
    int sensorID = o2::its3::constants::detID::getSensorID(iDet);
    int layerID = o2::its3::constants::detID::getDetID2Layer(iDet);
    o2::its3::SuperSegmentations[layerID].flatToCurved(xFlat, 0., x, y);
    o2::math_utils::Point3D<float> locC{x, y, z};
    auto gloC = gman->getMatrixL2G(iDet)(locC);
    float phi = o2::math_utils::to02Pi(std::atan2(gloC.Y(), gloC.X()));
    /* phi = (phi >= 0) ? phi : (2 * TMath::Pi() + phi); */
    float z = gloC.Z();
    auto xBin = hSensors[sensorID]->GetXaxis()->FindBin(z);
    auto yBin = hSensors[sensorID]->GetYaxis()->FindBin(phi);
    hSensors[sensorID]->SetBinContent(xBin, yBin, iDet);
    if (write) {
      fs << std::setfill('0') << std::setw(4) << iDet << " -> Layer " << layerID << " Sensor " << sensorID << " phi=" << phi << " z=" << z << " Path: " << gman->getMatrixPath(iDet)
         << "\n";
    }
  }

  auto c = new TCanvas("cL0", "Numbering Layer 0", 1000, 700);
  c->Divide(1, 2);
  c->cd(1);
  Draw(hL0Up, phiOffsetL0, TMath::Pi() - phiOffsetL0, 3);
  c->cd(2);
  Draw(hL0Bot, TMath::Pi() + phiOffsetL0, 2 * TMath::Pi() - phiOffsetL0, 3);
  c->SaveAs("its3_tile_layer0.pdf");

  c = new TCanvas("cL1", "Numbering Layer 1", 1000, 700);
  c->Divide(1, 2);
  c->cd(1);
  Draw(hL1Up, phiOffsetL1, TMath::Pi() - phiOffsetL1, 4);
  c->cd(2);
  Draw(hL1Bot, TMath::Pi() + phiOffsetL1, 2 * TMath::Pi() - phiOffsetL1, 4);
  c->SaveAs("its3_tile_layer1.pdf");

  c = new TCanvas("cL2", "Numbering Layer 2", 1000, 700);
  c->Divide(1, 2);
  c->cd(1);
  Draw(hL2Up, phiOffsetL2, TMath::Pi() - phiOffsetL2, 5);
  c->cd(2);
  Draw(hL2Bot, TMath::Pi() + phiOffsetL2, 2 * TMath::Pi() - phiOffsetL2, 5);
  c->SaveAs("its3_tile_layer2.pdf");
}
