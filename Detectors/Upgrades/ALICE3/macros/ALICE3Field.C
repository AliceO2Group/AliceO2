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
//
// Author: J. E. Munoz Mendez jesus.munoz@cern.ch

#include <functional>
#include <cmath>
#include <TCanvas.h>
#include <TH2F.h>
#include <TStyle.h>

std::function<void(const double*, double*)> field()
{
  return [](const double* x, double* b) {
    // RADIUS
    static constexpr double Rc = 170.; // [cm] — R_out_coil per Ian DetectorConstruction.cc; confirmed by A. Ortiz definition
    static constexpr double R1 = 220.; // [cm]
    static constexpr double R2 = 290.; // [cm]

    // FIELD
    static constexpr double B1 = 2.;                                  // [T]
    static constexpr double B2 = -B1 * Rc * Rc / (R2 * R2 - R1 * R1); // [T] — B1 in numerator, confirmed by A. Ortiz Aug 2026
    static constexpr double beamStart = 500.;                         // [cm]
    static constexpr double tokGauss = 1. / 0.1;                      // conversion from Tesla to kGauss

    static constexpr bool isMagAbs = true;

    // To set the B2

    if ((std::abs(x[2]) <= beamStart) && (sqrt(x[0] * x[0] + x[1] * x[1]) < Rc)) {
      b[0] = 0.;
      b[1] = 0.;
      b[2] = B1 * tokGauss;
    } else if ((std::abs(x[2]) <= beamStart) &&
               (sqrt(x[0] * x[0] + x[1] * x[1]) >= Rc &&
                sqrt(x[0] * x[0] + x[1] * x[1]) < R1)) {
      b[0] = 0.;
      b[1] = 0.;
      b[2] = 0.;
    } else if ((std::abs(x[2]) <= beamStart) &&
               (sqrt(x[0] * x[0] + x[1] * x[1]) >= R1 &&
                sqrt(x[0] * x[0] + x[1] * x[1]) < R2)) {
      b[0] = 0.;
      b[1] = 0.;
      if (isMagAbs) {
        b[2] = B2 * tokGauss;
      } else {
        b[2] = 0.;
      }
    } else {
      b[0] = 0.;
      b[1] = 0.;
      b[2] = 0.;
    }
  };
}

void ALICE3Field()
{
  gStyle->SetPalette(kRainBow);
  gStyle->SetNumberContours(255);

  auto fieldFunc = field();
  // RZ plane visualization
  TCanvas* cRZ = new TCanvas("cRZ", "Field in RZ plane", 800, 800);
  gPad->SetRightMargin(0.15);
  TH2F* hRZ = new TH2F("hRZ", "Magnetic Field B_z in RZ plane;Z [m];R [m];B_{z} [kGauss]", 100, -10, 10, 100, -5, 5);
  hRZ->SetBit(TH1::kNoStats); // disable stats box
  for (int i = 1; i <= hRZ->GetNbinsX(); i++) {
    const double Z = hRZ->GetXaxis()->GetBinCenter(i);
    for (int j = 1; j <= hRZ->GetNbinsY(); j++) {
      const double R = hRZ->GetYaxis()->GetBinCenter(j);
      const double pos[3] = {R * 100, 0, Z * 100}; // convert to cm
      double b[3] = {0, 0, 0};
      fieldFunc(pos, b);
      hRZ->SetBinContent(i, j, b[2]);
    }
  }

  hRZ->GetZaxis()->SetRangeUser(-30, 30);
  hRZ->Draw("COLZ");
  cRZ->Update();

  // XY plane visualization
  TCanvas* cXY = new TCanvas("cXY", "Field in XY plane", 800, 800);
  gPad->SetRightMargin(0.15);
  TH2F* hXY = new TH2F("hXY", "Magnetic Field B_z in XY plane;X [m];Y [m];B_{z} [kGauss]", 100, -5, 5, 100, -5, 5);
  hXY->SetBit(TH1::kNoStats); // disable stats box

  for (int i = 1; i <= hXY->GetNbinsX(); i++) {
    const double X = hXY->GetXaxis()->GetBinCenter(i);
    for (int j = 1; j <= hXY->GetNbinsY(); j++) {
      const double Y = hXY->GetYaxis()->GetBinCenter(j);
      const double pos[3] = {X * 100, Y * 100, 0}; // convert to cm
      double b[3] = {0, 0, 0};
      fieldFunc(pos, b);
      hXY->SetBinContent(i, j, b[2]);
    }
  }

  hXY->GetZaxis()->SetRangeUser(-30, 30);
  hXY->Draw("COLZ");
  cXY->Update();
}
