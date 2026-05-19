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

#include <TCanvas.h>
#include <TGraph.h>
#include <TArc.h>
#include <TH2F.h>
#include <TMath.h>
#include <TLatex.h>
#include <TStyle.h>
#include <algorithm>
#include <cmath>

void defineIOTOFGeo(const double rAvg = 21,              // cm, average radius of the layer (used for stave size calculations)
                    const int nStaves = 24,              // Number of staves
                    const double staveWidth = 5.42,      // cm, Stave width (arc length at avg radius at 0 degrees)
                    const double staveHeightX2X0 = 0.02, // Stave height (radial at 0 degrees)
                    const double staveTilt = 10          // Stave tilt angle in degrees
)
{
  const double Si_X0 = 9.5f; // cm, radiation length of silicon
  const double staveHeight = staveHeightX2X0 * Si_X0;

  // 1. Define inner and outer radii for the disk.
  //    The radius corresponds to the distance of the center of the stave to the origin
  const double rInner = rAvg - staveHeight / 2.0;
  const double rOuter = rAvg + staveHeight / 2.0;

  const double alpha = staveTilt * TMath::DegToRad(); // Tilt angle in radians
  const double H = staveHeight;
  const double W = staveWidth;

  // 2. Analytical calculation of Inscribed and Outscribed Radii
  //    We project the global origin (0,0) into the local, unrotated coordinate
  //    system of a single stave centered at (0,0).
  const double u0 = -rAvg * TMath::Cos(alpha);
  const double v0 = rAvg * TMath::Sin(alpha);

  // Inscribed Radius: Distance to the closest point on the stave rectangle
  const double uc = std::max(-H / 2.0, std::min(H / 2.0, u0));
  const double vc = std::max(-W / 2.0, std::min(W / 2.0, v0));
  const double rInscribed = TMath::Sqrt((uc - u0) * (uc - u0) + (vc - v0) * (vc - v0));

  // Outscribed Radius: Maximum distance to one of the 4 corners
  double rOutscribed = 0;
  const double uCorners[4] = {-H / 2.0, H / 2.0, H / 2.0, -H / 2.0};
  const double vCorners[4] = {-W / 2.0, -W / 2.0, W / 2.0, W / 2.0};
  for (int i = 0; i < 4; ++i) {
    const double dist = std::hypot(uCorners[i] - u0, vCorners[i] - v0);
    if (dist > rOutscribed) {
      rOutscribed = dist;
    }
  }

  // 3. Visualization
  new TCanvas("DiskWithStaves", "Disk with Staves", 800, 800);
  gPad->SetGrid();
  gPad->SetLeftMargin(0.15);
  gPad->SetBottomMargin(0.15);
  gPad->SetRightMargin(0.05);
  gPad->SetTopMargin(0.05);

  const double maxR = std::max(rOuter, rOutscribed) * 1.5;
  gPad->DrawFrame(-maxR, -maxR, maxR, maxR, ";X (cm);Y (cm)");

  // Draw Inner and Outer Disk Radii (Reference)
  TArc* arcInner = new TArc(0, 0, rInner);
  arcInner->SetLineStyle(2);
  arcInner->SetLineColor(kGray + 1);
  arcInner->SetFillStyle(0);
  arcInner->Draw("same");

  TArc* arcOuter = new TArc(0, 0, rOuter);
  arcOuter->SetLineStyle(2);
  arcOuter->SetLineColor(kGray + 1);
  arcOuter->SetFillStyle(0);
  arcOuter->Draw("same");

  // Draw Inscribed and Outscribed circles
  TArc* arcInscribed = new TArc(0, 0, rInscribed);
  arcInscribed->SetLineColor(kBlue);
  arcInscribed->SetLineWidth(2);
  arcInscribed->SetFillStyle(0);
  arcInscribed->Draw("same");

  TArc* arcOutscribed = new TArc(0, 0, rOutscribed);
  arcOutscribed->SetLineColor(kRed);
  arcOutscribed->SetLineWidth(2);
  arcOutscribed->SetFillStyle(0);
  arcOutscribed->Draw("same");

  // Generate and Draw Staves
  for (int i = 0; i < nStaves; ++i) {
    double phi = i * TMath::TwoPi() / nStaves;
    double xPts[5], yPts[5];
    for (int j = 0; j < 4; ++j) {
      double u = uCorners[j];
      double v = vCorners[j];
      // Apply stave tilt (alpha) around its own center
      double uRot = u * TMath::Cos(alpha) - v * TMath::Sin(alpha);
      double vRot = u * TMath::Sin(alpha) + v * TMath::Cos(alpha);
      // Move stave to rAvg and apply azimuthal rotation (phi)
      double x_phi0 = rAvg + uRot;
      double y_phi0 = vRot;
      xPts[j] = x_phi0 * TMath::Cos(phi) - y_phi0 * TMath::Sin(phi);
      yPts[j] = x_phi0 * TMath::Sin(phi) + y_phi0 * TMath::Cos(phi);
    }
    // Close the geometric polygon
    xPts[4] = xPts[0];
    yPts[4] = yPts[0];
    TGraph* gStave = new TGraph(5, xPts, yPts);
    gStave->SetFillColorAlpha(kGreen + 2, 0.4);
    gStave->SetLineColor(kBlack);
    gStave->SetLineWidth(1);
    gStave->Draw("f same"); // Fill
    gStave->Draw("l same"); // Outline
  }

  // 7. Add Legend / Parameter Text
  TLatex* tex = new TLatex();
  tex->SetNDC();
  tex->SetTextSize(0.028);
  tex->SetTextFont(42);
  tex->SetTextColor(kBlack);
  tex->DrawLatex(0.12, 0.88, Form("R_{inner} = %.1f, R_{outer} = %.1f", rInner, rOuter));
  tex->DrawLatex(0.12, 0.84, Form("Staves: %d, Tilt: %.1f#circ", nStaves, staveTilt));
  tex->SetTextColor(kBlue);
  tex->DrawLatex(0.12, 0.80, Form("Inscribed Radius = %.2f", rInscribed));
  tex->SetTextColor(kRed);
  tex->DrawLatex(0.12, 0.76, Form("Outscribed Radius = %.2f", rOutscribed));
}
