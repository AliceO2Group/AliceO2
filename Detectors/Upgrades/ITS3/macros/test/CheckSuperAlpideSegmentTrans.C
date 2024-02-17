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

/// \file CheckSuperAlpideSegmentTrans.C
/// \brief Simple macro to check ITS3 Alpide Trans

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "TArc.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TNtuple.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TTree.h"

#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Base/SpecsV2.h"

#endif

using namespace o2::its3;

constexpr float PI = 3.14159274101257324e+00f;
constexpr float Rad2Deg = 180.f / PI;
constexpr float Deg2Rad = 1. / Rad2Deg;

constexpr auto nRows{SegmentationSuperAlpide::mNRows};
constexpr auto nCols{SegmentationSuperAlpide::mNCols};
constexpr auto fLength{SegmentationSuperAlpide::mLength};
constexpr auto fWidth{SegmentationSuperAlpide::mWidth};

TH2* DrawReverseBins(TH2* h)
{
  TH2F* h2 = new TH2F(Form("%s_invert", h->GetName()), h->GetTitle(), nCols, 0,
                      nCols, nRows, 0, nRows);

  h2->GetXaxis()->SetLabelOffset(999);
  h2->GetXaxis()->SetTickLength(0);
  h2->GetYaxis()->SetLabelOffset(999);
  h2->GetYaxis()->SetTickLength(0);

  for (int i = 1; i <= h->GetNbinsX(); i++) {
    for (int j = 1; j <= h->GetNbinsY(); j++) {
      h2->SetBinContent(i, h->GetNbinsY() - j + 1, h->GetBinContent(i, j));
    }
  }

  return h2;
}

void ReverseYAxis(TH1* h)
{
  gPad->Update();
  TGaxis* newaxis =
    new TGaxis(gPad->GetUxmin(), gPad->GetUymax(), gPad->GetUxmin() - 0.001,
               gPad->GetUymin(), h->GetYaxis()->GetXmin(),
               h->GetYaxis()->GetXmax(), 510, "+");
  newaxis->SetLabelOffset(-0.03);
  newaxis->Draw();
}

void DrawXAxisCol(TH1* h)
{
  gPad->Update();
  TGaxis* newaxis =
    new TGaxis(gPad->GetUxmin(), gPad->GetUymax(), gPad->GetUxmin() - 0.001,
               gPad->GetUymin(), h->GetXaxis()->GetXmin(),
               h->GetYaxis()->GetXmax(), 510, "+");
  newaxis->SetLabelOffset(-0.03);
  newaxis->Draw();
}

void CheckSuperAlpideSegmentTrans()
{
  gStyle->SetOptStat(1111111);

  for (int iLayer{0}; iLayer < 3; ++iLayer) {
    float r_inner = constants::radii[iLayer] - constants::thickness / 2.;
    float r_outer = constants::radii[iLayer] + constants::thickness / 2.;
    float phiReadout_inner =
      constants::tile::readout::width / r_inner * Rad2Deg;
    float phiReadout_outer =
      constants::tile::readout::width / r_outer * Rad2Deg;
    float pixelarray_inner =
      constants::pixelarray::width / r_inner * Rad2Deg + phiReadout_inner;
    float pixelarray_outer =
      constants::pixelarray::width / r_outer * Rad2Deg + phiReadout_outer;
    auto arc_inner =
      new TArc(0, 0, r_inner, phiReadout_inner, pixelarray_inner);
    arc_inner->SetFillStyle(0);
    arc_inner->SetLineColor(kBlue);
    auto arc_outer =
      new TArc(0, 0, r_outer, phiReadout_outer, pixelarray_outer);
    arc_outer->SetFillStyle(0);
    arc_outer->SetLineColor(kRed);
    // Generate points on arc
    auto* h_c2f_base =
      new TH2F(Form("h_c2f_base_%d", iLayer), "Curved 2 Flat", 100,
               -r_outer - 0.1, r_outer + 0.1, 100, -0.2, 1.2);
    auto* h_f2c_res =
      new TH2F(Form("h_f2c_res_%d", iLayer), "XY Residuals;x [cm]; y [cm]",
               101, -1e-3, 1e-3, 101, -2e-3, 2e-3);
    // float stepSize = pixelarray_inner - phiReadout_inner;
    float stepSize = 1e-3;
    auto* g_arc_inner = new TGraph();
    g_arc_inner->SetMarkerStyle(5);
    g_arc_inner->SetMarkerColor(kBlue + 1);
    auto* g_arc_inner_flat = new TGraph();
    g_arc_inner_flat->SetMarkerStyle(5);
    g_arc_inner_flat->SetMarkerColor(kBlue + 1);
    auto* g_arc_outer = new TGraph();
    g_arc_outer->SetMarkerStyle(5);
    g_arc_outer->SetMarkerColor(kRed + 1);
    auto* g_arc_outer_flat = new TGraph();
    g_arc_outer_flat->SetMarkerStyle(5);
    g_arc_outer_flat->SetMarkerColor(kRed + 1);
    float xmin_inner = {0}, xmax_inner = {0};
    float xmin_outer = {0}, xmax_outer = {0};
    for (float phi{phiReadout_inner}; phi <= pixelarray_inner;
         phi += stepSize) {
      float x_inner = r_inner * std::cos(phi * Deg2Rad),
            y_inner = r_inner * std::sin(phi * Deg2Rad), x_inner_flat,
            y_inner_flat, x_inner_curved, y_inner_curved;
      float x_outer = r_outer * std::cos(phi * Deg2Rad),
            y_outer = r_outer * std::sin(phi * Deg2Rad), x_outer_flat,
            y_outer_flat, x_outer_curved, y_outer_curved;
      g_arc_inner->AddPoint(x_inner, y_inner);
      g_arc_outer->AddPoint(x_outer, y_outer);
      // Test Segmentation
      SuperSegmentations[iLayer].curvedToFlat(x_inner, y_inner, x_inner_flat, y_inner_flat);
      SuperSegmentations[iLayer].flatToCurved(x_inner_flat, y_inner_flat, x_inner_curved, y_inner_curved);
      SuperSegmentations[iLayer].curvedToFlat(x_outer, y_outer, x_outer_flat, y_outer_flat);
      SuperSegmentations[iLayer].flatToCurved(x_outer_flat, y_outer_flat, x_outer_curved, y_outer_curved);
      g_arc_inner_flat->AddPoint(x_inner_flat, y_inner_flat);
      g_arc_outer_flat->AddPoint(x_outer_flat, y_outer_flat);
      h_f2c_res->Fill(x_inner - x_inner_curved, y_inner - y_inner_curved);
      h_f2c_res->Fill(x_outer - x_outer_curved, y_outer - y_outer_curved);
      // Info("C2F", "Outer: ( %f / %f ) ---> ( %f / %f ) ---> ( %f / %f )",
      //      x_inner, y_inner, x_inner_flat, y_inner_flat, x_inner_curved,
      //      y_inner_curved);
      // Info("C2F", "Inner: ( %f / %f ) ---> ( %f / %f ) ---> ( %f / %f )",
      //      x_outer, y_outer, x_outer_flat, y_outer_flat, x_outer_curved,
      //      y_outer_curved);
      if (x_inner_flat < xmin_inner) {
        xmin_inner = x_inner_flat;
      } else if (x_inner_flat > xmax_inner) {
        xmax_inner = x_inner_flat;
      }
      if (x_outer_flat < xmin_outer) {
        xmin_outer = x_outer_flat;
      } else if (x_outer_flat > xmax_outer) {
        xmax_outer = x_outer_flat;
      }
    }
    float width_inner = xmax_inner - xmin_inner,
          width_outer = xmax_outer - xmin_outer;
    Info("C2F", "Inner: Xmin=%f Xmax=%f ---> %f", xmin_inner, xmax_inner,
         width_inner);
    Info("C2F", "Outer: Xmin=%f Xmax=%f ---> %f", xmin_outer, xmax_outer,
         width_outer);
    if (float dev =
          abs(width_inner - constants::pixelarray::width) / width_inner;
        dev > 0.001) {
      Error("C2F", "Inner: Not equal length projection! Real=%f (Dev=%f%%",
            constants::pixelarray::width, dev);
    }
    if (float dev =
          abs(width_outer - constants::pixelarray::width) / width_outer;
        dev > 0.001) {
      Error("C2F", "Outer: Not equal length projection! Real=%f (Dev=%f%%",
            constants::pixelarray::width, dev);
    }

    // L2D, D2L transformations
    auto* h_local = new TH2F(Form("h_local_%d", iLayer), "Local Coordinates",
                             nCols, 0, nCols, nRows, 0, nRows);
    auto* h_detector = new TH2F(
      Form("h_detector_%d", iLayer), "Detector Coordinates", nCols * 4,
      -fLength / 2., fLength / 2., nRows * 4, -fWidth / 2., fWidth / 2.);
    auto* h_l2d_row =
      new TH1F(Form("h_l2d_row_%d", iLayer),
               "Residual: row_{gen} - row_{ii}; #DeltaRow", 1000, -10, 10);
    auto* h_l2d_col =
      new TH1F(Form("h_l2d_col_%d", iLayer),
               "Residual: col_{gen} - col_{ii}; #DeltaCol", 1000, -10, 10);

    for (int iRow{0}; iRow < nRows; ++iRow) {
      for (int iCol{0}; iCol < nCols; ++iCol) {
        float xRow{0}, zCol{0};
        int iiRow{0}, iiCol{0};
        auto v1 =
          SuperSegmentations[iLayer].detectorToLocal(iRow, iCol, xRow, zCol);
        auto v2 = SuperSegmentations[iLayer].localToDetector(xRow, zCol, iiRow,
                                                             iiCol);
        // Info("L2D",
        //      "iRow=%d, iCol=%d --d2l(%s)--> xRow=%f, zCol=%f --l2d(%s)--> "
        //      "iiRow=%d, iiCol=%d",
        //      iRow, iCol, v1 ? "good" : "bad", xRow, zCol, v2 ? "good" :
        //      "bad", iiRow, iiCol);
        if (!v1 || !v2) {
          Error("LOOP", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Layer %d", iLayer);
          return;
        }
        h_local->Fill(iCol, iRow);
        h_detector->Fill(zCol, xRow);
        h_l2d_row->Fill(iRow - iiRow);
        h_l2d_col->Fill(iCol - iiCol);
      }
    }

    // Plots
    auto* c = new TCanvas();
    c->SetTitle(Form("Layer %d", iLayer));
    c->Divide(2, 3);
    c->cd(1);
    h_c2f_base->Draw();
    arc_inner->Draw("same;only");
    g_arc_inner->Draw("same;p");
    g_arc_inner_flat->Draw("same;p");
    arc_outer->Draw("same;only");
    g_arc_outer->Draw("same;p");
    g_arc_outer_flat->Draw("same;p");
    c->cd(2);
    h_f2c_res->Draw("colz");
    c->cd(3);
    auto* h_local_invert = DrawReverseBins(h_local);
    h_local_invert->Draw();
    ReverseYAxis(h_local_invert);
    DrawXAxisCol(h_local_invert);
    c->cd(4);
    h_detector->Draw();
    c->cd(5);
    h_l2d_row->Draw();
    c->cd(6);
    h_l2d_col->Draw();
    c->SaveAs(Form("test_%d.pdf", iLayer));
  }
}
