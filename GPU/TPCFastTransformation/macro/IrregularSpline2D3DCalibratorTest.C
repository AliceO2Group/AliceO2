// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  IrregularSpline2D3DCalibratorTest.C
/// \brief A macro fo testing the IrregularSpline2D3DCalibrator class
///
/// \author  Felix Lapp
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

/*  Load the macro:
  root -l SemiregularSpline2D3D.C++
*/

/*

alienv load O2/latest
root -l IrregularSpline2D3DCalibratorTest.C++

*/

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "TFile.h"
#include "TRandom.h"
#include "TNtuple.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TGraph2D.h"
#include "TStyle.h"
#include "algorithm"
#include <chrono>

#include "GPU/IrregularSpline2D3D.h"
#include "GPU/SemiregularSpline2D3D.h"
#include "GPU/IrregularSpline2D3DCalibrator.h"

#endif

using namespace o2::gpu;

IrregularSpline2D3D splineF;
float* splineF_data = 0;

void initF()
{
  const int nKnotsU = 6;
  const int nKnotsV = 6;
  float knotsU[nKnotsU];
  float knotsV[nKnotsV];
  for (int i = 0; i < nKnotsU; i++) {
    knotsU[i] = i / (double)(nKnotsU - 1);
  }
  knotsU[0] = 0.;
  knotsU[nKnotsU - 1] = 1.;
  for (int i = 0; i < nKnotsV; i++) {
    knotsV[i] = i / (double)(nKnotsV - 1);
  }
  knotsV[0] = 0.;
  knotsV[nKnotsV - 1] = 1.;
  splineF.construct(nKnotsU, knotsU, nKnotsU,
                    nKnotsV, knotsV, nKnotsV);

  cout << "number of knots: " << splineF.getNumberOfKnots() << endl;

  splineF_data = new float[splineF.getNumberOfKnots() * 3];
  for (int i = 0; i < splineF.getNumberOfKnots(); i++) {
    splineF_data[3 * i + 0] = gRandom->Uniform(-1., 1.);
    splineF_data[3 * i + 1] = gRandom->Uniform(-1., 1.);
    splineF_data[3 * i + 2] = gRandom->Uniform(-1., 1.);
  }

  for (int i = 0; i < splineF.getNumberOfKnots(); i++) { // set Fy=Fz=Fx
    splineF_data[3 * i + 1] = splineF_data[3 * i + 0];
    splineF_data[3 * i + 2] = splineF_data[3 * i + 0];
  }

  splineF.correctEdges(splineF_data);
}

void F(float u, float v, float& fx, float& fy, float& fz)
{
  splineF.getSplineVec(splineF_data, u, v, fx, fy, fz);
}

bool initTPC(const char* fileName, int slice, int row)
{
  // open NTuple file

  TFile file(fileName, "READ");
  if (!file.IsOpen()) {
    cout << "distortion file not found!" << endl;
    exit(0);
  }
  TNtuple* nt = (TNtuple*)file.Get("dist");
  if (!nt) {
    cout << "distortion ntuple is not found in the file!" << endl;
    exit(0);
  }

  float fslice, frow, su, sv, dx, du, dv;
  nt->SetBranchAddress("slice", &fslice);
  nt->SetBranchAddress("row", &frow);
  nt->SetBranchAddress("su", &su);
  nt->SetBranchAddress("sv", &sv);
  nt->SetBranchAddress("dx", &dx);
  nt->SetBranchAddress("du", &du);
  nt->SetBranchAddress("dv", &dv);

  int nKnots = 101;
  splineF.constructRegular(nKnots, nKnots);
  delete[] splineF_data;
  splineF_data = new float[3 * splineF.getNumberOfKnots()];
  for (int i = 0; i < 3 * splineF.getNumberOfKnots(); i++)
    splineF_data[i] = 0.;

  int nent = 0;
  for (int i = 0; i < nt->GetEntriesFast(); i++) {
    nt->GetEntry(i);
    if (nearbyint(fslice) != slice || nearbyint(frow) != row)
      continue;
    if (nent >= splineF.getNumberOfKnots()) {
      cout << "Init TPC: unexpected entries: entry " << nent << " expected n entries " << splineF.getNumberOfKnots() << endl;
      return 0;
    }
    splineF_data[3 * nent + 0] = dx;
    splineF_data[3 * nent + 1] = du;
    splineF_data[3 * nent + 2] = dv;
    nent++;
  }
  if (nent != splineF.getNumberOfKnots()) {
    cout << "Init TPC: unexpected N entries: read " << nent << " expected " << splineF.getNumberOfKnots() << endl;
    return 0;
  }
  return 1;
}

int IrregularSpline2D3DCalibratorTest()
{

  const bool kDraw = 1;
  const bool kAsk = 0;
  const bool kTestTPC = 0;

  cout << "Spline calibration starts" << endl;

  IrregularSpline2D3DCalibrator finder;

  TCanvas* canv = nullptr;
  if (kDraw) {
    canv = new TCanvas("cQA", "2D spline calibration", 1800, 1000);
    canv->Divide(2, 2);
  }

  gStyle->SetMarkerSize(1.);
  gStyle->SetMarkerStyle(8);
  gStyle->SetMarkerColor(kRed);

  TGraph2D* gknotsF = nullptr;
  TGraph2D* gknots = nullptr;
  TGraph2D* gknotsDiff = nullptr;
  TGraph2D* gf0 = nullptr;
  TGraph2D* gfs = nullptr;
  TGraph2D* gfdiff = nullptr;

  TNtuple* ntDiff = new TNtuple("diff", "diff", "u:v:dfx");
  TH1F* qaX = new TH1F("qaX", "diff F - spline", 1000, -0.05, 0.05);

  char keyPressed = '\0';
  for (int sample = 1; (keyPressed != 'q') && (sample < 2); sample++) {

    int seed = sample;
    /*
      gRandom->SetSeed(0);
      seed = gRandom->Integer(100);      
    */
    gRandom->SetSeed(seed);
    if (kTestTPC) {
      bool ok = initTPC("tpcDistortion.root", 0, 0);
      if (!ok)
        break;

      finder.setRasterSize(41, 41);
      finder.setMaxNKnots(21, 21);
      finder.setMaximalDeviation(0.01);
      finder.startCalibration(F);
    } else {
      initF();
      finder.setRasterSize(61, 61);
      finder.setMaxNKnots(11, 11);
      finder.setMaximalDeviation(0.2);
      finder.startCalibration(F);
    }

    do {

      cout << "n knots: u " << finder.getSpline().getGridU().getNumberOfKnots()
           << " v " << finder.getSpline().getGridV().getNumberOfKnots() << endl;

      delete gknotsF;
      delete gknots;
      delete gknotsDiff;
      TGraph2D* gknotsF = new TGraph2D();
      gknots = new TGraph2D();
      gknotsDiff = new TGraph2D();

      delete gf0;
      delete gfs;
      delete gfdiff;
      gf0 = new TGraph2D();
      gf0->SetTitle("Input function F");
      gf0->SetLineColor(kBlue);

      gfs = new TGraph2D();
      gfs->SetTitle("Spline");
      gfs->SetLineColor(kGreen);

      gfdiff = new TGraph2D();
      gfdiff->SetName("gfdiff");
      gfdiff->SetTitle("diff between F and Spline");
      gfdiff->SetLineColor(kBlue);

      ntDiff->Reset();
      qaX->Reset();
      {
        const IrregularSpline1D& gridU = splineF.getGridU();
        const IrregularSpline1D& gridV = splineF.getGridV();
        int nKnots = 0;
        for (int i = 0; i < gridU.getNumberOfKnots(); i++) {
          double u = gridU.getKnot(i).u;
          for (int j = 0; j < gridV.getNumberOfKnots(); j++) {
            double v = gridV.getKnot(j).u;
            float fx, fy, fz;
            F(u, v, fx, fy, fz);
            gknotsF->SetPoint(nKnots++, u, v, fx);
          }
        }
      }

      const IrregularSpline1D& gridU = finder.getSpline().getGridU();
      const IrregularSpline1D& gridV = finder.getSpline().getGridV();

      int nKnots = 0;
      for (int i = 0; i < gridU.getNumberOfKnots(); i++) {
        double u = gridU.getKnot(i).u;
        for (int j = 0; j < gridV.getNumberOfKnots(); j++) {
          double v = gridV.getKnot(j).u;
          float fx, fy, fz;
          F(u, v, fx, fy, fz);
          float fx1, fy1, fz1;
          finder.getSpline().getSplineVec(finder.getSplineData(), u, v, fx1, fy1, fz1);
          gknots->SetPoint(nKnots, u, v, fx1);
          gknotsDiff->SetPoint(nKnots++, u, v, fx1 - fx);
        }
      }

      float stepu = 1.e-2;
      float stepv = 1.e-2;
      int nPoints = 0;
      for (float u = 0; u <= 1; u += stepu) {
        for (float v = 0; v <= 1; v += stepv) {
          float fx0, fy0, fz0;
          F(u, v, fx0, fy0, fz0);
          //finder.getRaster().getSplineVec(finder.getRasterData(), u, v, fx0, fy0, fz0);
          float fx1, fy1, fz1;
          finder.getSpline().getSplineVec(finder.getSplineData(), u, v, fx1, fy1, fz1);
          if (u >= 0 && v >= 0 && u <= 1 && v <= 1) {
            qaX->Fill((fx1 - fx0));
            qaX->Fill((fy1 - fy0));
            qaX->Fill((fz1 - fz0));
          }
          gf0->SetPoint(nPoints, u, v, fx0);
          gfs->SetPoint(nPoints, u, v, fx1);
          gfdiff->SetPoint(nPoints++, u, v, fx1 - fx0);
          ntDiff->Fill(u, v, fx1 - fx0);
        }
      }

      if (kDraw) {
        canv->cd(1);
        gStyle->SetPalette(1);
        gf0->Draw("surf");
        gknotsF->Draw("P,same");

        canv->cd(2);
        gStyle->SetPalette(1);
        gfs->Draw("surf");
        gknots->Draw("P,same");

        canv->cd(3);

        gStyle->SetPalette(1);
        //ntDiff->Draw("dfx:v:u","","surf");
        gfdiff->Draw("surf");
        gknotsDiff->Draw("P,same");

        canv->cd(4);
        qaX->Draw();
      }
      if (kDraw && kAsk) {
        keyPressed = getchar();
        if (keyPressed == 'd')
          canv->Update();
        if (keyPressed == 'q')
          break;
      }

      bool isChanged = finder.doCalibrationStep();
      if (!isChanged)
        break;
    } while (keyPressed != 'q');

    if (keyPressed == 'q')
      break;

    if ((keyPressed != 'q') && kDraw)
      canv->Update();

    std::cout << " input spline: " << std::endl;
    splineF.getGrid(0).print();
    splineF.getGrid(1).print();
    std::cout << " output spline: " << std::endl;
    finder.getSpline().getGrid(0).print();
    finder.getSpline().getGrid(1).print();
    cout << "seed = " << seed << std::endl;

    if (kAsk)
      keyPressed = getchar();
  }

  return 0;
}
