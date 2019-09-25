// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  IrregularSpline2D3DTest.C
/// \brief A macro fo testing the IrregularSpline2D3D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

/*  Load the macro:
  root -l IrregularSpline2D3DTest.C++
*/

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "GPU/IrregularSpline2D3D.h"
#include "TFile.h"
#include "TRandom.h"
#include "TNtuple.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TGraph2D.h"
#include "TStyle.h"

#endif

//#define GPUCA_NO_VC //Test without Vc, otherwise ctest fails linking

float Fx(float u, float v)
{
  const int PolynomDegree = 7;
  static double cu[PolynomDegree + 1], cv[PolynomDegree + 1];
  static bool isInitialized = 0;

  if (!isInitialized) {
    for (int i = 0; i <= PolynomDegree; i++) {
      cu[i] = i * gRandom->Uniform(-1, 1);
      cv[i] = i * gRandom->Uniform(-1, 1);
    }
    isInitialized = 1;
  }
  u -= 0.5;
  v -= 0.5;
  double uu = 1.;
  double vv = 1.;
  double f = 0;
  for (int i = 0; i <= PolynomDegree; i++) {
    f += cu[i] * uu;
    f += cv[i] * vv;
    uu *= u;
    vv *= v;
  }
  return f;
}

float Fy(float u, float v) { return v; }
float Fz(float u, float v) { return (u - .5) * (u - .5); }

int IrregularSpline2D3DTest()
{
  using namespace o2::gpu;

  gRandom->SetSeed(0);
  UInt_t seed = gRandom->Integer(100000); // 605
  gRandom->SetSeed(seed);
  cout << "Random seed: " << seed << " " << gRandom->GetSeed() << endl;

  IrregularSpline2D3D spline;

  {
    const int nKnotsU = 7, nKnotsV = 7;
    float knotsU[nKnotsU], knotsV[nKnotsV];

    int nAxisTicksU = 20;
    int nAxisTicksV = 20;

    double du = 1. / (nKnotsU - 1);
    double dv = 1. / (nKnotsV - 1);
    for (int i = 1; i < nKnotsU - 1; i++) {
      knotsU[i] = i * du + gRandom->Uniform(-du / 3., du / 3.);
    }
    for (int i = 1; i < nKnotsV - 1; i++) {
      knotsV[i] = i * dv + gRandom->Uniform(-dv / 3., dv / 3.);
    }

    knotsU[0] = 0.;
    knotsU[nKnotsU - 1] = 1.;
    knotsV[0] = 0.;
    knotsV[nKnotsV - 1] = 1.;

    std::sort(knotsU, knotsU + nKnotsU);
    std::sort(knotsV, knotsV + nKnotsV);

    spline.construct(nKnotsU, knotsU, nAxisTicksU, nKnotsV, knotsV, nAxisTicksV);
  }

  int nKnotsTot = spline.getNumberOfKnots();

  const IrregularSpline1D& gridU = spline.getGridU();
  const IrregularSpline1D& gridV = spline.getGridV();

  float* data0 = new float[3 * nKnotsTot];
  float* data = new float[3 * nKnotsTot];

  int nu = gridU.getNumberOfKnots();

  for (int i = 0; i < gridU.getNumberOfKnots(); i++) {
    double u = gridU.getKnot(i).u;
    for (int j = 0; j < gridV.getNumberOfKnots(); j++) {
      double v = gridV.getKnot(j).u;
      int ind = (nu * j + i) * 3;
      data0[ind + 0] = Fx(u, v);
      data0[ind + 1] = Fy(u, v);
      data0[ind + 2] = Fz(u, v);
      // use random values instaad of Fx
      //data0[ind + 0] = gRandom->Uniform(-.3, .3);
    }
  }

  for (int i = 0; i < 3 * nKnotsTot; i++) {
    data[i] = data0[i];
  }

  spline.correctEdges(data);

  TCanvas* canv = new TCanvas("cQA", "2D splines  QA", 1500, 1500);
  canv->Draw();
  // canv->Divide(3,1);
  // canv->Update();

  TGraph2D* gknots = new TGraph2D();
  gknots->SetName("gknots");
  gknots->SetTitle("gknots");
  gknots->SetLineColor(kRed);
  gknots->SetMarkerSize(1.);
  gknots->SetMarkerStyle(8);
  gknots->SetMarkerColor(kRed);

  int gknotsN = 0;
  TNtuple* knots = new TNtuple("knots", "knots", "u:v:f");
  double diff = 0;
  for (int i = 0; i < gridU.getNumberOfKnots(); i++) {
    for (int j = 0; j < gridV.getNumberOfKnots(); j++) {
      double u = gridU.getKnot(i).u;
      double v = gridV.getKnot(j).u;
      int ind = (nu * j + i) * 3;
      double fx0 = data0[ind + 0];
      knots->Fill(u, v, fx0);
      float x, y, z;
      spline.getSpline(data, u, v, x, y, z);
      diff += (fx0 - x) * (fx0 - x);
      gknots->SetPoint(gknotsN++, u, v, fx0);
    }
  }
  cout << "mean diff at knots: " << sqrt(diff) / gknotsN << endl;

  knots->SetMarkerSize(1.);
  knots->SetMarkerStyle(8);
  knots->SetMarkerColor(kRed);

  TNtuple* nt = new TNtuple("nt", "nt", "u:v:f0:fs");

  TGraph2D* gf0 = new TGraph2D();
  gf0->SetName("gf0");
  gf0->SetTitle("gf0");
  gf0->SetLineColor(kRed);
  int gf0N = 0;

  TGraph2D* gfs = new TGraph2D();
  gfs->SetName("gfs");
  gfs->SetTitle("gfs");
  gfs->SetLineColor(kBlue);
  int gfsN = 0;

  TH1F* qaX = new TH1F("qaX", "qaX [um]", 1000, -100., 100.);
  TH1F* qaY = new TH1F("qaY", "qaY [um]", 1000, -10., 10.);
  TH1F* qaZ = new TH1F("qaZ", "qaZ [um]", 1000, -10., 10.);

  int iter = 0;
  float stepu = 1.e-3;
  float stepv = 1.e-3;

  for (float u = -.05; u <= 1.05; u += stepu) {
    for (float v = -.05; v <= 1.05; v += stepv) {
      iter++;
      float x, y, z;
      spline.getSplineVec(data, u, v, x, y, z);
      if (u >= 0 && v >= 0 && u <= 1 && v <= 1) {
        qaX->Fill(1.e4 * (x - Fx(u, v)));
        qaY->Fill(1.e4 * (y - Fy(u, v)));
        qaZ->Fill(1.e4 * (z - Fz(u, v)));
        nt->Fill(u, v, Fx(u, v), x);
      }
      gf0->SetPoint(gf0N++, u, v, Fx(u, v));
      gfs->SetPoint(gfsN++, u, v, x);
    }
  }

  gStyle->SetPalette(1);

  // gknots->Draw("P");
  gfs->Draw("surf");
  // gf0->Draw("surf,same");
  gknots->Draw("P,same");
  canv->Update();

  delete[] data0;
  delete[] data;

  return 0;
}
