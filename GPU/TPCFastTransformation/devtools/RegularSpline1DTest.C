// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  RegularSpline1DTest.C
/// \brief A macro fo testing the IrregularSpline1D class
///
/// \author  Felix Lapp
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

/*  Load the macro:
  root -l RegularSpline1DTest.C++
*/

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "GPU/RegularSpline1D.h"
#include "TFile.h"
#include "TRandom.h"
#include "TNtuple.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TH1.h"
#include "TCanvas.h"

#endif

float F(float u)
{
  // A test function
  const int PolynomDegree = 7;
  static double coeff[PolynomDegree + 1];
  static bool firstCall = 1;
  if (firstCall) {
    gRandom->SetSeed(0);
    for (int i = 0; i <= PolynomDegree; i++) {
      coeff[i] = gRandom->Uniform(-1, 1);
    }
    firstCall = 0;
  }

  u -= 0.5;
  double uu = 1.;
  double f = 0;
  for (int i = 0; i <= PolynomDegree; i++) {
    f += coeff[i] * uu;
    uu *= u;
  }
  return f;
}

using namespace o2::gpu;
typedef double myfloat;

int RegularSpline1DTest()
{

  cout << "Test roundf(): " << endl;
  for (float x = 0.; x <= 1.; x += 0.1) {
    cout << "roundf(" << x << ") = " << roundf(x) << " " << (int)roundf(x) << endl;
  }

  cout << "Test 5 knots for n bins:" << endl;
  for (int n = 4; n < 20; n++) {
    int bin1 = (int)roundf(.25 * n);
    int bin2 = (int)roundf(.50 * n);
    int bin3 = (int)roundf(.75 * n);
    cout << n << ": 0 " << bin1 << " " << bin2 << " " << bin3 << " " << n << endl;
  }

  cout << "Test interpolation.." << endl;

  const int initNKnotsU = 10;
  int nAxisBinsU = 10;

  float du = 1. / (initNKnotsU - 1);

  float knotsU[initNKnotsU];

  knotsU[0] = 0;
  knotsU[initNKnotsU - 1] = 1;

  for (int i = 1; i < initNKnotsU - 1; i++) {
    knotsU[i] = i * du + gRandom->Uniform(-du / 3, du / 3);
  }

  RegularSpline1D spline;

  spline.construct(initNKnotsU);

  int nKnotsTot = spline.getNumberOfKnots();
  cout << "Knots: initial " << initNKnotsU << ", created " << nKnotsTot << endl;
  for (int i = 0; i < nKnotsTot; i++) {
    cout << "knot " << i << ": " << spline.knotIndexToU(i) << endl;
  }

  myfloat* data0 = new myfloat[nKnotsTot]; // original data
  myfloat* data = new myfloat[nKnotsTot];  // corrected data

  int nu = spline.getNumberOfKnots();

  for (int i = 0; i < spline.getNumberOfKnots(); i++) {
    data0[i] = F(spline.knotIndexToU(i));
    //SG random
    data0[i] = gRandom->Uniform(-1., 1.);
    data[i] = data0[i];
  }

  for (int i = 1; i < nKnotsTot; i++) {
    cout << "data[" << i << "]: " << data[i] << endl;
  }

  spline.correctEdges(data);

  TH1F* qaX = new TH1F("qaX", "qaX [um]", 1000, -1000., 1000.);

  int iter = 0;
  float stepu = 1.e-4;
  int nSteps = (int)(1. / stepu + 1);

  TNtuple* knots = new TNtuple("knots", "knots", "u:f");
  double diff = 0;
  for (int i = 0; i < nu; i++) {
    double u = spline.knotIndexToU(i);
    double f0 = data0[i];
    knots->Fill(u, f0);
    double fs = spline.getSpline(data, u);
    diff += (fs - f0) * (fs - f0);
  }
  cout << "mean diff at knots: " << sqrt(diff) / nu << endl;

  knots->SetMarkerSize(1.);
  knots->SetMarkerStyle(8);
  knots->SetMarkerColor(kRed);

  TNtuple* n = new TNtuple("n", "n", "u:f0:fs");

  for (float u = 0; u <= 1.; u += stepu) {
    iter++;
    float f = spline.getSpline(data, u);
    qaX->Fill(1.e4 * (f - F(u)));
    n->Fill(u, F(u), f);
  }

  TCanvas* canv = new TCanvas("cQA", "IrregularSpline1D  QA", 2000, 1000);
  n->SetMarkerColor(kBlue);
  n->Draw("fs:u", "", "");
  knots->Draw("f:u", "", "same");

  canv->Update();

  return 0;
}
