/*

   // works only with ROOT >= 6

   alienv load ROOT/latest-root6
   alienv load Vc/latest

   root -l

   .x loadlibs.C
   .x IrregularSpline1DTest.C++
 */

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "TFile.h"
#include "TRandom.h"
#include "TNtuple.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TH1.h"
#include "TCanvas.h"
#include "IrregularSpline1D.h"

const int32_t PolynomDegree = 7;
double coeff[PolynomDegree + 1];

float F(float u)
{
  u -= 0.5;
  double uu = 1.;
  double f = 0;
  for (int32_t i = 0; i <= PolynomDegree; i++) {
    f += coeff[i] * uu;
    uu *= u;
  }
  return f;
}

typedef double myfloat;

int32_t IrregularSpline1DTest()
{

  using namespace GPUCA_NAMESPACE::gpu;

  std::cout << "Test roundf(): " << std::endl;
  for (float x = 0.; x <= 1.; x += 0.1) {
    std::cout << "roundf(" << x << ") = " << roundf(x) << " " << (int32_t)roundf(x) << std::endl;
  }

  std::cout << "Test 5 knots for n bins:" << std::endl;
  for (int32_t n = 4; n < 20; n++) {
    int32_t bin1 = (int32_t)roundf(.25 * n);
    int32_t bin2 = (int32_t)roundf(.50 * n);
    int32_t bin3 = (int32_t)roundf(.75 * n);
    std::cout << n << ": 0 " << bin1 << " " << bin2 << " " << bin3 << " " << n << std::endl;
  }

  std::cout << "Test interpolation.." << std::endl;

  gRandom->SetSeed(0);
  uint32_t seed = gRandom->Integer(100000); // 605

  gRandom->SetSeed(seed);
  std::cout << "Random seed: " << seed << " " << gRandom->GetSeed() << std::endl;
  for (int32_t i = 0; i <= PolynomDegree; i++) {
    coeff[i] = gRandom->Uniform(-1, 1);
  }

  const int32_t initNKnotsU = 10;
  int32_t nAxisBinsU = 10;

  float du = 1. / (initNKnotsU - 1);

  float knotsU[initNKnotsU];

  knotsU[0] = 0;
  knotsU[initNKnotsU - 1] = 1;

  for (int32_t i = 1; i < initNKnotsU - 1; i++) {
    knotsU[i] = i * du + gRandom->Uniform(-du / 3, du / 3);
  }

  IrregularSpline1D spline;

  spline.construct(initNKnotsU, knotsU, nAxisBinsU);

  int32_t nKnotsTot = spline.getNumberOfKnots();
  std::cout << "Knots: initial " << initNKnotsU << ", created " << nKnotsTot << std::endl;
  for (int32_t i = 0; i < nKnotsTot; i++) {
    std::cout << "knot " << i << ": " << spline.getKnot(i).u << std::endl;
  }

  const IrregularSpline1D& gridU = spline;

  myfloat* data0 = new myfloat[nKnotsTot]; // original data
  myfloat* data = new myfloat[nKnotsTot];  // corrected data

  int32_t nu = gridU.getNumberOfKnots();

  for (int32_t i = 0; i < gridU.getNumberOfKnots(); i++) {
    data0[i] = F(gridU.getKnot(i).u);
    // SG random
    data0[i] = gRandom->Uniform(-1., 1.);
    data[i] = data0[i];
  }

  spline.correctEdges(data);

  TCanvas* canv = new TCanvas("cQA", "IrregularSpline1D  QA", 2000, 1000);
  canv->Draw();

  TH1F* qaX = new TH1F("qaX", "qaX [um]", 1000, -1000., 1000.);

  int32_t iter = 0;
  float stepu = 1.e-4;
  int32_t nSteps = (int32_t)(1. / stepu + 1);

  TNtuple* knots = new TNtuple("knots", "knots", "u:f");
  double diff = 0;
  for (int32_t i = 0; i < nu; i++) {
    double u = gridU.getKnot(i).u;
    double f0 = data0[i];
    knots->Fill(u, f0);
    double fs = spline.getSpline(data, u);
    diff += (fs - f0) * (fs - f0);
  }
  std::cout << "mean diff at knots: " << sqrt(diff) / nu << std::endl;

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

  /*
      canv->cd(1);
        qaX->Draw();
        canv->cd(2);
   */
  n->SetMarkerColor(kBlack);
  // n->Draw("f0:u");
  n->SetMarkerColor(kBlue);
  n->Draw("fs:u", "", "");
  knots->Draw("f:u", "", "same");

  canv->Update();

  return 0;
}

#endif
