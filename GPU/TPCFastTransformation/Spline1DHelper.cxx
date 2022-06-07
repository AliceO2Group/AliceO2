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

/// \file  Spline1DHelper.cxx
/// \brief Implementation of Spline1DHelper class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "Spline1DHelper.h"
#include "BandMatrixSolver.h"
#include "SymMatrixSolver.h"
#include "GPUCommonLogger.h"

#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompBK.h"
#include <vector>

#include "TRandom.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TFile.h"
#include "GPUCommonMath.h"
#include <iostream>

templateClassImp(GPUCA_NAMESPACE::gpu::Spline1DHelper);

using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
Spline1DHelper<DataT>::Spline1DHelper() : mError(), mSpline()
{
}

template <typename DataT>
int Spline1DHelper<DataT>::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
}

template <typename DataT>
void Spline1DHelper<DataT>::getScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                                             double& cSl, double& cDl, double& cSr, double& cDr)
{
  /// Get derivatives of the interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
  /// over the spline values Sl, Sr and the slopes Dl, Dr

  // F(u) = cSl * Sl + cSr * Sr + cDl * Dl + cDr * Dr;

  u = u - knotL.u;
  double v = u * double(knotL.Li); // scaled u
  double vm1 = v - 1.;
  double a = u * vm1;
  double v2 = v * v;
  cSl = v2 * (2 * v - 3.) + 1.; // == 2*v3 - 3*v2 + 1
  cDl = vm1 * a;                // == (v2 - 2v + 1)*u
  cSr = 1. - cSl;
  cDr = v * a; // == (v2 - v)*u
}

template <typename DataT>
void Spline1DHelper<DataT>::getDScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                                              double& cSl, double& cDl, double& cSr, double& cDr)
{
  u = u - knotL.u;
  double dv = double(knotL.Li);
  double v = u * dv; // scaled u
  double v2 = v * v;
  cSl = 6 * (v2 - v) * dv;
  cDl = 3 * v2 - 4 * v + 1.;
  cSr = -cSl;
  cDr = 3 * v2 - 2 * v;
  // at v==0 : 0, 1, 0, 0
  // at v==1 : 0, 0, 0, 1
}

template <typename DataT>
void Spline1DHelper<DataT>::getDDScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                                               double& cSl, double& cDl, double& cSr, double& cDr)
{
  u = u - knotL.u;
  double dv = double(knotL.Li);
  double v = u * dv; // scaled u
  cSl = (12 * v - 6) * dv * dv;
  cDl = (6 * v - 4) * dv;
  cSr = -cSl;
  cDr = (6 * v - 2) * dv;
}

template <typename DataT>
void Spline1DHelper<DataT>::getDDScoefficientsLeft(const typename Spline1D<double>::Knot& knotL,
                                                   double& cSl, double& cDl, double& cSr, double& cDr)
{
  double dv = double(knotL.Li);
  cSl = -6 * dv * dv;
  cDl = -4 * dv;
  cSr = -cSl;
  cDr = -2 * dv;
}

template <typename DataT>
void Spline1DHelper<DataT>::getDDScoefficientsRight(const typename Spline1D<double>::Knot& knotL,
                                                    double& cSl, double& cDl, double& cSr, double& cDr)
{
  double dv = double(knotL.Li);
  cSl = 6 * dv * dv;
  cDl = 2 * dv;
  cSr = -cSl;
  cDr = 4 * dv;
}

template <typename DataT>
void Spline1DHelper<DataT>::getDDDScoefficients(const typename Spline1D<double>::Knot& knotL,
                                                double& cSl, double& cDl, double& cSr, double& cDr)
{
  double dv = double(knotL.Li);
  cSl = 12 * dv * dv * dv;
  cDl = 6 * dv * dv;
  cSr = -cSl;
  cDr = cDl;
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateDataPoints(
  Spline1DContainer<DataT>& spline,
  double xMin, double xMax,
  const double vx[], const double vf[], int nDataPoints)
{
  /// Create best-fit spline parameters for a given input function F

  assert(spline.isConstructed());

  const int nYdim = spline.getYdimensions();
  spline.setXrange(xMin, xMax);

  // create the same spline in double precision
  setSpline(spline);

  const int nPar = 2 * spline.getNumberOfKnots(); // n parameters for 1-Dimentional Y

  // BandMatrixSolver<6> band(nPar, nYdim);
  SymMatrixSolver band(nPar, nYdim);

  for (int iPoint = 0; iPoint < nDataPoints; ++iPoint) {
    double u = mSpline.convXtoU(vx[iPoint]);
    int iKnot = mSpline.getLeftKnotIndexForU(u);
    const typename Spline1D<double>::Knot& knot0 = mSpline.getKnot(iKnot);
    double cS0, cZ0, cS1, cZ1;
    getScoefficients(knot0, u, cS0, cZ0, cS1, cZ1);
    double c[4] = {cS0, cZ0, cS1, cZ1};

    // chi2 += (c[0]*S0 + c[1]*Z0 + c[2]*S1 + c[3]*Z1 - f)^2

    int i = 2 * iKnot;              // index of parameter S0
    for (int j = 0; j < 4; j++) {   // parameters S0, Z0, S1, Z1
      for (int k = j; k < 4; k++) { // loop over the second parameter
        band.A(i + j, i + k) += c[j] * c[k];
      }
    }
    const double* f = &vf[iPoint * nYdim];
    for (int j = 0; j < 4; j++) { // parameters S0, Z0, S1, Z1
      for (int dim = 0; dim < nYdim; dim++) {
        band.B(i + j, dim) += c[j] * f[dim];
      }
    }
  }

  for (int iKnot = 0; iKnot < spline.getNumberOfKnots() - 2; ++iKnot) {
    const typename Spline1D<double>::Knot& knot0 = mSpline.getKnot(iKnot);
    const typename Spline1D<double>::Knot& knot1 = mSpline.getKnot(iKnot + 1);
    // const typename Spline1D<double>::Knot& knot2 = mSpline.getKnot(iKnot + 2);

    // set S'' and S''' at knot1 equal at both sides
    // chi2 += w^2*(S''from the left - S'' from the right)^2
    // chi2 += w^2*(S'''from the left - S''' from the right)^2

    for (int order = 2; order <= 3; order++) {
      double cS0, cZ0, cS1r, cZ1r;
      double cS1l, cZ1l, cS2, cZ2;
      if (order == 2) {
        getDDScoefficientsRight(knot0, cS0, cZ0, cS1r, cZ1r);
        getDDScoefficientsLeft(knot1, cS1l, cZ1l, cS2, cZ2);
      } else {
        getDDDScoefficients(knot0, cS0, cZ0, cS1r, cZ1r);
        getDDDScoefficients(knot1, cS1l, cZ1l, cS2, cZ2);
      }

      double w = 0.01;
      double c[6] = {w * cS0, w * cZ0,
                     w * (cS1r - cS1l), w * (cZ1r - cZ1l),
                     -w * cS2, -w * cZ2};

      // chi2 += w^2*(c[0]*S0 + c[1]*Z0 + c[2]*S1 + c[3]*Z1 + c[4]*S2 + c[5]*Z2)^2

      int i = 2 * iKnot;              // index of parameter S0
      for (int j = 0; j < 6; j++) {   // loop over 6 parameters
        for (int k = j; k < 6; k++) { // loop over the second parameter
          band.A(i + j, i + k) += c[j] * c[k];
        }
      }
    }
  } // iKnot

  // experimental: set slopes at neighbouring knots equal - doesn't work
  /*
    for (int iKnot = 0; iKnot < spline.getNumberOfKnots() - 2; ++iKnot) {
      const typename Spline1D<double>::Knot& knot0 = mSpline.getKnot(iKnot);
      const typename Spline1D<double>::Knot& knot1 = mSpline.getKnot(iKnot + 1);
      double w = 1.;
      int i = 2 * iKnot; // index of parameter S0
      double d = knot1.u - knot0.u;
      {
        double c1[4] = {1, d, -1, 0};
        double c2[4] = {1, 0, -1, d};
        // chi2 += w*(c1[0]*S0 + c1[1]*Z0 + c1[2]*S1 + c1[3]*Z1)^2
        // chi2 += w*(c2[0]*S0 + c2[1]*Z0 + c2[2]*S1 + c2[3]*Z1)^2
        for (int j = 0; j < 4; j++) {   // parameters S0, Z0, S1, Z1
          for (int k = j; k < 4; k++) { // loop over the second parameter
            band.A(i + j, i + k) += w * (c1[j] * c1[k] + c2[j] * c2[k]);
          }
        }
      }
    } // iKnot
  }
  */

  band.solve();

  for (int i = 0; i < nPar; i++) {
    for (int j = 0; j < nYdim; j++) {
      spline.getParameters()[i * nYdim + j] = band.B(i, j);
    }
  }
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateDerivatives(
  Spline1DContainer<DataT>& spline,
  const double vx[], const double vf[], int nDataPoints)
{
  /// Create best-fit spline parameters for a given input function F

  assert(spline.isConstructed());

  const int nYdim = spline.getYdimensions();

  // create the same spline in double precision
  setSpline(spline);

  const int nPar = spline.getNumberOfKnots(); // n parameters for 1-Dimentional Y

  BandMatrixSolver<2> band(nPar, nYdim);

  for (int iPoint = 0; iPoint < nDataPoints; ++iPoint) {
    double u = mSpline.convXtoU(vx[iPoint]);
    int iKnot = mSpline.getLeftKnotIndexForU(u);
    const typename Spline1D<double>::Knot& knot0 = mSpline.getKnot(iKnot);
    double cS0, cZ0, cS1, cZ1;
    getScoefficients(knot0, u, cS0, cZ0, cS1, cZ1);
    double c[2] = {cZ0, cZ1};

    // chi2 += (cS0*S0 + c[0]*Z0 + cS1*S1 + c[1]*Z1 - f)^2
    //      == (c[0]*Z0 + c[1]*Z1 - (f - cS0*S0 - cS1*S1))^2

    int i = iKnot; // index of parameter Z0
    band.A(i + 0, i + 0) += c[0] * c[0];
    band.A(i + 0, i + 1) += c[0] * c[1];
    band.A(i + 1, i + 1) += c[1] * c[1];

    const double* f = &vf[iPoint * nYdim];
    const DataT* S0 = &spline.getParameters()[2 * iKnot * nYdim];
    const DataT* S1 = &spline.getParameters()[2 * (iKnot + 1) * nYdim];
    for (int j = 0; j < 2; j++) { // parameters Z0, Z1
      for (int dim = 0; dim < nYdim; dim++) {
        band.B(i + j, dim) += c[j] * (f[dim] - cS0 * S0[dim] - cS1 * S1[dim]);
      }
    }
  }

  band.solve();

  for (int i = 0; i < nPar; i++) {
    for (int j = 0; j < nYdim; j++) {
      spline.getParameters()[(2 * i + 1) * nYdim + j] = band.B(i, j);
    }
  }
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunctionClassic(Spline1DContainer<DataT>& spline,
                                                       double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F)
{
  /// Create classic spline parameters for a given input function F
  /// set slopes at the knots such, that the second derivative of the spline is continious.

  int Ndim = spline.getYdimensions();
  const int nKnots = spline.getNumberOfKnots();

  TMatrixD A(nKnots, nKnots);

  A.Zero();

  /*
    const Spline1D::Knot& knot0 = spline.getKnot(i);
    double x = (u - knot0.u) * knot0.Li; // scaled u
    double cS1 = (6 - 12*x)*knot0.Li*knot0.Li;
    double cZ0 = (6*x-4)*knot0.Li;
    double cZ1 = (6*x-2)*knot0.Li;
    // f''(u) = cS1*(f1-f0) + cZ0*z0 + cZ1*z1;
   */

  // second derivative at knot0 is 0
  {
    const typename Spline1D<DataT>::Knot& knot0 = spline.getKnot(0);
    double cZ0 = (-4) * knot0.Li;
    double cZ1 = (-2) * knot0.Li;
    // f''(u) = cS1*(f1-f0) + cZ0*z0 + cZ1*z1;
    A(0, 0) = cZ0;
    A(0, 1) = cZ1;
  }

  // second derivative at knot nKnots-1  is 0
  {
    const typename Spline1D<DataT>::Knot& knot0 = spline.getKnot(nKnots - 2);
    double cZ0 = (6 - 4) * knot0.Li;
    double cZ1 = (6 - 2) * knot0.Li;
    // f''(u) = cS1*(f1-f0) + cZ0*z0 + cZ1*z1;
    A(nKnots - 1, nKnots - 2) = cZ0;
    A(nKnots - 1, nKnots - 1) = cZ1;
  }

  // second derivative at other knots is same from the left and from the right
  for (int i = 1; i < nKnots - 1; i++) {
    const typename Spline1D<DataT>::Knot& knot0 = spline.getKnot(i - 1);
    double cZ0 = (6 - 4) * knot0.Li;
    double cZ1_0 = (6 - 2) * knot0.Li;
    // f''(u) = cS1*(f1-f0) + cZ0*z0 + cZ1*z1;

    const typename Spline1D<DataT>::Knot& knot1 = spline.getKnot(i);
    double cZ1_1 = (-4) * knot1.Li;
    double cZ2 = (-2) * knot1.Li;
    // f''(u) = cS2*(f2-f1) + cZ1_1*z1 + cZ2*z2;
    A(i, i - 1) = cZ0;
    A(i, i) = cZ1_0 - cZ1_1;
    A(i, i + 1) = -cZ2;
  }

  A.Invert();

  spline.setXrange(xMin, xMax);
  DataT* parameters = spline.getParameters();

  TVectorD b(nKnots);
  b.Zero();

  double uToXscale = (((double)xMax) - xMin) / spline.getUmax();

  for (int i = 0; i < nKnots; ++i) {
    const typename Spline1D<DataT>::Knot& knot = spline.getKnot(i);
    double u = knot.u;
    double f[Ndim];
    F(xMin + u * uToXscale, f);
    for (int dim = 0; dim < Ndim; dim++) {
      parameters[(2 * i) * Ndim + dim] = f[dim];
    }
  }

  for (int dim = 0; dim < Ndim; dim++) {

    // second derivative at knot0 is 0
    {
      double f0 = parameters[(2 * 0) * Ndim + dim];
      double f1 = parameters[(2 * 1) * Ndim + dim];
      const typename Spline1D<DataT>::Knot& knot0 = spline.getKnot(0);
      double cS1 = (6) * knot0.Li * knot0.Li;
      // f''(u) = cS1*(f1-f0) + cZ0*z0 + cZ1*z1;
      b(0) = -cS1 * (f1 - f0);
    }

    // second derivative at knot nKnots-1  is 0
    {
      double f0 = parameters[2 * (nKnots - 2) * Ndim + dim];
      double f1 = parameters[2 * (nKnots - 1) * Ndim + dim];
      const typename Spline1D<DataT>::Knot& knot0 = spline.getKnot(nKnots - 2);
      double cS1 = (6 - 12) * knot0.Li * knot0.Li;
      // f''(u) = cS1*(f1-f0) + cZ0*z0 + cZ1*z1;
      b(nKnots - 1) = -cS1 * (f1 - f0);
    }

    // second derivative at other knots is same from the left and from the right
    for (int i = 1; i < nKnots - 1; i++) {
      double f0 = parameters[2 * (i - 1) * Ndim + dim];
      double f1 = parameters[2 * (i)*Ndim + dim];
      double f2 = parameters[2 * (i + 1) * Ndim + dim];
      const typename Spline1D<DataT>::Knot& knot0 = spline.getKnot(i - 1);
      double cS1 = (6 - 12) * knot0.Li * knot0.Li;
      // f''(u) = cS1*(f1-f0) + cZ0*z0 + cZ1*z1;

      const typename Spline1D<DataT>::Knot& knot1 = spline.getKnot(i);
      double cS2 = (6) * knot1.Li * knot1.Li;
      // f''(u) = cS2*(f2-f1) + cZ1_1*z1 + cZ2*z2;
      b(i) = -cS1 * (f1 - f0) + cS2 * (f2 - f1);
    }

    TVectorD c = A * b;
    for (int i = 0; i < nKnots; i++) {
      parameters[(2 * i + 1) * Ndim + dim] = c[i];
    }
  }
}

template <typename DataT>
void Spline1DHelper<DataT>::makeDataPoints(
  Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
  int nAuxiliaryDataPoints, std::vector<double>& vx, std::vector<double>& vf)
{
  /// Create best-fit spline parameters for a given input function F
  assert(spline.isConstructed());
  int nFdimensions = spline.getYdimensions();
  int nDataPoints = 0;
  if (nAuxiliaryDataPoints < 2) {
    storeError(-3, "Spline1DHelper::setSpline: too few nAuxiliaryDataPoints, increase to 2");
    nAuxiliaryDataPoints = 2;
  }
  spline.setXrange(xMin, xMax);
  setSpline(spline);
  nDataPoints = 1 + mSpline.getUmax() + mSpline.getUmax() * nAuxiliaryDataPoints;

  vx.resize(nDataPoints);
  vf.resize(nDataPoints * nFdimensions);

  double scalePoints2Knots = ((double)mSpline.getUmax()) / (nDataPoints - 1.);
  for (int i = 0; i < nDataPoints; ++i) {
    double u = i * scalePoints2Knots;
    double x = mSpline.convUtoX(u);
    vx[i] = x;
    F(x, &vf[i * nFdimensions]);
  }
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunction(
  Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
  int nAuxiliaryDataPoints)
{
  /// Create best-fit spline parameters for a given input function F
  std::vector<double> vx;
  std::vector<double> vf;
  makeDataPoints(spline, xMin, xMax, F, nAuxiliaryDataPoints, vx, vf);
  approximateDataPoints(spline, xMin, xMax, &vx[0], &vf[0], vx.size());
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunctionGradually(
  Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
  int nAuxiliaryDataPoints)
{
  /// Create best-fit spline parameters for a given input function F

  std::vector<double> vx;
  std::vector<double> vf;
  makeDataPoints(spline, xMin, xMax, F, nAuxiliaryDataPoints, vx, vf);
  int nDataPoints = vx.size();
  spline.setXrange(xMin, xMax);
  setSpline(spline);

  int nFdimensions = spline.getYdimensions();

  // set F values at knots
  for (int iKnot = 0; iKnot < mSpline.getNumberOfKnots(); ++iKnot) {
    const typename Spline1D<double>::Knot& knot = mSpline.getKnot(iKnot);
    double x = mSpline.convUtoX(knot.u);
    double s[nFdimensions];
    F(x, s);
    for (int dim = 0; dim < nFdimensions; dim++) {
      spline.getParameters()[2 * iKnot * nFdimensions + dim] = s[dim];
    }
  }
  approximateDerivatives(spline, &vx[0], &vf[0], nDataPoints);
}

template <typename DataT>
void Spline1DHelper<DataT>::setSpline(const Spline1DContainer<DataT>& spline)
{
  const int nKnots = spline.getNumberOfKnots();
  std::vector<int> knots(nKnots);
  for (int i = 0; i < nKnots; i++) {
    knots[i] = spline.getKnot(i).getU();
  }
  mSpline.recreate(0, nKnots, knots.data());
  mSpline.setXrange(spline.getXmin(), spline.getXmax());
}

#ifndef GPUCA_ALIROOT_LIB
template <typename DataT>
int Spline1DHelper<DataT>::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

  BandMatrixSolver<0>::test(0);

  // input function F

  const int Ndim = 5;
  const int Fdegree = 4;
  double Fcoeff[Ndim][2 * (Fdegree + 1)];

  auto F = [&](double x, double f[]) -> void {
    double cosx[Fdegree + 1], sinx[Fdegree + 1];
    double xi = 0;
    for (int i = 0; i <= Fdegree; i++, xi += x) {
      GPUCommonMath::SinCosd(xi, sinx[i], cosx[i]);
    }
    for (int dim = 0; dim < Ndim; dim++) {
      f[dim] = 0; // Fcoeff[0]/2;
      for (int i = 1; i <= Fdegree; i++) {
        f[dim] += Fcoeff[dim][2 * i] * cosx[i] + Fcoeff[dim][2 * i + 1] * sinx[i];
      }
    }
  };

  TCanvas* canv = nullptr;
  TNtuple* nt = nullptr;
  TNtuple* knots = nullptr;

  auto ask = [&]() -> bool {
    if (!canv) {
      return 0;
    }
    canv->Update();
    LOG(info) << "type 'q ' to exit";
    std::string str;
    std::getline(std::cin, str);
    return (str != "q" && str != ".q");
  };

  LOG(info) << "Test 1D interpolation with the compact spline";

  int nTries = 100;

  if (draw) {
    canv = new TCanvas("cQA", "Spline1D  QA", 1000, 600);
    nTries = 10000;
  }

  double statDf1 = 0;
  double statDf2 = 0;
  double statDf1D = 0;
  double statN = 0;

  int seed = 1;

  for (int itry = 0; itry < nTries; itry++) {

    // init random F
    for (int dim = 0; dim < Ndim; dim++) {
      gRandom->SetSeed(seed++);
      for (int i = 0; i < 2 * (Fdegree + 1); i++) {
        Fcoeff[dim][i] = gRandom->Uniform(-1, 1);
      }
    }

    // spline

    int nKnots = 4;
    const int uMax = nKnots * 3;

    Spline1D<DataT, Ndim> spline1;
    int knotsU[nKnots];

    do { // set knots randomly
      knotsU[0] = 0;
      double du = 1. * uMax / (nKnots - 1);
      for (int i = 1; i < nKnots; i++) {
        knotsU[i] = (int)(i * du); // + gRandom->Uniform(-du / 3, du / 3);
      }
      knotsU[nKnots - 1] = uMax;
      spline1.recreate(nKnots, knotsU);
      if (nKnots != spline1.getNumberOfKnots()) {
        LOG(info) << "warning: n knots changed during the initialisation " << nKnots
                  << " -> " << spline1.getNumberOfKnots();
        continue;
      }
    } while (0);

    std::string err = FlatObject::stressTest(spline1);
    if (!err.empty()) {
      LOG(info) << "error at FlatObject functionality: " << err;
      return -1;
    } else {
      // LOG(info) << "flat object functionality is ok" ;
    }

    nKnots = spline1.getNumberOfKnots();
    int nAuxiliaryPoints = 2;
    Spline1D<DataT, Ndim> spline2(spline1);
    spline1.approximateFunction(0., TMath::Pi(), F, nAuxiliaryPoints);

    //if (itry == 0)
    {
      TFile outf("testSpline1D.root", "recreate");
      if (outf.IsZombie()) {
        LOG(info) << "Failed to open output file testSpline1D.root ";
      } else {
        const char* name = "Spline1Dtest";
        spline1.writeToFile(outf, name);
        Spline1D<DataT, Ndim>* p = spline1.readFromFile(outf, name);
        if (p == nullptr) {
          LOG(info) << "Failed to read Spline1D from file testSpline1D.root ";
        } else {
          spline1 = *p;
        }
        outf.Close();
      }
    }

    Spline1DHelper<DataT> helper;
    helper.approximateFunctionGradually(spline2, 0., TMath::Pi(), F, nAuxiliaryPoints);

    // 1-D splines for each dimension
    Spline1D<DataT, 1> splines3[Ndim];
    {
      for (int dim = 0; dim < Ndim; dim++) {
        auto F3 = [&](double u, double f[]) -> void {
          double ff[Ndim];
          F(u, ff);
          f[0] = ff[dim];
        };
        splines3[dim].recreate(nKnots, knotsU);
        splines3[dim].approximateFunction(0., TMath::Pi(), F3, nAuxiliaryPoints);
      }
    }

    double stepX = 1.e-2;
    for (double x = 0; x < TMath::Pi(); x += stepX) {
      double f[Ndim];
      DataT s1[Ndim], s2[Ndim];
      F(x, f);
      spline1.interpolate(x, s1);
      spline2.interpolate(x, s2);
      for (int dim = 0; dim < Ndim; dim++) {
        statDf1 += (s1[dim] - f[dim]) * (s1[dim] - f[dim]);
        statDf2 += (s2[dim] - f[dim]) * (s2[dim] - f[dim]);
        DataT s1D = splines3[dim].interpolate(x);
        statDf1D += (s1D - s1[dim]) * (s1D - s1[dim]);
      }
      statN += Ndim;
    }
    // LOG(info) << "std dev   : " << sqrt(statDf1 / statN) ;

    if (draw) {
      delete nt;
      delete knots;
      nt = new TNtuple("nt", "nt", "u:f:s");
      double drawMax = -1.e20;
      double drawMin = 1.e20;
      double stepX = 1.e-4;
      for (double x = 0; x < TMath::Pi(); x += stepX) {
        double f[Ndim];
        DataT s[Ndim];
        F(x, f);
        spline1.interpolate(x, s);
        nt->Fill(spline1.convXtoU(x), f[0], s[0]);
        drawMax = std::max(drawMax, std::max(f[0], (double)s[0]));
        drawMin = std::min(drawMin, std::min(f[0], (double)s[0]));
      }

      nt->SetMarkerStyle(8);

      {
        TNtuple* ntRange = new TNtuple("ntRange", "nt", "u:f");
        drawMin -= 0.1 * (drawMax - drawMin);

        ntRange->Fill(0, drawMin);
        ntRange->Fill(0, drawMax);
        ntRange->Fill(uMax, drawMin);
        ntRange->Fill(uMax, drawMax);
        ntRange->SetMarkerColor(kWhite);
        ntRange->SetMarkerSize(0.1);
        ntRange->Draw("f:u", "", "");
        delete ntRange;
      }

      nt->SetMarkerColor(kGray);
      nt->SetMarkerSize(2.);
      nt->Draw("f:u", "", "P,same");

      nt->SetMarkerSize(.5);
      nt->SetMarkerColor(kBlue);
      nt->Draw("s:u", "", "P,same");

      knots = new TNtuple("knots", "knots", "type:u:s");
      for (int i = 0; i < nKnots; i++) {
        double u = spline1.getKnot(i).u;
        DataT s[Ndim];
        spline1.interpolate(spline1.convUtoX(u), s);
        knots->Fill(1, u, s[0]);
      }

      knots->SetMarkerStyle(8);
      knots->SetMarkerSize(1.5);
      knots->SetMarkerColor(kRed);
      knots->SetMarkerSize(1.5);
      knots->Draw("s:u", "type==1", "same"); // knots

      if (drawDataPoints) {
        std::vector<double> vx;
        std::vector<double> vf;
        helper.makeDataPoints(spline1, 0., TMath::Pi(), F, nAuxiliaryPoints, vx, vf);
        for (unsigned int j = 0; j < vx.size(); j++) {
          DataT s[Ndim];
          spline1.interpolate(vx[j], s);
          knots->Fill(2, spline1.convXtoU(vx[j]), s[0]);
        }
        knots->SetMarkerColor(kBlack);
        knots->SetMarkerSize(1.);
        knots->Draw("s:u", "type==2", "same"); // data points
      }
      if (!ask()) {
        break;
      }
    } // draw
  }
  //delete canv;
  //delete nt;
  //delete knots;

  statDf1 = sqrt(statDf1 / statN);
  statDf2 = sqrt(statDf2 / statN);
  statDf1D = sqrt(statDf1D / statN);

  LOG(info) << std::defaultfloat;

  LOG(info) << "\n std dev for Compact Spline   : " << statDf1 << " / " << statDf2;
  LOG(info) << " mean difference between 1-D and " << Ndim
            << "-D splines   : " << statDf1D;

  if (statDf1 < 0.05 && statDf2 < 0.06 && statDf1D < 1.e-20) {
    LOG(info) << "Everything is fine";
  } else {
    LOG(info) << "Something is wrong!!";
    return -2;
  }
  return 0;
}
#endif

template class GPUCA_NAMESPACE::gpu::Spline1DHelper<float>;
template class GPUCA_NAMESPACE::gpu::Spline1DHelper<double>;
