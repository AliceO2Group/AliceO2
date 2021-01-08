// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline1DHelper.cxx
/// \brief Implementation of Spline1DHelper class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

#include "Spline1DHelper.h"
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
Spline1DHelper<DataT>::Spline1DHelper() : mError(), mSpline(), mFdimensions(0)
{
}

template <typename DataT>
int Spline1DHelper<DataT>::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
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
void Spline1DHelper<DataT>::approximateFunction(
  Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
  int nAuxiliaryDataPoints)
{
  /// Create best-fit spline parameters for a given input function F
  setSpline(spline, spline.getYdimensions(), nAuxiliaryDataPoints);
  approximateFunction(spline.getParameters(), xMin, xMax, F);
  spline.setXrange(xMin, xMax);
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunctionGradually(
  Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
  int nAuxiliaryDataPoints)
{
  /// Create best-fit spline parameters gradually for a given input function F
  setSpline(spline, spline.getYdimensions(), nAuxiliaryDataPoints);
  approximateFunctionGradually(spline.getParameters(), xMin, xMax, F);
  spline.setXrange(xMin, xMax);
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunction(
  DataT* Sparameters, double xMin, double xMax, std::function<void(double x, double f[])> F) const
{
  /// Create best-fit spline parameters for a given input function F
  /// output in Sparameters
  std::vector<double> vF(getNumberOfDataPoints() * mFdimensions);
  double mUtoXscale = (((double)xMax) - xMin) / mSpline.getUmax();
  for (int i = 0; i < getNumberOfDataPoints(); i++) {
    F(xMin + mUtoXscale * mDataPoints[i].u, &vF[i * mFdimensions]);
  }
  approximateFunction(Sparameters, vF.data());
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunctionGradually(
  DataT* Sparameters, double xMin, double xMax, std::function<void(double x, double f[])> F) const
{
  /// Create best-fit spline parameters gradually for a given input function F
  /// output in Sparameters
  std::vector<double> vF(getNumberOfDataPoints() * mFdimensions);
  double mUtoXscale = (((double)xMax) - xMin) / mSpline.getUmax();
  for (int i = 0; i < getNumberOfDataPoints(); i++) {
    F(xMin + mUtoXscale * mDataPoints[i].u, &vF[i * mFdimensions]);
  }
  approximateFunctionGradually(Sparameters, vF.data());
}

template <typename DataT>
int Spline1DHelper<DataT>::setSpline(
  const Spline1DContainer<DataT>& spline, int nFdimensions, int nAuxiliaryDataPoints)
{
  // Prepare creation of a best-fit spline
  //
  // Data points will be set at all integer U (that includes all knots),
  // plus at nAuxiliaryDataPoints points between the integers.
  //
  // nAuxiliaryDataPoints must be >= 2
  //
  // nAuxiliaryDataPoints==1 is also possible, but there must be at least
  // one integer U without a knot, in order to get 2*nKnots data points in total.
  //
  // The return value is an error index, 0 means no error

  int ret = 0;

  mFdimensions = nFdimensions;
  int nPoints = 0;
  if (!spline.isConstructed()) {
    ret = storeError(-1, "Spline1DHelper<DataT>::setSpline: input spline is not constructed");
    mSpline.recreate(0, 2);
    nAuxiliaryDataPoints = 2;
    nPoints = 4;
  } else {
    std::vector<int> knots;
    for (int i = 0; i < spline.getNumberOfKnots(); i++) {
      knots.push_back(spline.getKnot(i).getU());
    }
    mSpline.recreate(0, spline.getNumberOfKnots(), knots.data());

    nPoints = 1 + mSpline.getUmax() + mSpline.getUmax() * nAuxiliaryDataPoints;
    if (nPoints < 2 * mSpline.getNumberOfKnots()) {
      nAuxiliaryDataPoints = 2;
      nPoints = 1 + mSpline.getUmax() + mSpline.getUmax() * nAuxiliaryDataPoints;
      ret = storeError(-3, "Spline1DHelper::setSpline: too few nAuxiliaryDataPoints, increase to 2");
    }
  }

  const int nPar = 2 * mSpline.getNumberOfKnots(); // n parameters for 1D

  mDataPoints.resize(nPoints);
  double scalePoints2Knots = ((double)mSpline.getUmax()) / (nPoints - 1.);

  for (int i = 0; i < nPoints; ++i) {
    DataPoint& p = mDataPoints[i];
    double u = i * scalePoints2Knots;
    int iKnot = mSpline.getLeftKnotIndexForU(u);
    const typename Spline1D<double>::Knot& knot0 = mSpline.getKnot(iKnot);
    const typename Spline1D<double>::Knot& knot1 = mSpline.getKnot(iKnot + 1);
    double l = knot1.u - knot0.u;
    double s = (u - knot0.u) * knot0.Li; // scaled u
    double s2 = s * s;
    double sm1 = s - 1.;

    p.iKnot = iKnot;
    p.isKnot = 0;
    p.u = u;
    p.cS1 = s2 * (3. - 2. * s);
    p.cS0 = 1. - p.cS1;
    p.cZ0 = s * sm1 * sm1 * l;
    p.cZ1 = s2 * sm1 * l;
  }

  const int nKnots = mSpline.getNumberOfKnots();

  mKnotDataPoints.resize(nKnots);

  for (int i = 0; i < nKnots; ++i) {
    const typename Spline1D<double>::Knot& knot = mSpline.getKnot(i);
    int iu = (int)(knot.u + 0.1f);
    mKnotDataPoints[i] = iu * (1 + nAuxiliaryDataPoints);
    mDataPoints[mKnotDataPoints[i]].isKnot = 1;
  }

  TMatrixDSym A(nPar);
  A.Zero();

  for (int i = 0; i < nPoints; ++i) {
    DataPoint& p = mDataPoints[i];
    int j = p.iKnot * 2;
    A(j + 0, j + 0) += p.cS0 * p.cS0;
    A(j + 1, j + 0) += p.cS0 * p.cZ0;
    A(j + 2, j + 0) += p.cS0 * p.cS1;
    A(j + 3, j + 0) += p.cS0 * p.cZ1;

    A(j + 1, j + 1) += p.cZ0 * p.cZ0;
    A(j + 2, j + 1) += p.cZ0 * p.cS1;
    A(j + 3, j + 1) += p.cZ0 * p.cZ1;

    A(j + 2, j + 2) += p.cS1 * p.cS1;
    A(j + 3, j + 2) += p.cS1 * p.cZ1;

    A(j + 3, j + 3) += p.cZ1 * p.cZ1;
  }

  // copy symmetric matrix elements

  for (int i = 0; i < nPar; i++) {
    for (int j = i + 1; j < nPar; j++) {
      A(i, j) = A(j, i);
    }
  }

  TMatrixDSym Z(nKnots);
  mLSMmatrixSvalues.resize(nKnots * nKnots);
  for (int i = 0, k = 0; i < nKnots; i++) {
    for (int j = 0; j < nKnots; j++, k++) {
      mLSMmatrixSvalues[k] = A(i * 2 + 1, j * 2);
      Z(i, j) = A(i * 2 + 1, j * 2 + 1);
    }
  }

  {
    TDecompBK bk(A, 0);
    bool ok = bk.Invert(A);

    if (!ok) {
      ret = storeError(-4, "Spline1DHelper::setSpline: internal error - can not invert the matrix");
      A.Zero();
    }
    mLSMmatrixFull.resize(nPar * nPar);
    for (int i = 0, k = 0; i < nPar; i++) {
      for (int j = 0; j < nPar; j++, k++) {
        mLSMmatrixFull[k] = A(i, j);
      }
    }
  }

  {
    TDecompBK bk(Z, 0);
    if (!bk.Invert(Z)) {
      ret = storeError(-5, "Spline1DHelper::setSpline: internal error - can not invert the matrix");
      Z.Zero();
    }
    mLSMmatrixSderivatives.resize(nKnots * nKnots);
    for (int i = 0, k = 0; i < nKnots; i++) {
      for (int j = 0; j < nKnots; j++, k++) {
        mLSMmatrixSderivatives[k] = Z(i, j);
      }
    }
  }

  return ret;
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunction(
  DataT* Sparameters,
  const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// Approximate a function given as an array of values at data points

  const int nPar = 2 * mSpline.getNumberOfKnots();
  double b[nPar];
  for (int idim = 0; idim < mFdimensions; idim++) {
    for (int i = 0; i < nPar; i++) {
      b[i] = 0.;
    }
    for (int i = 0; i < getNumberOfDataPoints(); ++i) {
      const DataPoint& p = mDataPoints[i];
      double* bb = &(b[p.iKnot * 2]);
      double f = (double)DataPointF[i * mFdimensions + idim];
      bb[0] += f * p.cS0;
      bb[1] += f * p.cZ0;
      bb[2] += f * p.cS1;
      bb[3] += f * p.cZ1;
    }

    const double* row = mLSMmatrixFull.data();

    for (int i = 0; i < nPar; i++, row += nPar) {
      double s = 0.;
      for (int j = 0; j < nPar; j++) {
        s += row[j] * b[j];
      }
      Sparameters[i * mFdimensions + idim] = (float)s;
    }
  }
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateFunctionGradually(
  DataT* Sparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim */]) const
{
  /// gradually approximate a function given as an array of values at data points
  /// output in Sparameters
  copySfromDataPoints(Sparameters, DataPointF);
  approximateDerivatives(Sparameters, DataPointF);
}

template <typename DataT>
void Spline1DHelper<DataT>::copySfromDataPoints(
  DataT* Sparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// a tool for the gradual approximation: set spline values S_i at knots == function values
  /// output in Sparameters
  for (int i = 0; i < mSpline.getNumberOfKnots(); ++i) { // set F values at knots
    int ip = mKnotDataPoints[i];
    for (int d = 0; d < mFdimensions; d++) {
      Sparameters[2 * i * mFdimensions + d] = DataPointF[ip * mFdimensions + d];
    }
  }
}

template <typename DataT>
void Spline1DHelper<DataT>::approximateDerivatives(
  DataT* Sparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// a tool for the gradual approximation:
  /// calibrate spline derivatives D_i using already calibrated spline values S_i
  /// input and output output in Sparameters

  const int nKnots = mSpline.getNumberOfKnots();
  const int Ndim = mFdimensions;
  double b[nKnots * mFdimensions];
  for (int i = 0; i < nKnots * Ndim; i++) {
    b[i] = 0.;
  }

  for (int i = 0; i < getNumberOfDataPoints(); ++i) {
    const DataPoint& p = mDataPoints[i];
    if (p.isKnot) {
      continue;
    }
    for (int d = 0; d < Ndim; d++) {
      double f = (double)DataPointF[i * Ndim + d];
      b[(p.iKnot + 0) * Ndim + d] += f * p.cZ0;
      b[(p.iKnot + 1) * Ndim + d] += f * p.cZ1;
    }
  }

  const double* row = mLSMmatrixSvalues.data();
  for (int i = 0; i < nKnots; ++i, row += nKnots) {
    double s[Ndim];
    for (int d = 0; d < Ndim; d++) {
      s[d] = 0.;
    }
    for (int j = 0; j < nKnots; ++j) {
      for (int d = 0; d < Ndim; d++) {
        s[d] += row[j] * Sparameters[2 * j * Ndim + d];
      }
    }
    for (int d = 0; d < Ndim; d++) {
      b[i * Ndim + d] -= s[d];
    }
  }

  row = mLSMmatrixSderivatives.data();
  for (int i = 0; i < nKnots; ++i, row += nKnots) {
    double s[Ndim];
    for (int d = 0; d < Ndim; d++) {
      s[d] = 0.;
    }
    for (int j = 0; j < nKnots; ++j) {
      for (int d = 0; d < Ndim; d++) {
        s[d] += row[j] * b[j * Ndim + d];
      }
    }
    for (int d = 0; d < Ndim; d++) {
      Sparameters[(2 * i + 1) * Ndim + d] = (float)s[d];
    }
  }
}

#ifndef GPUCA_ALIROOT_LIB
template <typename DataT>
int Spline1DHelper<DataT>::test(const bool draw, const bool drawDataPoints)
{
  using namespace std;

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
    cout << "type 'q ' to exit" << endl;
    std::string str;
    std::getline(std::cin, str);
    return (str != "q" && str != ".q");
  };

  std::cout << "Test 1D interpolation with the compact spline" << std::endl;

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
        cout << "warning: n knots changed during the initialisation " << nKnots
             << " -> " << spline1.getNumberOfKnots() << std::endl;
        continue;
      }
    } while (0);

    std::string err = FlatObject::stressTest(spline1);
    if (!err.empty()) {
      cout << "error at FlatObject functionality: " << err << endl;
      return -1;
    } else {
      // cout << "flat object functionality is ok" << endl;
    }

    nKnots = spline1.getNumberOfKnots();
    int nAuxiliaryPoints = 1;
    Spline1D<DataT, Ndim> spline2(spline1);
    spline1.approximateFunction(0., TMath::Pi(), F, nAuxiliaryPoints);
    //if (itry == 0)
    {
      TFile outf("testSpline1D.root", "recreate");
      if (outf.IsZombie()) {
        cout << "Failed to open output file testSpline1D.root " << std::endl;
      } else {
        const char* name = "Spline1Dtest";
        spline1.writeToFile(outf, name);
        Spline1D<DataT, Ndim>* p = spline1.readFromFile(outf, name);
        if (p == nullptr) {
          cout << "Failed to read Spline1D from file testSpline1D.root " << std::endl;
        } else {
          spline1 = *p;
        }
        outf.Close();
      }
    }

    Spline1DHelper<DataT> helper;
    helper.setSpline(spline2, Ndim, nAuxiliaryPoints);
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
    // cout << "std dev   : " << sqrt(statDf1 / statN) << std::endl;

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
        for (int j = 0; j < helper.getNumberOfDataPoints(); j++) {
          const typename Spline1DHelper<DataT>::DataPoint& p = helper.getDataPoint(j);
          if (p.isKnot) {
            continue;
          }
          DataT s[Ndim];
          spline1.interpolate(spline1.convUtoX(p.u), s);
          knots->Fill(2, p.u, s[0]);
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

  cout << "\n std dev for Compact Spline   : " << statDf1 << " / " << statDf2
       << std::endl;
  cout << " mean difference between 1-D and " << Ndim
       << "-D splines   : " << statDf1D << std::endl;

  if (statDf1 < 0.05 && statDf2 < 0.06 && statDf1D < 1.e-20) {
    cout << "Everything is fine" << endl;
  } else {
    cout << "Something is wrong!!" << endl;
    return -2;
  }
  return 0;
}
#endif

template class GPUCA_NAMESPACE::gpu::Spline1DHelper<float>;
template class GPUCA_NAMESPACE::gpu::Spline1DHelper<double>;

#endif
