// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  SplineHelper1D.cxx
/// \brief Implementation of SplineHelper1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

#include "SplineHelper1D.h"
#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompBK.h"

using namespace GPUCA_NAMESPACE::gpu;

template <typename DataT>
SplineHelper1D<DataT>::SplineHelper1D() : mError(), mSpline(), mFdimensions(0)
{
}

template <typename DataT>
int SplineHelper1D<DataT>::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
}

template <typename DataT>
void SplineHelper1D<DataT>::approximateFunctionClassic(Spline1D<DataT>& spline,
                                                       DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[/*spline.getFdimensions()*/])> F)
{
  /// Create classic spline parameters for a given input function F
  /// set slopes at the knots such, that the second derivative of the spline is continious.

  int Ndim = spline.getFdimensions();
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
  DataT* parameters = spline.getFparameters();

  TVectorD b(nKnots);
  b.Zero();

  double uToXscale = (((double)xMax) - xMin) / spline.getUmax();

  for (int i = 0; i < nKnots; ++i) {
    const typename Spline1D<DataT>::Knot& knot = spline.getKnot(i);
    double u = knot.u;
    DataT f[Ndim];
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
void SplineHelper1D<DataT>::approximateFunction(
  Spline1D<DataT>& spline, DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[/*spline.getFdimensions()*/])> F,
  int nAxiliaryDataPoints)
{
  /// Create best-fit spline parameters for a given input function F
  setSpline(spline, spline.getFdimensions(), nAxiliaryDataPoints);
  approximateFunction(spline.getFparameters(), xMin, xMax, F);
  spline.setXrange(xMin, xMax);
}

template <typename DataT>
void SplineHelper1D<DataT>::approximateFunctionGradually(
  Spline1D<DataT>& spline, DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[/*spline.getFdimensions()*/])> F,
  int nAxiliaryDataPoints)
{
  /// Create best-fit spline parameters gradually for a given input function F
  setSpline(spline, spline.getFdimensions(), nAxiliaryDataPoints);
  approximateFunctionGradually(spline.getFparameters(), xMin, xMax, F);
  spline.setXrange(xMin, xMax);
}

template <typename DataT>
void SplineHelper1D<DataT>::approximateFunction(
  DataT* Fparameters, DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[])> F) const
{
  /// Create best-fit spline parameters for a given input function F
  /// output in Fparameters
  std::vector<DataT> vF(getNumberOfDataPoints() * mFdimensions);
  double mUtoXscale = (((double)xMax) - xMin) / mSpline.getUmax();
  for (int i = 0; i < getNumberOfDataPoints(); i++) {
    F(xMin + mUtoXscale * mDataPoints[i].u, &vF[i * mFdimensions]);
  }
  approximateFunction(Fparameters, vF.data());
}

template <typename DataT>
void SplineHelper1D<DataT>::approximateFunctionGradually(
  DataT* Fparameters, DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[])> F) const
{
  /// Create best-fit spline parameters gradually for a given input function F
  /// output in Fparameters
  std::vector<DataT> vF(getNumberOfDataPoints() * mFdimensions);
  double mUtoXscale = (((double)xMax) - xMin) / mSpline.getUmax();
  for (int i = 0; i < getNumberOfDataPoints(); i++) {
    F(xMin + mUtoXscale * mDataPoints[i].u, &vF[i * mFdimensions]);
  }
  approximateFunctionGradually(Fparameters, vF.data());
}

template <typename DataT>
int SplineHelper1D<DataT>::setSpline(
  const Spline1D<DataT>& spline, int nFdimensions, int nAxiliaryDataPoints)
{
  // Prepare creation of a best-fit spline
  //
  // Data points will be set at all integer U (that includes all knots),
  // plus at nAxiliaryDataPoints points between the integers.
  //
  // nAxiliaryDataPoints must be >= 2
  //
  // nAxiliaryDataPoints==1 is also possible, but there must be at least
  // one integer U without a knot, in order to get 2*nKnots data points in total.
  //
  // The return value is an error index, 0 means no error

  int ret = 0;

  mSpline.cloneFromObject(spline, nullptr);
  mFdimensions = nFdimensions;
  int nPoints = 0;
  if (!spline.isConstructed()) {
    ret = storeError(-1, "SplineHelper1D<DataT>::setSpline: input spline is not constructed");
    mSpline.recreate(2, 0);
    nAxiliaryDataPoints = 2;
    nPoints = 4;
  } else {
    mSpline.cloneFromObject(spline, nullptr);
    nPoints = 1 + spline.getUmax() + spline.getUmax() * nAxiliaryDataPoints;
    if (nPoints < 2 * spline.getNumberOfKnots()) {
      nAxiliaryDataPoints = 2;
      nPoints = 1 + spline.getUmax() + spline.getUmax() * nAxiliaryDataPoints;
      ret = storeError(-3, "SplineHelper1D::setSpline: too few nAxiliaryDataPoints, increase to 2");
    }
  }

  const int nPar = mSpline.getNumberOfParameters(1); // n parameters for 1D

  mDataPoints.resize(nPoints);
  double scalePoints2Knots = ((double)spline.getUmax()) / (nPoints - 1.);

  for (int i = 0; i < nPoints; ++i) {
    DataPoint& p = mDataPoints[i];
    double u = i * scalePoints2Knots;
    int iKnot = spline.getKnotIndexU(u);
    const typename Spline1D<DataT>::Knot& knot0 = spline.getKnot(iKnot);
    const typename Spline1D<DataT>::Knot& knot1 = spline.getKnot(iKnot + 1);
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
    const typename Spline1D<DataT>::Knot& knot = spline.getKnot(i);
    int iu = (int)(knot.u + 0.1f);
    mKnotDataPoints[i] = iu * (1 + nAxiliaryDataPoints);
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
      ret = storeError(-4, "SplineHelper1D::setSpline: internal error - can not invert the matrix");
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
      ret = storeError(-5, "SplineHelper1D::setSpline: internal error - can not invert the matrix");
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
void SplineHelper1D<DataT>::approximateFunction(
  DataT* Fparameters,
  const DataT DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// Approximate a function given as an array of values at data points

  const int nPar = mSpline.getNumberOfParameters(1);
  double b[nPar];
  for (int idim = 0; idim < mFdimensions; idim++) {
    for (int i = 0; i < nPar; i++)
      b[i] = 0.;

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
      Fparameters[i * mFdimensions + idim] = (float)s;
    }
  }
}

template <typename DataT>
void SplineHelper1D<DataT>::approximateFunctionGradually(
  DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim */]) const
{
  /// gradually approximate a function given as an array of values at data points
  /// output in Fparameters
  copySfromDataPoints(Fparameters, DataPointF);
  approximateDerivatives(Fparameters, DataPointF);
}

template <typename DataT>
void SplineHelper1D<DataT>::copySfromDataPoints(
  DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// a tool for the gradual approximation: set spline values S_i at knots == function values
  /// output in Fparameters
  for (int i = 0; i < mSpline.getNumberOfKnots(); ++i) { // set F values at knots
    int ip = mKnotDataPoints[i];
    for (int d = 0; d < mFdimensions; d++) {
      Fparameters[2 * i * mFdimensions + d] = DataPointF[ip * mFdimensions + d];
    }
  }
}

template <typename DataT>
void SplineHelper1D<DataT>::approximateDerivatives(
  DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const
{
  /// a tool for the gradual approximation:
  /// calibrate spline derivatives D_i using already calibrated spline values S_i
  /// input and output output in Fparameters

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
    for (int d = 0; d < Ndim; d++)
      s[d] = 0.;
    for (int j = 0; j < nKnots; ++j) {
      for (int d = 0; d < Ndim; d++)
        s[d] += row[j] * Fparameters[2 * j * Ndim + d];
    }
    for (int d = 0; d < Ndim; d++)
      b[i * Ndim + d] -= s[d];
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
      Fparameters[(2 * i + 1) * Ndim + d] = (float)s[d];
    }
  }
}

template class GPUCA_NAMESPACE::gpu::SplineHelper1D<float>;
template class GPUCA_NAMESPACE::gpu::SplineHelper1D<double>;

#endif
