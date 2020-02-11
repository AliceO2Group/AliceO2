// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  Spline182.cxx
/// \brief Implementation of SplineHelper2D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

#include "SplineHelper2D.h"
#include "TMath.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompBK.h"

using namespace GPUCA_NAMESPACE::gpu;

SplineHelper2D::SplineHelper2D() : mError(), mSpline()
{
  mSpline.constructKnotsRegular(2, 2);
}

int SplineHelper2D::setSpline(const Spline2D& spline, int nAxiliaryPointsU, int nAxiliaryPointsV)
{
  // Prepare creation of 2D irregular spline
  // The should be at least one (better, two) axiliary measurements on each segnment between two knots and at least 2*nKnots measurements in total
  // Returns 0 when the spline can not be constructed with the given nAxiliaryPoints

  int ret = 0;

  if (!spline.isConstructed()) {
    ret = storeError(-1, "SplineHelper2D::setSpline2D: input spline is not constructed");
    mSpline.constructKnotsRegular(2, 2);
  } else {
    mSpline.cloneFromObject(spline, nullptr);
  }
  if (mHelperU.setSpline(mSpline.getGridU(), nAxiliaryPointsU) != 0) {
    ret = storeError(-2, "SplineHelper2D::setSpline2D: error by setting U axis");
  }

  if (mHelperV.setSpline(mSpline.getGridV(), nAxiliaryPointsV) != 0) {
    ret = storeError(-3, "SplineHelper2D::setSpline2D: error by setting V axis");
  }

  return ret;
}

void SplineHelper2D::constructParameters(int Ndim, const float DataPointF[/*getNumberOfDataPoints() x Ndim*/], float parameters[/*mSpline.getNumberOfParameters(Ndim) */]) const
{
  // Create 2D irregular spline in a compact way

  const int Ndim2 = 2 * Ndim;
  const int Ndim3 = 3 * Ndim;
  const int Ndim4 = 4 * Ndim;

  int nDataPointsU = getNumberOfDataPointsU();
  int nDataPointsV = getNumberOfDataPointsV();

  int nKnotsU = mSpline.getGridU().getNumberOfKnots();
  int nKnotsV = mSpline.getGridV().getNumberOfKnots();

  std::unique_ptr<float[]> rotDataPointF(new float[nDataPointsU * nDataPointsV * Ndim]); // U DataPoints x V DataPoints :  rotated DataPointF for one output dimension
  std::unique_ptr<float[]> Dv(new float[nKnotsV * nDataPointsU * Ndim]);                 // V knots x U DataPoints

  std::unique_ptr<float[]> parU(new float[mHelperU.getSpline().getNumberOfParameters(Ndim)]);
  std::unique_ptr<float[]> parV(new float[mHelperV.getSpline().getNumberOfParameters(Ndim)]);

  // rotated data points (u,v)->(v,u)

  for (int ipu = 0; ipu < nDataPointsU; ipu++) {
    for (int ipv = 0; ipv < nDataPointsV; ipv++) {
      for (int dim = 0; dim < Ndim; dim++) {
        rotDataPointF[Ndim * (ipu * nDataPointsV + ipv) + dim] = DataPointF[Ndim * (ipv * nDataPointsU + ipu) + dim];
      }
    }
  }

  // get S and S'u at all the knots by interpolating along the U axis

  for (int iKnotV = 0; iKnotV < nKnotsV; ++iKnotV) {
    int ipv = mHelperV.getKnotDataPoint(iKnotV);
    const float* DataPointFrow = &(DataPointF[Ndim * ipv * nDataPointsU]);
    mHelperU.constructParametersGradually(Ndim, DataPointFrow, parU.get());

    for (int iKnotU = 0; iKnotU < nKnotsU; ++iKnotU) {
      float* knotPar = &parameters[Ndim4 * (iKnotV * nKnotsU + iKnotU)];
      for (int dim = 0; dim < Ndim; ++dim) {
        knotPar[dim] = parU[Ndim * (2 * iKnotU) + dim];                // store S for all the knots
        knotPar[Ndim2 + dim] = parU[Ndim * (2 * iKnotU) + Ndim + dim]; // store S'u for all the knots //SG!!!
      }
    }

    // recalculate F values for all ipu DataPoints at V = ipv
    for (int ipu = 0; ipu < nDataPointsU; ipu++) {
      float splineF[Ndim];
      float u = mHelperU.getDataPoint(ipu).u;
      mSpline.getGridU().interpolate(Ndim, parU.get(), u, splineF);
      for (int dim = 0; dim < Ndim; dim++) {
        rotDataPointF[(ipu * nDataPointsV + ipv) * Ndim + dim] = splineF[dim];
      }
    }
  }

  // calculate S'v at all data points with V == V of a knot

  for (int ipu = 0; ipu < nDataPointsU; ipu++) {
    const float* DataPointFcol = &(rotDataPointF[ipu * nDataPointsV * Ndim]);
    mHelperV.constructParametersGradually(Ndim, DataPointFcol, parV.get());
    for (int iKnotV = 0; iKnotV < nKnotsV; iKnotV++) {
      for (int dim = 0; dim < Ndim; dim++) {
        float dv = parV[(iKnotV * 2 + 1) * Ndim + dim];
        Dv[(iKnotV * nDataPointsU + ipu) * Ndim + dim] = dv;
      }
    }
  }

  // fit S'v and S''_vu at all the knots

  for (int iKnotV = 0; iKnotV < nKnotsV; ++iKnotV) {
    const float* Dvrow = &(Dv[iKnotV * nDataPointsU * Ndim]);
    mHelperU.constructParameters(Ndim, Dvrow, parU.get());
    for (int iKnotU = 0; iKnotU < nKnotsU; ++iKnotU) {
      for (int dim = 0; dim < Ndim; ++dim) {
        parameters[Ndim4 * (iKnotV * nKnotsU + iKnotU) + Ndim + dim] = parU[Ndim * 2 * iKnotU + dim];         // store S'v for all the knots
        parameters[Ndim4 * (iKnotV * nKnotsU + iKnotU) + Ndim3 + dim] = parU[Ndim * 2 * iKnotU + Ndim + dim]; // store S''vu for all the knots
      }
    }
  }
}

std::unique_ptr<float[]> SplineHelper2D::constructParameters(int Ndim, std::function<void(float u, float v, float f[/*Ndim*/])> F, float uMin, float uMax, float vMin, float vMax)
{
  /// Creates compact spline parameters for a given input function

  std::vector<float> dataPointF(getNumberOfDataPoints() * Ndim);

  double scaleU = (uMax - uMin) / ((double)mSpline.getGridU().getUmax());
  double scaleV = (vMax - vMin) / ((double)mSpline.getGridV().getUmax());

  for (int iu = 0; iu < getNumberOfDataPointsU(); iu++) {
    float u = uMin + mHelperU.getDataPoint(iu).u * scaleU;
    for (int iv = 0; iv < getNumberOfDataPointsV(); iv++) {
      float v = vMin + mHelperV.getDataPoint(iv).u * scaleV;
      F(u, v, &dataPointF[(iv * getNumberOfDataPointsU() + iu) * Ndim]);
    }
  }
  std::unique_ptr<float[]> parameters(new float[mSpline.getNumberOfParameters(Ndim)]);
  constructParameters(Ndim, dataPointF.data(), parameters.get());
  return parameters;
}

#endif
