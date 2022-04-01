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

/// \file  SplineHelper.h
/// \brief Definition of SplineHelper class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEHELPER_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEHELPER_H

#include <cmath>
#include <vector>

#include "GPUCommonDef.h"
#include "Rtypes.h"
#include "TString.h"
#include "Spline1D.h"
#include "Spline.h"
#include "Spline1DHelperOld.h"
#include <functional>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The SplineHelper class is to initialize Spline* objects
///
template <typename DataT>
class SplineHelper
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  SplineHelper();

  /// Copy constructor: disabled
  SplineHelper(const SplineHelper&) CON_DELETE;

  /// Assignment operator: disabled
  SplineHelper& operator=(const SplineHelper&) CON_DELETE;

  /// Destructor
  ~SplineHelper() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  /// Create best-fit spline parameters for a given input function F
  void approximateFunction(SplineContainer<DataT>& spline,
                           const double xMin[/* Xdim */], const double xMax[/* Xdim */],
                           std::function<void(const double x[/* Xdim */], double f[/* Fdim */])> F,
                           const int nAxiliaryDataPoints[/* Xdim */] = nullptr);

  /// _______________   Interface for a step-wise construction of the best-fit spline   ________________________

  /// precompute everything needed for the construction
  int setSpline(const SplineContainer<DataT>& spline, const int nAxiliaryPoints[/* Xdim */]);

  /// approximate std::function, output in Fparameters
  void approximateFunction(
    DataT* Fparameters, const double xMin[/* mXdimensions */], const double xMax[/* mXdimensions */],
    std::function<void(const double x[/* mXdimensions */], double f[/* mFdimensions */])> F) const;

  /// approximate std::function, output in Fparameters. F calculates values for a batch of points.
  void approximateFunctionBatch(
    DataT* Fparameters, const double xMin[/* mXdimensions */], const double xMax[/* mXdimensions */],
    std::function<void(const std::vector<double> x[/* mXdimensions */], double f[/*mFdimensions*/])> F,
    unsigned int batchsize) const;

  /// approximate a function given as an array of values at data points
  void approximateFunction(
    DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  int getNumberOfDataPoints(int dimX) const { return mHelpers[dimX].getNumberOfDataPoints(); }

  int getNumberOfDataPoints() const { return mNumberOfDataPoints; }

  const Spline1DHelperOld<DataT>& getHelper(int dimX) const { return mHelpers[dimX]; }

  /// _______________  Utilities   ________________________

  ///  Gives error string
  const char* getLastError() const { return mError.Data(); }

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// Test the Spline class functionality
  static int test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

  static int arraytopoints(int point, int result[], const int numbers[], int dim);

  static int pointstoarray(const int indices[], const int numbers[], int dim);

 private:
  /// Stores an error message
  int storeError(Int_t code, const char* msg);

  TString mError = "";     ///< error string
  int mXdimensions;        ///< number of X dimensions
  int mFdimensions;        ///< number of F dimensions
  int mNumberOfParameters; ///< number of parameters
  int mNumberOfDataPoints; ///< number of data points
  std::vector<Spline1DHelperOld<DataT>> mHelpers;
};

template <typename DataT>
void SplineHelper<DataT>::approximateFunction(
  SplineContainer<DataT>& spline,
  const double xMin[/* Xdim */], const double xMax[/* Xdim */],
  std::function<void(const double x[/* Xdim */], double f[/* Fdim */])> F,
  const int nAxiliaryDataPoints[/* Xdim */])
{
  /// Create best-fit spline parameters for a given input function F
  setSpline(spline, nAxiliaryDataPoints);
  approximateFunction(spline.getParameters(), xMin, xMax, F);
  DataT xxMin[spline.getXdimensions()];
  DataT xxMax[spline.getXdimensions()];
  for (int i = 0; i < spline.getXdimensions(); i++) {
    xxMin[i] = xMin[i];
    xxMax[i] = xMax[i];
  }
  spline.setXrange(xxMin, xxMax);
}

template <typename DataT>
int SplineHelper<DataT>::setSpline(
  const SplineContainer<DataT>& spline, const int nAxiliaryPoints[/* Xdim */])
{
  // Prepare creation of an irregular spline
  // The should be at least one (better, two) axiliary measurements on each segnment between two knots and at least 2*nKnots measurements in total
  // Returns 0 when the spline can not be constructed with the given nAxiliaryPoints

  int ret = 0;
  mXdimensions = spline.getXdimensions();
  mFdimensions = spline.getYdimensions();
  mNumberOfParameters = spline.getNumberOfParameters();
  mNumberOfDataPoints = 1;
  mHelpers.clear();
  mHelpers.resize(mXdimensions);
  for (int i = 0; i < mXdimensions; i++) {
    int np = (nAxiliaryPoints != nullptr) ? nAxiliaryPoints[i] : 4;
    if (mHelpers[i].setSpline(spline.getGrid(i), mFdimensions, np) != 0) {
      ret = storeError(-2, "SplineHelper::setSpline: error by setting an axis");
    }
    mNumberOfDataPoints *= mHelpers[i].getNumberOfDataPoints();
  }

  return ret;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
