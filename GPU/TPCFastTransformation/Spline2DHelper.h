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

/// \file  Spline2DHelper.h
/// \brief Definition of Spline2DHelper class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_Spline2DHelper_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_Spline2DHelper_H

#include <cmath>
#include <vector>

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "Spline1D.h"
#include "Spline2D.h"
#include "Spline1DHelperOld.h"
#include <functional>
#include <string>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The Spline2DHelper class is a helper to initialize Spline* objects
///
template <typename DataT>
class Spline2DHelper
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  Spline2DHelper();

  /// Copy constructor: disabled
  Spline2DHelper(const Spline2DHelper&) CON_DELETE;

  /// Assignment operator: disabled
  Spline2DHelper& operator=(const Spline2DHelper&) CON_DELETE;

  /// Destructor
  ~Spline2DHelper() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  /// Create best-fit spline parameters for a given input function F
  void approximateFunction(
    Spline2DContainer<DataT>& spline,
    double x1Min, double x1Max, double x2Min, double x2Max,
    std::function<void(double x1, double x2, double f[/*spline.getYdimensions()*/])> F,
    int nAuxiliaryDataPointsU1 = 4, int nAuxiliaryDataPointsU2 = 4);

  // A wrapper around approximateDataPoints()
  void approximateFunctionViaDataPoints(
    Spline2DContainer<DataT>& spline,
    double x1Min, double x1Max, double x2Min, double x2Max,
    std::function<void(double x1, double x2, double f[/*spline.getYdimensions()*/])> F,
    int nAuxiliaryDataPointsU1 = 4, int nAuxiliaryDataPointsU2 = 4);

  /// Create best-fit spline parameters for a given set of data points
  void approximateDataPoints(
    Spline2DContainer<DataT>& spline, DataT* splineParameters, double x1Min, double x1Max, double x2Min, double x2Max,
    const double dataPointX1[/*nDataPoints*/], const double dataPointX2[/*nDataPoints*/],
    const double dataPointF[/*nDataPoints x spline.getYdimensions*/], int nDataPoints);

  /// _______________   Interface for a step-wise construction of the best-fit spline   ________________________

  /// precompute everything needed for the construction
  int setSpline(const Spline2DContainer<DataT>& spline, int nAuxiliaryPointsU1, int nAuxiliaryPointsU2);

  /// approximate std::function, output in Fparameters
  void approximateFunction(
    DataT* Fparameters, double x1Min, double x1Max, double x2Min, double x2Max,
    std::function<void(double x1, double x2, double f[/*mFdimensions*/])> F) const;

  /// approximate std::function, output in Fparameters. F calculates values for a batch of points.
  void approximateFunctionBatch(
    DataT* Fparameters, double x1Min, double x1Max, double x2Min, double x2Max,
    std::function<void(const std::vector<double>& x1, const std::vector<double>& x2, std::vector<double> f[/*mFdimensions*/])> F,
    unsigned int batchsize) const;

  /// approximate a function given as an array of values at data points
  void approximateFunction(
    DataT* Fparameters, const double DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  int getNumberOfDataPointsU1() const { return mHelperU1.getNumberOfDataPoints(); }

  int getNumberOfDataPointsU2() const { return mHelperU2.getNumberOfDataPoints(); }

  int getNumberOfDataPoints() const { return getNumberOfDataPointsU1() * getNumberOfDataPointsU2(); }

  const Spline1DHelperOld<DataT>& getHelperU1() const { return mHelperU1; }
  const Spline1DHelperOld<DataT>& getHelperU2() const { return mHelperU2; }

  /// _______________  Utilities   ________________________

  ///  Gives error string
  const char* getLastError() const { return mError.c_str(); }

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB) // code invisible on GPU and in the standalone compilation
  /// Test the Spline2D class functionality
  static int test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

 private:
  void setGrid(Spline2DContainer<DataT>& spline, double x1Min, double x1Max, double x2Min, double x2Max);
  void getScoefficients(int iu, int iv, double u, double v,
                        double c[16], int indices[16]);

  /// Stores an error message
  int storeError(int code, const char* msg);

  std::string mError = ""; ///< error string
  int mFdimensions;        ///< n of F dimensions
  Spline1DHelperOld<DataT> mHelperU1;
  Spline1DHelperOld<DataT> mHelperU2;
  Spline1D<double, 0> fGridU;
  Spline1D<double, 0> fGridV;

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline2DHelper, 0);
#endif
};

template <typename DataT>
void Spline2DHelper<DataT>::approximateFunction(
  Spline2DContainer<DataT>& spline,
  double x1Min, double x1Max, double x2Min, double x2Max,
  std::function<void(double x1, double x2, double f[/*spline.getYdimensions()*/])> F,
  int nAuxiliaryDataPointsU1, int nAuxiliaryDataPointsU2)
{
  /// Create best-fit spline parameters for a given input function F
  setSpline(spline, nAuxiliaryDataPointsU1, nAuxiliaryDataPointsU2);
  approximateFunction(spline.getParameters(), x1Min, x1Max, x2Min, x2Max, F);
  spline.setXrange(x1Min, x1Max, x2Min, x2Max);
}

template <typename DataT>
int Spline2DHelper<DataT>::setSpline(
  const Spline2DContainer<DataT>& spline, int nAuxiliaryPointsU, int nAuxiliaryPointsV)
{
  // Prepare creation of 2D irregular spline
  // The should be at least one (better, two) Auxiliary measurements on each segnment between two knots and at least 2*nKnots measurements in total
  // Returns 0 when the spline can not be constructed with the given nAuxiliaryPoints

  int ret = 0;
  mFdimensions = spline.getYdimensions();
  if (mHelperU1.setSpline(spline.getGridX1(), mFdimensions, nAuxiliaryPointsU) != 0) {
    ret = storeError(-2, "Spline2DHelper::setSpline2D: error by setting U axis");
  }
  if (mHelperU2.setSpline(spline.getGridX2(), mFdimensions, nAuxiliaryPointsV) != 0) {
    ret = storeError(-3, "Spline2DHelper::setSpline2D: error by setting V axis");
  }
  return ret;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
