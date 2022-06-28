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

/// \file  Spline1DHelper.h
/// \brief Definition of Spline1DHelper class

/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_Spline1DHelper_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_Spline1DHelper_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "Spline1D.h"

#include <vector>
#include <string>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The Spline1DHelper class is to initialize parameters for Spline1D class
///
template <typename DataT>
class Spline1DHelper
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  Spline1DHelper();

  /// Copy constructor: disabled
  Spline1DHelper(const Spline1DHelper&) CON_DEFAULT;

  /// Assignment operator: disabled
  Spline1DHelper& operator=(const Spline1DHelper&) CON_DEFAULT;

  /// Destructor
  ~Spline1DHelper() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  /// Create best-fit spline parameters for a set of data points
  void approximateDataPoints(Spline1DContainer<DataT>& spline,
                             double xMin, double xMax,
                             const double vx[], const double vf[], int nDataPoints);

  /// Create best-fit spline parameters for a function F
  void approximateFunction(
    Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
    int nAuxiliaryDataPoints = 4);

  /// Approximate only derivatives assuming the spline values at knozts are already set
  void approximateDerivatives(Spline1DContainer<DataT>& spline,
                              const double vx[], const double vf[], int nDataPoints);

  void approximateFunctionGradually(
    Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
    int nAuxiliaryDataPoints);

  /// Create classic spline parameters for a given input function F
  void approximateFunctionClassic(Spline1DContainer<DataT>& spline,
                                  double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F);

  /// _______________  Utilities   ________________________

  const Spline1D<double>& getSpline() const { return mSpline; }

  /// Get derivatives of the interpolated value {S(u): 1D -> nYdim} at the segment [knotL, next knotR]
  /// over the spline values Sl, Sr and the slopes Dl, Dr
  static void getScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                               double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                                double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDDScoefficients(const typename Spline1D<double>::Knot& knotL, double u,
                                 double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDDScoefficientsLeft(const typename Spline1D<double>::Knot& knotL,
                                     double& cSl, double& cDl, double& cSr, double& cDr);

  static void getDDScoefficientsRight(const typename Spline1D<double>::Knot& knotL,
                                      double& cSl, double& cDl, double& cSr, double& cDr);
  static void getDDDScoefficients(const typename Spline1D<double>::Knot& knotL,
                                  double& cSl, double& cDl, double& cSr, double& cDr);

  ///  Gives error string
  const char* getLastError() const { return mError.c_str(); }

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB) // code invisible on GPU and in the standalone compilation
  /// Test the Spline1D class functionality
  static int test(const bool draw = 0, const bool drawDataPoints = 1);
#endif

 private:
  /// Stores an error message
  int storeError(int code, const char* msg);

  std::string mError = ""; ///< error string

  void setSpline(const Spline1DContainer<DataT>& spline);

  void makeDataPoints(Spline1DContainer<DataT>& spline, double xMin, double xMax, std::function<void(double x, double f[/*spline.getFdimensions()*/])> F,
                      int nAuxiliaryDataPoints, std::vector<double>& vx, std::vector<double>& vf);

  /// helpers for the construction of 1D spline

  Spline1D<double> mSpline; ///< copy of the spline grid

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline1DHelper, 0);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
