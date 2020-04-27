// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  SplineHelper1D.h
/// \brief Definition of SplineHelper1D class

/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEHELPER1D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEHELPER1D_H

#include <cmath>
#include <vector>

#include "GPUCommonDef.h"
#include "Rtypes.h"
#include "TString.h"
#include "Spline1D.h"
#include <functional>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The SplineHelper1D class is to initialize parameters for Spline1D class
///
template <typename DataT>
class SplineHelper1D
{
 public:
  ///
  /// \brief Helper structure for 1D spline construction
  ///
  struct DataPoint {
    double u;    ///< u coordinate
    double cS0;  ///< a coefficient for s0
    double cZ0;  ///< a coefficient for s'0
    double cS1;  ///< a coefficient for s1
    double cZ1;  ///< a coefficient for s'1
    int iKnot;   ///< index of the left knot of the segment
    bool isKnot; ///< is the point placed at a knot
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  SplineHelper1D();

  /// Copy constructor: disabled
  SplineHelper1D(const SplineHelper1D&) CON_DEFAULT;

  /// Assignment operator: disabled
  SplineHelper1D& operator=(const SplineHelper1D&) CON_DEFAULT;

  /// Destructor
  ~SplineHelper1D() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  /// Create best-fit spline parameters for a given input function F
  void approximateFunction(Spline1D<DataT>& spline,
                           DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[/*spline.getFdimensions()*/])> F,
                           int nAxiliaryDataPoints = 4);

  /// Create best-fit spline parameters gradually for a given input function F
  void approximateFunctionGradually(Spline1D<DataT>& spline,
                                    DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[/*spline.getFdimensions()*/])> F,
                                    int nAxiliaryDataPoints = 4);

  /// Create classic spline parameters for a given input function F
  void approximateFunctionClassic(Spline1D<DataT>& spline,
                                  DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[/*spline.getFdimensions()*/])> F);

  /// _______________   Interface for a step-wise construction of the best-fit spline   ________________________

  /// precompute everything needed for the construction
  int setSpline(const Spline1D<DataT>& spline, int nFdimensions, int nAxiliaryDataPoints);

  /// approximate std::function, output in Fparameters
  void approximateFunction(DataT* Fparameters, DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[])> F) const;

  /// approximate std::function gradually, output in Fparameters
  void approximateFunctionGradually(DataT* Fparameters, DataT xMin, DataT xMax, std::function<void(DataT x, DataT f[])> F) const;

  /// number of data points
  int getNumberOfDataPoints() const { return mDataPoints.size(); }

  /// approximate a function given as an array of values at data points
  void approximateFunction(DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  /// gradually approximate a function given as an array of values at data points
  void approximateFunctionGradually(DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim */]) const;

  /// a tool for the gradual approximation: set spline values S_i at knots == function values
  void copySfromDataPoints(DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  /// a tool for the gradual approximation:
  /// calibrate spline derivatives D_i using already calibrated spline values S_i
  void approximateDerivatives(DataT* Fparameters, const DataT DataPointF[/*getNumberOfDataPoints() x nFdim*/]) const;

  /// _______________  Utilities   ________________________

  const Spline1D<DataT>& getSpline() const { return mSpline; }

  int getKnotDataPoint(int iknot) const { return mKnotDataPoints[iknot]; }

  const DataPoint& getDataPoint(int ip) const { return mDataPoints[ip]; }

  ///  Gives error string
  const char* getLastError() const { return mError.Data(); }

 private:
  /// Stores an error message
  int storeError(Int_t code, const char* msg);

  TString mError = ""; ///< error string

  /// helpers for the construction of 1D spline

  Spline1D<DataT> mSpline;            ///< copy of the spline
  int mFdimensions;                   ///< n of F dimensions
  std::vector<DataPoint> mDataPoints; ///< measurement points
  std::vector<int> mKnotDataPoints;   ///< which measurement points are at knots
  std::vector<double> mLSMmatrixFull; ///< a matrix to convert the measurements into the spline parameters with the LSM method
  std::vector<double> mLSMmatrixSderivatives;
  std::vector<double> mLSMmatrixSvalues;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
