// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  SplineHelper2D.h
/// \brief Definition of SplineHelper2D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEHELPER2D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEHELPER2D_H

#include <cmath>
#include <vector>

#include "GPUCommonDef.h"
#include "Rtypes.h"
#include "TString.h"
#include "Spline1D.h"
#include "Spline2D.h"
#include "SplineHelper1D.h"
#include <functional>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The SplineHelper2D class is to initialize Spline* objects
///

class SplineHelper2D
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  SplineHelper2D();

  /// Copy constructor: disabled
  SplineHelper2D(const SplineHelper2D&) CON_DELETE;

  /// Assignment operator: disabled
  SplineHelper2D& operator=(const SplineHelper2D&) CON_DELETE;

  /// Destructor
  ~SplineHelper2D() CON_DEFAULT;

  /// _______________  Main functionality  ________________________

  int setSpline(const Spline2D& spline, int nAxiliaryPointsU, int nAxiliaryPointsV);

  /// Creates compact spline parameters for a given input function
  std::unique_ptr<float[]> constructParameters(int Ndim, std::function<void(float u, float v, float f[/*Ndim*/])> F, float uMin, float uMax, float vMin, float vMax);

  /// _______________   Interface for a manual construction of compact splines   ________________________

  void constructParameters(int Ndim, const float DataPointF[/*getNumberOfDataPoints() x Ndim*/], float parameters[/*mSpline.getNumberOfParameters(Ndim)*/]) const;

  int getNumberOfDataPointsU() const { return mHelperU.getNumberOfDataPoints(); }
  int getNumberOfDataPointsV() const { return mHelperV.getNumberOfDataPoints(); }
  int getNumberOfDataPoints() const { return getNumberOfDataPointsU() * getNumberOfDataPointsV(); }

  const SplineHelper1D& getHelperU() const { return mHelperU; }
  const SplineHelper1D& getHelperV() const { return mHelperV; }

  /// _______________  Utilities   ________________________

  ///  Gives error string
  const char* getLastError() const { return mError.Data(); }

 private:
  /// Stores an error message
  int storeError(Int_t code, const char* msg);

  TString mError = ""; ///< error string

  Spline2D mSpline;
  SplineHelper1D mHelperU;
  SplineHelper1D mHelperV;
};

inline int SplineHelper2D::storeError(int code, const char* msg)
{
  mError = msg;
  return code;
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
