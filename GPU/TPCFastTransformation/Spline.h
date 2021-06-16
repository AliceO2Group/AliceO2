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

/// \file  Spline.h
/// \brief Definition of Spline class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE_H

#include "SplineSpec.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The Spline class performs a cubic spline interpolation on an two-dimensional nonunifom grid.
/// The class is an extension of the Spline1D class.
/// See Spline1D.h for more details.
///
/// The spline S(x) approximates a function F(x):R^n->R^m,
/// with multi-dimensional domain and multi-dimensional codomain.
/// x belongs to [xmin,xmax].
///
/// --- Example of creating a spline ---
///
///  constexpr int nDimX = 2, nDimY = 1;
///  int nKnots[nDimX] = {2, 3}; //  2 x 3 knots
///  int knotsU1[] = {0, 1};     // relative knot positions
///  int knotsU2[] = {0, 2, 5};
///  int *knotsU[nDimX] = {knotsU1, knotsU2};
///
///  o2::gpu::Spline<float, nDimX, nDimY> spline(nKnots, knotsU);
///
///  auto F = [&](const double x[], double f[]) {
///    f[0] = 1.f + x[0] + x[1] * x[1]; // F(x)
///  };
///  double xMin[nDimX] = {0.f, 0.f};
///  double xMax[nDimX] = {1.f, 1.f};
///  spline.approximateFunction( xMin, xMax, F); // initialize spline to approximate F on [0., 1.]x[0., 1.] area
///
///  float x[] = {.1, .3};
///  float S = spline.interpolate(x); // interpolated value at (.1,.3)
///
///  -- another way to create of the spline is:
///
///  o2::gpu::Spline<float> spline(nDimX, nDimY, nKnots, knotsU );
///  spline.interpolate(x, &S);
///
///  --- See also SplineHelper::test();
///

/// ==================================================================================================
///
/// Declare the Spline class as a template with two optional parameters.
///
/// Class specializations depend on the XdimT, YdimT values. They can be found in SplineSpecs.h
///
/// \param DataT data type: float or double
/// \param XdimT
///    XdimT > 0 : the number of X dimensions is known at the compile time and is equal to XdimT
///    XdimT = 0 : the number of X dimensions will be set in the runtime
///    XdimT < 0 : the number of X dimensions will be set in the runtime, and it will not exceed abs(XdimT)
/// \param YdimT same for the Y dimensions
///
template <typename DataT, int XdimT = 0, int YdimT = 0>
class Spline
  : public SplineSpec<DataT, XdimT, YdimT, SplineUtil::getSpec(XdimT, YdimT)>
{
  typedef SplineContainer<DataT> TVeryBase;
  typedef SplineSpec<DataT, XdimT, YdimT, SplineUtil::getSpec(XdimT, YdimT)> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;
  typedef typename TVeryBase::Knot Knot;

#if !defined(GPUCA_GPUCODE)
  using TBase::TBase; // inherit constructors

  /// Assignment operator
  Spline& operator=(const Spline& v)
  {
    TVeryBase::cloneFromObject(v, nullptr);
    return *this;
  }
#else
  /// Disable constructors for the GPU implementation
  Spline() CON_DELETE;
  Spline(const Spline&) CON_DELETE;
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)
  /// read a class object from the file
  static Spline* readFromFile(TFile& inpf, const char* name)
  {
    return (Spline*)TVeryBase::readFromFile(inpf, name);
  }
#endif

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline, 0);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
