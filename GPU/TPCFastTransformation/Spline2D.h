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

/// \file  Spline2D.h
/// \brief Definition of Spline2D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE2D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE2D_H

#include "Spline1D.h"
#include "Spline2DSpec.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"

#if !defined(__CINT__) && !defined(__ROOTCINT__) && !defined(__ROOTCLING__) && !defined(GPUCA_GPUCODE) && !defined(GPUCA_NO_VC) && defined(__cplusplus) && __cplusplus >= 201703L
#include <Vc/Vc>
#include <Vc/SimdArray>
#endif

class TFile;

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The Spline2D class performs a cubic spline interpolation on an two-dimensional nonunifom grid.
/// The class is an extension of the Spline1D class.
/// See Spline1D.h for more details.
///
/// The spline S(x1,x2) approximates a function F(x1,x2):R^2->R^m,
/// with 2-dimensional domain and multi-dimensional codomain.
/// x1,x2 belong to [x1min,x1max] x [x2min,x2max].
///
/// --- Example of creating a spline ---
///
///  auto F = [&](double x1, double x2, double f[] ) {
///   f[0] = 1.f + x1 + x2*x2; // F(x1,x2)
///  };
///  const int nKnotsU=2;
///  const int nKnotsV=3;
///  int knotsU[nKnotsU] = {0, 1};
///  int knotsV[nKnotsV] = {0, 2, 5};
///  Spline2D<float,1> spline(nKnotsU, knotsU, nKnotsV, knotsV ); // spline with 1-dimensional codomain
///  spline.approximateFunction(0., 1., 0.,1., F); //initialize spline to approximate F on [0., 1.]x[0., 1.] area
///  float S = spline.interpolate(.1, .3 ); // interpolated value at (.1,.3)
///
///  --- See also Spline2DHelper::test();
///

/// ==================================================================================================
///
/// Declare the Spline1D class as a template with one optional parameters.
///
/// Class specializations depend on the XdimT, YdimT values. They can be found in SplineSpecs.h
///
/// \param DataT data type: float or double
/// \param YdimT
///    YdimT > 0 : the number of Y dimensions is known at the compile time and is equal to XdimT
///    YdimT = 0 : the number of Y dimensions will be set in the runtime
///    YdimT < 0 : the number of Y dimensions will be set in the runtime, and it will not exceed abs(YdimT)
///
template <typename DataT, int YdimT = 0>
class Spline2D
  : public Spline2DSpec<DataT, YdimT, SplineUtil::getSpec(YdimT)>
{
  typedef Spline2DContainer<DataT> TVeryBase;
  typedef Spline2DSpec<DataT, YdimT, SplineUtil::getSpec(YdimT)> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;
  typedef typename TVeryBase::Knot Knot;

#if !defined(GPUCA_GPUCODE)
  using TBase::TBase; // inherit constructors

  /// Assignment operator
  Spline2D& operator=(const Spline2D& v)
  {
    TVeryBase::cloneFromObject(v, nullptr);
    return *this;
  }
#else
  /// Disable constructors for the GPU implementation
  Spline2D() CON_DELETE;
  Spline2D(const Spline2D&) CON_DELETE;
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)
  /// read a class object from the file
  static Spline2D* readFromFile(TFile& inpf, const char* name)
  {
    return (Spline2D*)TVeryBase::readFromFile(inpf, name);
  }
#endif

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline2D, 0);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
