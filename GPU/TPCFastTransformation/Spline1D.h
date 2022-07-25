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

/// \file  Spline1D.h
/// \brief Definition of Spline1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE1D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINE1D_H

#include "Spline1DSpec.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/// The Spline1D class performs a cubic spline interpolation on a one-dimensional non-uniform grid.
///
/// The class is a flat C structure. It inherits from the FlatObject.
/// No virtual methods, no ROOT types are used.
///
/// --- Interpolation ---
///
/// The spline S(x) approximates a function F(x):[Xmin,Xmax]->Y
/// X is one-dimensional, Y may be multi-dimensional.
///
/// The spline has n knots x_i. The spline value S(x_i) and its derivative S'(x_i) are stored for every knot.
/// Inbetween the knots, S(x) is evaluated via interpolation by 3-rd degree polynomials.
///
/// The spline S(x) and its first derivative are continuous.
/// Depending on the initialization of the derivatives S'(x_i),
/// the second derivative may or may not be continuous at the knots.
///
/// --- Knots ---
///
/// The knots are not entirely irregular.
/// There is an internal scaled coordinate, called U, where all N knots have some integer positions:
/// {U0==0, U1, .., Un-1==Umax}.
/// It is implemented this way for fast matching of any X value to its neighboring knots.
///
/// For example, three knots with U coordinates u_i={0, 3, 5},
/// being stretched on the X segment [0., 1.], will have X coordinates x_i={0., 3./5., 1.}
///
/// For a few reasons, it is better to keep U-gaps between the knots minimal.
/// A spline with knots u_i={0,1,2} is mathematically the same as the spline with knots u_i={0,2,4}.
/// However, it uses less memory.
///
/// The minimal number of knots is 2.
///
/// --- Output dimensionality ---
///
/// There are two ways to set the dimensionality of Y - either in the constructor or as a template argument:
///
/// Spline1D<float> s( nYdimensions, nKnots );
/// Spline1D<float, nYdimensions> s( nKnots );
///
/// The second implementation works faster. Use it when nYdimensions is known at the compile time.
///
/// There is also a variation of the first specification which lets the compiler know
/// the maximal possible number of Y dimensions:
///
/// Spline1D<float, -nYdimensionsMax> s( nYdimensions, nKnots );
///
/// This specification does not allocate any variable-size arrays and therefore compiles for the GPU.
///
/// ---- External storage of spline parameters for a given F ---
///
/// One can store all F-dependent spline parameters outside of the spline object
/// and provide them at each interpolation call.
/// To do so, create a spline with nYdimensions=0; create spline parameters for F via Spline1DHelper class;
/// then use special interpolateU(..) methods for the interpolation.
///
/// This feature allows one to use the same spline object for the approximation of different functions
/// on the same grid of knots.
///
/// ---- Creation of a spline ----
///
/// The spline is supposed to be a best-fit spline, created by the approximateFunction() method.
///
/// Best-fit means that the spline values S_i and its derivatives D_i at the knots
/// are adjusted to minimize the overall difference between S(x) and F(x).
/// The spline constructed this way is much more accurate than a classical interpolation spline.
///
/// The difference to F() is minimized at all integer values of U coordinate (in particular, at all knots)
/// and at extra nAuxiliaryPoints points between the integer numbers.
///
/// nAuxiliaryPoints is given as a parameter of approximateFunction() method.
/// With nAuxiliaryPoints==3, the approximation accuracy is noticeably better than the one with 1 or 2.
/// Higher values usually give a little improvement over 3.
///
/// The number of auxiliary points does not influence the interpolation speed,
/// but a high number can slow down the spline's creation.
///
/// It is also possible to construct the spline classically - by taking F(x) values only at knots and making
/// the first and the second derivatives of S(x) continuous. Use the corresponding method from Spline1DHelper.
///
/// ---- Example of creating a spline ----
///
///  auto F = [&](double x, double &f) { // a function to be approximated
///   f[0] = x*x+3.f; // F(x)
///  };
///
///  const int nKnots = 3;
///
///  int knots[nKnots] = {0, 1, 5}; // relative(!) knot positions
///
///  Spline1D<float,1> spline( nKnots, knots ); // create 1-dimensional spline with the knots
///
///  spline.approximateFunction(0., 1., F); // let the spline approximate F on a segment [0., 1.]
///
///  float s = spline.interpolate(0.2); // interpolated value at x==0.2
///
///  --- See also Spline1D::test() method for examples
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
class Spline1D
  : public Spline1DSpec<DataT, YdimT, SplineUtil::getSpec(YdimT)>
{
  typedef Spline1DContainer<DataT> TVeryBase;
  typedef Spline1DSpec<DataT, YdimT, SplineUtil::getSpec(YdimT)> TBase;

 public:
  typedef typename TVeryBase::SafetyLevel SafetyLevel;
  typedef typename TVeryBase::Knot Knot;

#if !defined(GPUCA_GPUCODE)
  using TBase::TBase; // inherit constructors

  /// Assignment operator
  Spline1D() = default;
  Spline1D(const Spline1D& v) : TBase(v)
  {
    TVeryBase::cloneFromObject(v, nullptr);
  }
  Spline1D& operator=(const Spline1D& v)
  {
    TVeryBase::cloneFromObject(v, nullptr);
    return *this;
  }
#else
  /// Disable constructors for the GPU implementation
  Spline1D() CON_DELETE;
  Spline1D(const Spline1D&) CON_DELETE;
#endif

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) && !defined(GPUCA_ALIROOT_LIB)
  /// read a class object from the file
  static Spline1D* readFromFile(TFile& inpf, const char* name)
  {
    return (Spline1D*)TVeryBase::readFromFile(inpf, name);
  }
#endif

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(Spline1D, 0);
#endif
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
