// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  RegularSpline1D.h
/// \brief Definition of IrregularSpline1D class
///
/// \author  Felix Lapp
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_REGULARSPLINE1D_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_REGULARSPLINE1D_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
///
/// The RegularSpline1D class represents one-dimensional spline interpolation on a regular grid.
/// It is a simplification of the IrregularSpline1D class which represents irregular splines.
/// For the description please look at IrregularSpline1D.h .
///
/// The class consist of only one member: mNumberOfKnots. Basically, the class is a collection of methods.
///
class RegularSpline1D
{
 public:
  /// _____________  Constructors / destructors __________________________

  /// Default constructor
  RegularSpline1D() CON_DEFAULT;

  /// Destructor
  ~RegularSpline1D() CON_DEFAULT;

  /// Constructor. Number of knots will be set to at least 5
  void construct(int numberOfKnots);

  /// _______________  Main functionality   ________________________

  /// Correction of data values at the both edge knots.
  /// It is needed for the fast spline mathematics to work correctly. See explanation at the class comment above.
  ///
  /// \param data array of function values. It has the size of getNumberOfKnots()
  template <typename T>
  void correctEdges(T* data) const;

  /// Get interpolated value for f(u) using spline at knot "knot_1" and function values at knots {knot_0,knot_1,knot_2,knot_3}
  template <typename T>
  T getSpline(const int iknot, T f0, T f1, T f2, T f3, float u) const;

  /// Get interpolated value for f(u) using data array correctedData[getNumberOfKnots()] with corrected edges
  template <typename T>
  T getSpline(const T correctedData[], float u) const;

  /// Get number of knots
  int getNumberOfKnots() const { return mNumberOfKnots; }

  /// Get the U-Coordinate depending on the given knot-Index
  double knotIndexToU(int index) const;

  /// Get index of associated knot for a given U coordinate.
  ///
  /// Note: U values from the first interval are mapped to the second inrerval.
  /// Values from the last interval are mapped to the previous interval.
  ///
  int getKnotIndex(float u) const;

 private:
  unsigned char mNumberOfKnots = 5; ///< n knots on the grid
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

inline void RegularSpline1D::construct(int numberOfKnots)
{
  /// Constructor
  mNumberOfKnots = (numberOfKnots < 5) ? 5 : numberOfKnots;
}

template <typename T>
inline T RegularSpline1D::getSpline(const int iknot1, T f0, T f1, T f2, T f3, float u) const
{
  /// static method
  /// Get interpolated value for f(u) using a spline polynom for [iknot1,iknot2] interval.
  /// The polynom is constructed with function values f0,f1,f2,f3 at knots {iknot0,iknot1,iknot2,iknot3}
  /// The u value supposed to be inside the [knot1,knot2] region, but also may be any.

  ///f0 = f value at iknot1-1
  ///f1 = f value at iknot1
  ///f2 = f value at iknot1+1
  ///f3 = f value at iknot1+2
  ///u = u value where f(u) is searched for.

  f0 -= f1;
  f2 -= f1;
  f3 -= f1;

  const T half(0.5f);

  // Scale u so that the knot1 is at 0.0 and the knot2 is at 1.0
  T x = T((mNumberOfKnots - 1) * u - iknot1);
  T x2 = x * x;

  T z1 = half * (f2 - f0); // scaled u derivative at the knot 1
  T z2 = half * f3;        // scaled u derivative at the knot 2

  // f(u) = a*u^3 + b*u^2 + c*u + d
  //
  // f(0) = f1
  // f(1) = f2
  // f'(0) = z1 = 0.5 f2  -  0.5 f0
  // f'(1) = z2 = 0.5 f3  -  0.5 f1
  //
  // T d = f1;
  // T c = z1;

  T a = -f2 - f2 + z1 + z2;
  T b = f2 - z1 - a;
  return (a * x + b) * x2 + z1 * x + f1;
}

template <typename T>
inline T RegularSpline1D::getSpline(const T correctedData[], float u) const
{
  /// Get interpolated value for f(u) using data array correctedData[getNumberOfKnots()] with corrected edges
  int iknot = getKnotIndex(u);
  const T* f = correctedData + iknot - 1;
  return getSpline(iknot, f[0], f[1], f[2], f[3], u);
}

inline double RegularSpline1D::knotIndexToU(int iknot) const
{
  if (iknot <= 0)
    return 0;
  if (iknot >= mNumberOfKnots)
    return 1;
  return iknot / ((double)mNumberOfKnots - 1.);
}

inline int RegularSpline1D::getKnotIndex(float u) const
{
  //index is just u elem [0, 1] * numberOfKnots and then floored. (so the "left" coordinate beside u gets chosen)
  int index = (int)(u * (mNumberOfKnots - 1));
  if (index <= 1)
    index = 1;
  else if (index >= mNumberOfKnots - 3)
    index = mNumberOfKnots - 3;
  return index;
}

template <typename T>
inline void RegularSpline1D::correctEdges(T* data) const
{
  // data[i] is the i-th f-value
  constexpr T c0(0.5), c1(1.5);
  const int i = mNumberOfKnots - 1;
  data[0] = c0 * (data[0] + data[3]) + c1 * (data[1] - data[2]);
  data[i] = c0 * (data[i - 0] + data[i - 3]) + c1 * (data[i - 1] - data[i - 2]);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
