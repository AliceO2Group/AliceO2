// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CircleXY.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 10, 2020
/// @brief

#ifndef MATHUTILS_INCLUDE_MATHUTILS_DETAIL_CIRCLEXY_H_
#define MATHUTILS_INCLUDE_MATHUTILS_DETAIL_CIRCLEXY_H_

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"

namespace o2
{
namespace math_utils
{
namespace detail
{

template <typename T>
struct CircleXY {
  using value_t = T;

  T rC; // circle radius
  T xC; // x-center
  T yC; // y-center
  GPUd() CircleXY(T r = T(), T x = T(), T y = T());
  GPUd() T getCenterD2() const;
  ClassDefNV(CircleXY, 2);
};

template <typename T>
GPUdi() CircleXY<T>::CircleXY(T r, T x, T y) : rC(r), xC(x), yC(y)
{
}

template <typename T>
GPUdi() T CircleXY<T>::getCenterD2() const
{
  return xC * xC + yC * yC;
}
} // namespace detail
} // namespace math_utils
} // namespace o2

#endif /* MATHUTILS_INCLUDE_MATHUTILS_DETAIL_CIRCLEXY_H_ */
