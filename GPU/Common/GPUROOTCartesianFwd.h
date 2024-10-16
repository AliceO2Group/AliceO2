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

/// \file GPUROOTCartesianFwd.h
/// \author David Rohr

#ifndef GPUROOTCARTESIANFWD_H
#define GPUROOTCARTESIANFWD_H

// Standalone forward declarations for Cartesian2D / Cartesian3D / Point2D / Point3D etc.
// To be used on GPU where ROOT is not available.

#include "GPUCommonDef.h"

namespace ROOT
{
namespace Math
{
template <class T, uint32_t D1, uint32_t D2, class R>
class SMatrix;
template <class T, uint32_t D>
class MatRepSym;
template <class T, uint32_t D1, uint32_t D2>
class MatRepStd;
template <class CoordSystem, class Tag>
class PositionVector2D;
template <class CoordSystem, class Tag>
class PositionVector3D;
template <class CoordSystem, class Tag>
class DisplacementVector2D;
template <class CoordSystem, class Tag>
class DisplacementVector3D;
template <class T>
class Cartesian2D;
template <class T>
class Cartesian3D;
class DefaultCoordinateSystemTag;
} // namespace Math
} // namespace ROOT

namespace o2
{
namespace math_utils
{

namespace detail
{
template <typename T, int32_t I>
struct GPUPoint2D;
template <typename T, int32_t I>
struct GPUPoint3D;
} // namespace detail

#if (!defined(GPUCA_STANDALONE) || !defined(DGPUCA_NO_ROOT)) && !defined(GPUCA_GPUCODE) && !defined(GPUCOMMONRTYPES_H_ACTIVE)
template <typename T>
using Point2D = ROOT::Math::PositionVector2D<ROOT::Math::Cartesian2D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector2D = ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Point3D = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector3D = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
#else
template <typename T>
using Point2D = detail::GPUPoint2D<T, 0>;
template <typename T>
using Vector2D = detail::GPUPoint2D<T, 1>;
template <typename T>
using Point3D = detail::GPUPoint3D<T, 0>;
template <typename T>
using Vector3D = detail::GPUPoint3D<T, 1>;
#endif

} // namespace math_utils
} // namespace o2

#endif
