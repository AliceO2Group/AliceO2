// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Primitive2D.h
/// \brief Declarations of 2D primitives
/// \author ruben.shahoyan@cern.ch michael.lettrich@cern.ch

#ifndef ALICEO2_COMMON_MATH_PRIMITIVE2D_H
#define ALICEO2_COMMON_MATH_PRIMITIVE2D_H

#include "GPUCommonRtypes.h"
#include "MathUtils/detail/CircleXY.h"
#include "MathUtils/detail/IntervalXY.h"
#include "MathUtils/detail/Bracket.h"

namespace o2
{
namespace math_utils
{
template <typename T>
using CircleXY = detail::CircleXY<T>;
using CircleXYf_t = detail::CircleXY<float>;
using CircleXYd_t = detail::CircleXY<double>;

template <typename T>
using IntervalXY = detail::IntervalXY<T>;
using IntervalXYf_t = detail::IntervalXY<float>;
using IntervalXYd_t = detail::IntervalXY<double>;

template <typename T>
using Bracket = detail::Bracket<T>;
using Bracketf_t = detail::Bracket<float>;
using Bracketd_t = detail::Bracket<double>;

} // namespace math_utils
} // namespace o2

#endif
