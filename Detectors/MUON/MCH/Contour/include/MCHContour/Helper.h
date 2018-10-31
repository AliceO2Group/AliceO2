// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#ifndef O2_MCH_CONTOUR_HELPER_H
#define O2_MCH_CONTOUR_HELPER_H

#include <cmath>
#include <limits>
#include <cstdint>

namespace o2
{
namespace mch
{
namespace contour
{
namespace impl
{

template <typename T, typename U>
bool CanTypeFitValue(const U value)
{
  const intmax_t botT = intmax_t(std::numeric_limits<T>::lowest());
  const intmax_t botU = intmax_t(std::numeric_limits<U>::lowest());
  const uintmax_t topT = uintmax_t(std::numeric_limits<T>::max());
  const uintmax_t topU = uintmax_t(std::numeric_limits<U>::max());
  return !((botT > botU && value < static_cast<U>(botT)) || (topT < topU && value > static_cast<U>(topT)));
}

inline bool areEqual(double a, double b)
{
  return std::fabs(b - a) < 1E-4; // 1E-4 cm = 1 micron
}

inline bool areEqual(int a, int b) { return a == b; }

inline bool isStrictlyBelow(double a, double b) { return (a < b) && !areEqual(a, b); }

inline bool isStrictlyBelow(int a, int b) { return a < b; }

} // namespace impl
} // namespace contour
} // namespace mch
} // namespace o2

#endif // ALO_HELPER_H
