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

/// @file   utils.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Collection of auxillary methods

#ifndef UTILS_H
#define UTILS_H

#include "CommonConstants/MathConstants.h"
#include "MathUtils/Utils.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace align
{

using trackParam_t = typename track::TrackParametrizationWithError<double>;
using value_t = typename trackParam_t::value_t;
using dim2_t = typename trackParam_t::dim2_t;
using dim3_t = typename trackParam_t::dim3_t;
using params_t = typename trackParam_t::params_t;
using covMat_t = typename trackParam_t::covMat_t;

namespace utils
{
constexpr double AlmostZeroD = 1e-15;
constexpr float AlmostZeroF = 1e-11;
constexpr double AlmostOneD = 1. - AlmostZeroD;
constexpr float AlmostOneF = 1. - AlmostZeroF;
constexpr double TinyDist = 1.e-7; // ignore distances less that this

//_________________________________________________________________________________
enum { Coll,
       Cosm,
       NTrackTypes };

//_________________________________________________________________________________
template <typename F>
inline constexpr bool smallerAbs(F d, F tolD) noexcept
{
  return std::abs(d) < tolD;
};

//_________________________________________________________________________________
template <typename F>
inline constexpr bool smaller(F d, F tolD) noexcept
{
  return d < tolD;
}

inline constexpr bool isZeroAbs(double d) noexcept { return smallerAbs(d, AlmostZeroD); };
inline constexpr bool isZeroAbs(float f) noexcept { return smallerAbs(f, AlmostZeroF); }
inline constexpr bool isZeroPos(double d) noexcept { return smaller(d, AlmostZeroD); }
inline constexpr bool isZeroPos(float f) noexcept { return smaller(f, AlmostZeroF); }

//__________________________________________
inline constexpr int findKeyIndex(int key, const int* arr, int n) noexcept
{
  // finds index of key in the array
  int imn = 0;
  int imx = n - 1;
  while (imx >= imn) {
    const int mid = (imx + imn) >> 1;
    if (arr[mid] == key) {
      return mid;
    }

    if (arr[mid] < key) {
      imn = mid + 1;
    } else {
      imx = mid - 1;
    }
  }
  return -1;
}

//_______________________________________________________________
inline void printBits(size_t patt, int maxBits)
{
  // print maxBits of the pattern
  maxBits = std::min(64, maxBits);
  for (int i = 0; i < maxBits; i++) {
    printf("%c", ((patt >> i) & 0x1) ? '+' : '-');
  }
};

} // namespace utils
} // namespace align
} // namespace o2
#endif
