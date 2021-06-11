// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
inline constexpr double sectorDAlpha() noexcept
{
  return constants::math::PI / 9;
};

//_________________________________________________________________________________
inline constexpr double sector2Alpha(int sect) noexcept
{
  // get barrel sector alpha in -pi:pi format
  if (sect > 8) {
    sect -= 18;
  }
  return (sect + 0.5) * sectorDAlpha();
};

//_________________________________________________________________________________
inline int phi2Sector(double phi)
{
  // get barrel sector from phi in -pi:pi format
  int sect = math_utils::nintd((phi * constants::math::Rad2Deg - 10) / 20.);
  if (sect < 0) {
    sect += 18;
  }
  return sect;
};

//_________________________________________________________________________________
template <typename F>
inline constexpr void bringTo02Pi(F& phi) noexcept
{
  // bring phi to 0-2pi range
  if (phi < 0) {
    phi += constants::math::TwoPI;
  } else if (phi > constants::math::TwoPI) {
    phi -= constants::math::TwoPI;
  }
};

//_________________________________________________________________________________
template <typename F>
inline constexpr void bringToPiPM(F& phi) noexcept
{
  // bring phi to -pi:pi range
  if (phi > constants::math::PI) {
    phi -= constants::math::TwoPI;
  }
};

//_________________________________________________________________________________
template <typename F>
inline constexpr bool okForPhiMin(F phiMin, F phi) noexcept
{
  // check if phi is above the phiMin, phi's must be in 0-2pi range
  const F dphi = phi - phiMin;
  if ((dphi > 0 && dphi < constants::math::PI) || dphi < -constants::math::PI) {
    return true;
  } else {
    return false;
  }
};

//_________________________________________________________________________________
template <typename F>
inline constexpr bool okForPhiMax(F phiMax, F phi) noexcept
{
  // check if phi is below the phiMax, phi's must be in 0-2pi range
  const F dphi = phi - phiMax;
  if ((dphi < 0 && dphi > -constants::math::PI) || dphi > constants::math::PI) {
    return true;
  } else {
    return false;
  }
};

//_________________________________________________________________________________
template <typename F>
constexpr F meanPhiSmall(F phi0, F phi1)
{
  // return mean phi, assume phis in 0:2pi
  F phi;
  if (!okForPhiMin(phi0, phi1)) {
    phi = phi0;
    phi0 = phi1;
    phi1 = phi;
  }
  if (phi0 > phi1) {
    phi = (phi1 - (constants::math::TwoPI - phi0)) / 2; // wrap
  } else {
    phi = (phi0 + phi1) / 2;
  }
  bringTo02Pi(phi);
  return phi;
};

//_________________________________________________________________________________
template <typename F>
constexpr F deltaPhiSmall(F phi0, F phi1) noexcept
{
  // return delta phi, assume phi is in 0:2pi
  F del;
  if (!okForPhiMin(phi0, phi1)) {
    del = phi0;
    phi0 = phi1;
    phi1 = del;
  }
  del = phi1 - phi0;
  if (del < 0) {
    del += constants::math::TwoPI;
  }
  return del;
};

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

//_________________________________________________________________________________
inline constexpr int numberOfBitsSet(uint32_t x) noexcept
{
  // count number of non-0 bits in 32bit word
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
};

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
