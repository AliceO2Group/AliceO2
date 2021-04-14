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
const double kAlmostZeroD = 1e-15;
const float kAlmostZeroF = 1e-11;
const double kAlmostOneD = 1. - kAlmostZeroD;
const float kAlmostOneF = 1. - kAlmostZeroF;
const double kTinyDist = 1.e-7; // ignore distances less that this
//
enum { kColl,
       kCosm,
       kNTrackTypes };
//
inline double sector2Alpha(int sect);
inline int phi2Sector(double alpha);
inline double sectorDAlpha() { return constants::math::PI / 9; }
//
template <typename F>
void bringTo02Pi(F& phi);
template <typename F>
void bringToPiPM(F& phi);
template <typename F>
bool okForPhiMin(F phiMin, F phi);
template <typename F>
bool okForPhiMax(F phiMax, F phi);
template <typename F>
F meanPhiSmall(F phi0, F phi1);
template <typename F>
F deltaPhiSmall(F phi0, F phi1);
template <typename F>
bool smallerAbs(F d, F tolD)
{
  return std::abs(d) < tolD;
}
template <typename F>
bool smaller(F d, F tolD)
{
  return d < tolD;
}
//
inline int numberOfBitsSet(uint32_t x);
inline bool isZeroAbs(double d) { return smallerAbs(d, kAlmostZeroD); }
inline bool isZeroAbs(float f) { return smallerAbs(f, kAlmostZeroF); }
inline bool isZeroPos(double d) { return smaller(d, kAlmostZeroD); }
inline bool isZeroPos(float f) { return smaller(f, kAlmostZeroF); }
//
int findKeyIndex(int key, const int* arr, int n);
//
void printBits(size_t patt, int maxBits);

} // namespace utils

//_________________________________________________________________________________
template <typename F>
inline void utils::bringTo02Pi(F& phi)
{
  // bring phi to 0-2pi range
  if (phi < 0)
    phi += constants::math::TwoPI;
  else if (phi > constants::math::TwoPI)
    phi -= constants::math::TwoPI;
}

//_________________________________________________________________________________
template <typename F>
inline void utils::bringToPiPM(F& phi)
{
  // bring phi to -pi:pi range
  if (phi > constants::math::PI)
    phi -= constants::math::TwoPI;
}
//_________________________________________________________________________________
template <typename F>
inline bool utils::okForPhiMin(F phiMin, F phi)
{
  // check if phi is above the phiMin, phi's must be in 0-2pi range
  F dphi = phi - phiMin;
  return ((dphi > 0 && dphi < constants::math::PI) || dphi < -constants::math::PI) ? true : false;
}

//_________________________________________________________________________________
template <typename F>
inline bool utils::okForPhiMax(F phiMax, F phi)
{
  // check if phi is below the phiMax, phi's must be in 0-2pi range
  F dphi = phi - phiMax;
  return ((dphi < 0 && dphi > -constants::math::PI) || dphi > constants::math::PI) ? true : false;
}

//_________________________________________________________________________________
template <typename F>
inline F utils::meanPhiSmall(F phi0, F phi1)
{
  // return mean phi, assume phis in 0:2pi
  F phi;
  if (!okForPhiMin(phi0, phi1)) {
    phi = phi0;
    phi0 = phi1;
    phi1 = phi;
  }
  if (phi0 > phi1)
    phi = (phi1 - (constants::math::TwoPI - phi0)) / 2; // wrap
  else
    phi = (phi0 + phi1) / 2;
  bringTo02Pi(phi);
  return phi;
}

//_________________________________________________________________________________
template <typename F>
inline F utils::deltaPhiSmall(F phi0, F phi1)
{
  // return delta phi, assume phis in 0:2pi
  F del;
  if (!okForPhiMin(phi0, phi1)) {
    del = phi0;
    phi0 = phi1;
    phi1 = del;
  }
  del = phi1 - phi0;
  if (del < 0)
    del += constants::math::TwoPI;
  return del;
}

//_________________________________________________________________________________
inline int utils::numberOfBitsSet(uint32_t x)
{
  // count number of non-0 bits in 32bit word
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

//_________________________________________________________________________________
inline double utils::sector2Alpha(int sect)
{
  // get barrel sector alpha in -pi:pi format
  if (sect > 8)
    sect -= 18;
  return (sect + 0.5) * sectorDAlpha();
}

//_________________________________________________________________________________
inline int utils::phi2Sector(double phi)
{
  // get barrel sector from phi in -pi:pi format
  int sect = math_utils::nintd((phi * constants::math::Rad2Deg - 10) / 20.);
  if (sect < 0)
    sect += 18;
  return sect;
}

//_______________________________________________________________
inline void utils::printBits(size_t patt, int maxBits)
{
  // print maxBits of the pattern
  maxBits = std::min(64, maxBits);
  for (int i = 0; i < maxBits; i++)
    printf("%c", ((patt >> i) & 0x1) ? '+' : '-');
}

//__________________________________________
inline int utils::findKeyIndex(int key, const int* arr, int n)
{
  // finds index of key in the array
  int imn = 0, imx = n - 1;
  while (imx >= imn) {
    int mid = (imx + imn) >> 1;
    if (arr[mid] == key)
      return mid;
    if (arr[mid] < key)
      imn = mid + 1;
    else
      imx = mid - 1;
  }
  return -1;
}

} // namespace align
} // namespace o2
#endif
