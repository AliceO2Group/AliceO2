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

/// @file   TrackParametrizationHelpers.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 1, 2020
/// @brief Helper functions for track manipulation

#ifndef INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKUTILS_H_
#define INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKUTILS_H_

#include "GPUCommonRtypes.h"
#include "GPUCommonArray.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include <cmath>
#endif

#include "MathUtils/Utils.h"
#include "CommonConstants/MathConstants.h"

namespace o2
{
namespace track
{
// helper function
template <typename value_T = float>
GPUd() value_T BetheBlochSolid(value_T bg, value_T rho = 2.33, value_T kp1 = 0.20, value_T kp2 = 3.00, value_T meanI = 173e-9, value_T meanZA = 0.49848);

template <typename value_T = float>
GPUd() value_T BetheBlochSolidOpt(value_T bg);

template <typename value_T = float>
GPUd() void g3helx3(value_T qfield, value_T step, gpu::gpustd::array<value_T, 7>& vect);

//____________________________________________________
template <typename value_T>
GPUd() void g3helx3(value_T qfield, value_T step, gpu::gpustd::array<value_T, 7>& vect)
{
  /******************************************************************
   *                                                                *
   *       GEANT3 tracking routine in a constant field oriented     *
   *       along axis 3                                             *
   *       Tracking is performed with a conventional                *
   *       helix step method                                        *
   *                                                                *
   *       Authors    R.Brun, M.Hansroul  *********                 *
   *       Rewritten  V.Perevoztchikov                              *
   *                                                                *
   *       Rewritten in C++ by I.Belikov                            *
   *                                                                *
   *  qfield (kG)       - particle charge times magnetic field      *
   *  step   (cm)       - step length along the helix               *
   *  vect[7](cm,GeV/c) - input/output x, y, z, px/p, py/p ,pz/p, p *
   *                                                                *
   ******************************************************************/
#ifndef GPUCA_GPUCODE_DEVICE
  static_assert(std::is_floating_point_v<value_T>);
#endif

  const int ix = 0, iy = 1, iz = 2, ipx = 3, ipy = 4, ipz = 5, ipp = 6;
  constexpr value_T kOvSqSix = 0.408248f; // std::sqrt(1./6.);

  value_T cosx = vect[ipx], cosy = vect[ipy], cosz = vect[ipz];

  value_T rho = qfield * constants::math::B2C / vect[ipp];
  value_T tet = rho * step;

  value_T tsint, sintt, sint, cos1t;
  if (gpu::CAMath::Abs(tet) > 0.03f) {
    sint = gpu::CAMath::Sin(tet);
    sintt = sint / tet;
    tsint = (tet - sint) / tet;
    value_T t = gpu::CAMath::Sin(0.5f * tet);
    cos1t = 2 * t * t / tet;
  } else {
    tsint = tet * tet / 6.f;
    sintt = (1.f - tet * kOvSqSix) * (1.f + tet * kOvSqSix); // 1.- tsint;
    sint = tet * sintt;
    cos1t = 0.5f * tet;
  }

  value_T f1 = step * sintt;
  value_T f2 = step * cos1t;
  value_T f3 = step * tsint * cosz;
  value_T f4 = -tet * cos1t;
  value_T f5 = sint;

  vect[ix] += f1 * cosx - f2 * cosy;
  vect[iy] += f1 * cosy + f2 * cosx;
  vect[iz] += f1 * cosz + f3;

  vect[ipx] += f4 * cosx - f5 * cosy;
  vect[ipy] += f4 * cosy + f5 * cosx;
}

//____________________________________________________
template <typename value_T>
GPUd() value_T BetheBlochSolid(value_T bg, value_T rho, value_T kp1, value_T kp2, value_T meanI,
                               value_T meanZA)
{
  //
  // This is the parameterization of the Bethe-Bloch formula inspired by Geant.
  //
  // bg  - beta*gamma
  // rho - density [g/cm^3]
  // kp1 - density effect first junction point
  // kp2 - density effect second junction point
  // meanI - mean excitation energy [GeV]
  // meanZA - mean Z/A
  //
  // The default values for the kp* parameters are for silicon.
  // The returned value is in [GeV/(g/cm^2)].
  //
#ifndef GPUCA_GPUCODE_DEVICE
  static_assert(std::is_floating_point_v<value_T>);
#endif

  constexpr value_T mK = 0.307075e-3; // [GeV*cm^2/g]
  constexpr value_T me = 0.511e-3;    // [GeV/c^2]
  kp1 *= 2.303f;
  kp2 *= 2.303f;
  value_T bg2 = bg * bg, beta2 = bg2 / (1 + bg2);
  value_T maxT = 2.f * me * bg2; // neglecting the electron mass

  //*** Density effect
  value_T d2 = 0.;
  const value_T x = gpu::CAMath::Log(bg);
  const value_T lhwI = gpu::CAMath::Log(28.816f * 1e-9f * gpu::CAMath::Sqrt(rho * meanZA) / meanI);
  if (x > kp2) {
    d2 = lhwI + x - 0.5f;
  } else if (x > kp1) {
    double r = (kp2 - x) / (kp2 - kp1);
    d2 = lhwI + x - 0.5f + (0.5f - lhwI - kp1) * r * r * r;
  }
  auto dedx = mK * meanZA / beta2 * (0.5f * gpu::CAMath::Log(2 * me * bg2 * maxT / (meanI * meanI)) - beta2 - d2);
  return dedx > 0. ? dedx : 0.;
}

//____________________________________________________
template <typename value_T>
GPUd() value_T BetheBlochSolidOpt(value_T bg)
{
  //
  // This is the parameterization of the Bethe-Bloch formula inspired by Geant with hardcoded constants and better optimization
  //
  // bg  - beta*gamma
  // rho - density [g/cm^3]
  // kp1 - density effect first junction point
  // kp2 - density effect second junction point
  // meanI - mean excitation energy [GeV]
  // meanZA - mean Z/A
  //
  // The default values for the kp* parameters are for silicon.
  // The returned value is in [GeV/(g/cm^2)].
  //
  //  constexpr value_T rho = 2.33;
  //  constexpr value_T meanI = 173e-9;
  //  constexpr value_T me = 0.511e-3;    // [GeV/c^2]

  constexpr value_T mK = 0.307075e-3; // [GeV*cm^2/g]
  constexpr value_T kp1 = 0.20 * 2.303;
  constexpr value_T kp2 = 3.00 * 2.303;
  constexpr value_T meanZA = 0.49848;
  constexpr value_T lhwI = -1.7175226;         // gpu::CAMath::Log(28.816 * 1e-9 * gpu::CAMath::Sqrt(rho * meanZA) / meanI);
  constexpr value_T log2muTomeanI = 8.6839805; // gpu::CAMath::Log( 2. * me / meanI);

  value_T bg2 = bg * bg, beta2 = bg2 / (1. + bg2);

  //*** Density effect
  value_T d2 = 0.;
  const value_T x = gpu::CAMath::Log(bg);
  if (x > kp2) {
    d2 = lhwI - 0.5f + x;
  } else if (x > kp1) {
    value_T r = (kp2 - x) / (kp2 - kp1);
    d2 = lhwI - 0.5 + x + (0.5 - lhwI - kp1) * r * r * r;
  }
  auto dedx = mK * meanZA / beta2 * (log2muTomeanI + x + x - beta2 - d2);
  return dedx > 0. ? dedx : 0.;
}

//____________________________________________________
template <typename value_T>
GPUd() value_T inline BetheBlochSolidDerivative(value_T dedx, value_T bg)
{
  //
  // This is approximate derivative of the BB over betagamm, NO check for the consistency of the provided dedx and bg is done
  // Charge 1 particle is assumed for the provied dedx. For charge > 1 particles dedx/q^2 should be provided and obtained value must be scaled by q^2
  // The call should be usually done as
  // auto dedx = BetheBlochSolidOpt(bg);
  // // if derivative needed
  // auto ddedx = BetheBlochSolidDerivative(dedx, bg, bg*bg)
  //
  // dedx - precalculate dedx for bg
  // bg  - beta*gamma
  //
  constexpr value_T mK = 0.307075e-3; // [GeV*cm^2/g]
  constexpr value_T meanZA = 0.49848;
  auto bg2 = bg * bg;
  auto t1 = 1 + bg2;
  //  auto derH = (mK * meanZA * (t1+bg2) - dedx*bg2)/(bg*t1);
  auto derH = (mK * meanZA * (t1 + 1. / bg2) - dedx) / (bg * t1);
  return derH + derH;
}

} // namespace track
} // namespace o2

#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKUTILS_H_ */
