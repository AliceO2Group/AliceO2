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

/// @file   TrackParametrization.cxx
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 1, 2020
/// @brief

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

#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "ReconstructionDataFormats/DCA.h"
#include <MathUtils/Cartesian.h>
#include <GPUCommonLogger.h>

#ifndef GPUCA_GPUCODE_DEVICE
#include <iostream>
#endif

#ifndef GPUCA_ALIGPUCODE
#include <fmt/printf.h>
#endif

using namespace o2::gpu;
using namespace o2::track;

//______________________________________________________________
template <typename value_T>
GPUd() TrackParametrization<value_T>::TrackParametrization(const dim3_t& xyz, const dim3_t& pxpypz, int charge, bool sectorAlpha, const PID pid)
  : mX{0.f}, mAlpha{0.f}, mP{0.f}
{
  // construct track param from kinematics

  // Alpha of the frame is defined as:
  // sectorAlpha == false : -> angle of pt direction
  // sectorAlpha == true  : -> angle of the sector from X,Y coordinate for r>1
  //                           angle of pt direction for r==0
  //
  //
  constexpr value_t kSafe = 1e-5;
  value_t radPos2 = xyz[0] * xyz[0] + xyz[1] * xyz[1];
  value_t alp = 0;
  if (sectorAlpha || radPos2 < 1) {
    alp = gpu::CAMath::ATan2(pxpypz[1], pxpypz[0]);
  } else {
    alp = gpu::CAMath::ATan2(xyz[1], xyz[0]);
  }
  if (sectorAlpha) {
    alp = math_utils::detail::angle2Alpha<value_t>(alp);
  }
  //
  value_t sn, cs;
  math_utils::detail::sincos(alp, sn, cs);
  // protection against cosp<0
  if (cs * pxpypz[0] + sn * pxpypz[1] < 0) {
    LOG(debug) << "alpha from phiPos() will invalidate this track parameters, overriding to alpha from phi()";
    alp = gpu::CAMath::ATan2(pxpypz[1], pxpypz[0]);
    if (sectorAlpha) {
      alp = math_utils::detail::angle2Alpha<value_t>(alp);
    }
    math_utils::detail::sincos(alp, sn, cs);
  }

  // protection:  avoid alpha being too close to 0 or +-pi/2
  if (gpu::CAMath::Abs(sn) < 2 * kSafe) {
    if (alp > 0) {
      alp += alp < constants::math::PIHalf ? 2 * kSafe : -2 * kSafe;
    } else {
      alp += alp > -constants::math::PIHalf ? -2 * kSafe : 2 * kSafe;
    }
    math_utils::detail::sincos(alp, sn, cs);
  } else if (gpu::CAMath::Abs(cs) < 2 * kSafe) {
    if (alp > 0) {
      alp += alp > constants::math::PIHalf ? 2 * kSafe : -2 * kSafe;
    } else {
      alp += alp > -constants::math::PIHalf ? 2 * kSafe : -2 * kSafe;
    }
    math_utils::detail::sincos(alp, sn, cs);
  }
  // get the vertex of origin and the momentum
  dim3_t ver{xyz[0], xyz[1], xyz[2]};
  dim3_t mom{pxpypz[0], pxpypz[1], pxpypz[2]};
  //
  // Rotate to the local coordinate system
  math_utils::detail::rotateZ<value_t>(ver, -alp);
  math_utils::detail::rotateZ<value_t>(mom, -alp);
  //
  value_t ptI = 1.f / gpu::CAMath::Sqrt(mom[0] * mom[0] + mom[1] * mom[1]);
  mX = ver[0];
  mAlpha = alp;
  mP[kY] = ver[1];
  mP[kZ] = ver[2];
  mP[kSnp] = mom[1] * ptI;
  mP[kTgl] = mom[2] * ptI;
  mAbsCharge = gpu::CAMath::Abs(charge);
  mP[kQ2Pt] = charge ? ptI * charge : ptI;
  mPID = pid;
  //
  if (gpu::CAMath::Abs(1 - getSnp()) < kSafe) {
    mP[kSnp] = 1.f - kSafe; // Protection
  } else if (gpu::CAMath::Abs(-1 - getSnp()) < kSafe) {
    mP[kSnp] = -1.f + kSafe; // Protection
  }
  //
}

//_______________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::getPxPyPzGlo(dim3_t& pxyz) const
{
  // track momentum
  if (gpu::CAMath::Abs(getQ2Pt()) < constants::math::Almost0 || gpu::CAMath::Abs(getSnp()) > constants::math::Almost1) {
    return false;
  }
  value_t cs, sn, pt = getPt();
  value_t r = gpu::CAMath::Sqrt((1.f - getSnp()) * (1.f + getSnp()));
  math_utils::detail::sincos(getAlpha(), sn, cs);
  pxyz[0] = pt * (r * cs - getSnp() * sn);
  pxyz[1] = pt * (getSnp() * cs + r * sn);
  pxyz[2] = pt * getTgl();
  return true;
}

//____________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::getPosDirGlo(gpu::gpustd::array<value_t, 9>& posdirp) const
{
  // fill vector with lab x,y,z,px/p,py/p,pz/p,p,sinAlpha,cosAlpha
  value_t ptI = getPtInv();
  value_t snp = getSnp();
  if (gpu::CAMath::Abs(snp) > constants::math::Almost1) {
    return false;
  }
  value_t &sn = posdirp[7], &cs = posdirp[8];
  value_t csp = gpu::CAMath::Sqrt((1.f - snp) * (1.f + snp));
  value_t cstht = gpu::CAMath::Sqrt(1.f + getTgl() * getTgl());
  value_t csthti = 1.f / cstht;
  math_utils::detail::sincos(getAlpha(), sn, cs);
  posdirp[0] = getX() * cs - getY() * sn;
  posdirp[1] = getX() * sn + getY() * cs;
  posdirp[2] = getZ();
  posdirp[3] = (csp * cs - snp * sn) * csthti; // px/p
  posdirp[4] = (snp * cs + csp * sn) * csthti; // py/p
  posdirp[5] = getTgl() * csthti;              // pz/p
  posdirp[6] = cstht / ptI;                    // p
  return true;
}

//______________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::rotateParam(value_t alpha)
{
  // rotate to alpha frame
  if (gpu::CAMath::Abs(getSnp()) > constants::math::Almost1) {
    LOGP(debug, "Precondition is not satisfied: |sin(phi)|>1 ! {:f}", getSnp());
    return false;
  }
  //
  math_utils::detail::bringToPMPi<value_t>(alpha);
  //
  value_t ca = 0, sa = 0;
  math_utils::detail::sincos(alpha - getAlpha(), sa, ca);
  value_t snp = getSnp(), csp = gpu::CAMath::Sqrt((1.f - snp) * (1.f + snp)); // Improve precision
  // RS: check if rotation does no invalidate track model (cos(local_phi)>=0, i.e. particle
  // direction in local frame is along the X axis
  if ((csp * ca + snp * sa) < 0) {
    // LOGF(warning,"Rotation failed: local cos(phi) would become {:.2f}", csp * ca + snp * sa);
    return false;
  }
  //
  value_t tmp = snp * ca - csp * sa;
  if (gpu::CAMath::Abs(tmp) > constants::math::Almost1) {
    LOGP(debug, "Rotation failed: new snp {:.2f}", tmp);
    return false;
  }
  value_t xold = getX(), yold = getY();
  mAlpha = alpha;
  mX = xold * ca + yold * sa;
  mP[kY] = -xold * sa + yold * ca;
  mP[kSnp] = tmp;
  return true;
}

//____________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::propagateParamTo(value_t xk, const dim3_t& b)
{
  //----------------------------------------------------------------
  // Extrapolate this track params (w/o cov matrix) to the plane X=xk in the field b[].
  //
  // X [cm] is in the "tracking coordinate system" of this track.
  // b[]={Bx,By,Bz} [kG] is in the Global coordidate system.
  //----------------------------------------------------------------
  value_t dx = xk - getX();
  if (gpu::CAMath::Abs(dx) < constants::math::Almost0) {
    return true;
  }
  // Do not propagate tracks outside the ALICE detector
  if (gpu::CAMath::Abs(dx) > 1e5 || gpu::CAMath::Abs(getY()) > 1e5 || gpu::CAMath::Abs(getZ()) > 1e5) {
    LOG(warning) << "Anomalous track, traget X:" << xk;
    return false;
  }
  value_t crv = getCurvature(b[2]);
  if (crv == 0.) {
    return propagateParamTo(xk, 0.); // for the straight-line propagation use 1D field method
  }

  value_t x2r = crv * dx;
  value_t f1 = getSnp(), f2 = f1 + x2r;
  if (gpu::CAMath::Abs(f1) > constants::math::Almost1 || gpu::CAMath::Abs(f2) > constants::math::Almost1) {
    return false;
  }
  value_t r1 = gpu::CAMath::Sqrt((1.f - f1) * (1.f + f1));
  if (gpu::CAMath::Abs(r1) < constants::math::Almost0) {
    return false;
  }
  value_t r2 = gpu::CAMath::Sqrt((1.f - f2) * (1.f + f2));
  if (gpu::CAMath::Abs(r2) < constants::math::Almost0) {
    return false;
  }
  value_t dy2dx = (f1 + f2) / (r1 + r2);
  value_t step = (gpu::CAMath::Abs(x2r) < 0.05f) ? dx * gpu::CAMath::Abs(r2 + f2 * dy2dx)                                              // chord
                                                 : 2.f * CAMath::ASin(0.5f * dx * gpu::CAMath::Sqrt(1.f + dy2dx * dy2dx) * crv) / crv; // arc
  step *= gpu::CAMath::Sqrt(1.f + getTgl() * getTgl());
  //
  // get the track x,y,z,px/p,py/p,pz/p,p,sinAlpha,cosAlpha in the Global System
  gpu::gpustd::array<value_t, 9> vecLab{0.f};
  if (!getPosDirGlo(vecLab)) {
    return false;
  }

  // rotate to the system where Bx=By=0.
  value_t bxy2 = b[0] * b[0] + b[1] * b[1];
  value_t bt = gpu::CAMath::Sqrt(bxy2);
  value_t cosphi = 1.f, sinphi = 0.f;
  if (bt > constants::math::Almost0) {
    cosphi = b[0] / bt;
    sinphi = b[1] / bt;
  }
  value_t bb = gpu::CAMath::Sqrt(bxy2 + b[2] * b[2]);
  value_t costet = 1.f, sintet = 0.f;
  if (bb > constants::math::Almost0) {
    costet = b[2] / bb;
    sintet = bt / bb;
  }
  gpu::gpustd::array<value_t, 7> vect{costet * cosphi * vecLab[0] + costet * sinphi * vecLab[1] - sintet * vecLab[2],
                                      -sinphi * vecLab[0] + cosphi * vecLab[1],
                                      sintet * cosphi * vecLab[0] + sintet * sinphi * vecLab[1] + costet * vecLab[2],
                                      costet * cosphi * vecLab[3] + costet * sinphi * vecLab[4] - sintet * vecLab[5],
                                      -sinphi * vecLab[3] + cosphi * vecLab[4],
                                      sintet * cosphi * vecLab[3] + sintet * sinphi * vecLab[4] + costet * vecLab[5],
                                      vecLab[6]};

  // Do the helix step
  value_t q = getCharge();
  g3helx3(q * bb, step, vect);

  // rotate back to the Global System
  vecLab[0] = cosphi * costet * vect[0] - sinphi * vect[1] + cosphi * sintet * vect[2];
  vecLab[1] = sinphi * costet * vect[0] + cosphi * vect[1] + sinphi * sintet * vect[2];
  vecLab[2] = -sintet * vect[0] + costet * vect[2];

  vecLab[3] = cosphi * costet * vect[3] - sinphi * vect[4] + cosphi * sintet * vect[5];
  vecLab[4] = sinphi * costet * vect[3] + cosphi * vect[4] + sinphi * sintet * vect[5];
  vecLab[5] = -sintet * vect[3] + costet * vect[5];

  // rotate back to the Tracking System
  value_t sinalp = -vecLab[7], cosalp = vecLab[8];
  value_t t = cosalp * vecLab[0] - sinalp * vecLab[1];
  vecLab[1] = sinalp * vecLab[0] + cosalp * vecLab[1];
  vecLab[0] = t;
  t = cosalp * vecLab[3] - sinalp * vecLab[4];
  vecLab[4] = sinalp * vecLab[3] + cosalp * vecLab[4];
  vecLab[3] = t;

  // Do the final correcting step to the target plane (linear approximation)
  value_t x = vecLab[0], y = vecLab[1], z = vecLab[2];
  if (gpu::CAMath::Abs(dx) > constants::math::Almost0) {
    if (gpu::CAMath::Abs(vecLab[3]) < constants::math::Almost0) {
      return false;
    }
    dx = xk - vecLab[0];
    x += dx;
    y += vecLab[4] / vecLab[3] * dx;
    z += vecLab[5] / vecLab[3] * dx;
  }

  // Calculate the track parameters
  t = 1.f / gpu::CAMath::Sqrt(vecLab[3] * vecLab[3] + vecLab[4] * vecLab[4]);
  mX = xk;
  mP[kY] = y;
  mP[kZ] = z;
  mP[kSnp] = vecLab[4] * t;
  mP[kTgl] = vecLab[5] * t;
  mP[kQ2Pt] = q * t / vecLab[6];

  return true;
}

//____________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::propagateParamTo(value_t xk, value_t b)
{
  //----------------------------------------------------------------
  // propagate this track to the plane X=xk (cm) in the field "b" (kG)
  // Only parameters are propagated, not the matrix. To be used for small
  // distances only (<mm, i.e. misalignment)
  //----------------------------------------------------------------
  value_t dx = xk - getX();
  if (gpu::CAMath::Abs(dx) < constants::math::Almost0) {
    return true;
  }
  value_t crv = (gpu::CAMath::Abs(b) < constants::math::Almost0) ? 0.f : getCurvature(b);
  value_t x2r = crv * dx;
  value_t f1 = getSnp(), f2 = f1 + x2r;
  if ((gpu::CAMath::Abs(f1) > constants::math::Almost1) || (gpu::CAMath::Abs(f2) > constants::math::Almost1)) {
    return false;
  }
  value_t r1 = gpu::CAMath::Sqrt((1.f - f1) * (1.f + f1));
  if (gpu::CAMath::Abs(r1) < constants::math::Almost0) {
    return false;
  }
  value_t r2 = gpu::CAMath::Sqrt((1.f - f2) * (1.f + f2));
  if (gpu::CAMath::Abs(r2) < constants::math::Almost0) {
    return false;
  }
  double dy2dx = (f1 + f2) / (r1 + r2);
  bool arcz = gpu::CAMath::Abs(x2r) > 0.05f;
  if (arcz) {
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    //    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    //    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    //    track1 += rot/crv*track3;
    //
    auto arg = r1 * f2 - r2 * f1;
    if (gpu::CAMath::Abs(arg) > constants::math::Almost1) {
      return false;
    }
    value_t rot = CAMath::ASin(arg);                // more economic version from Yura.
    if (f1 * f1 + f2 * f2 > 1.f && f1 * f2 < 0.f) { // special cases of large rotations or large abs angles
      if (f2 > 0.f) {
        rot = constants::math::PI - rot; //
      } else {
        rot = -constants::math::PI - rot;
      }
    }
    mP[kZ] += getTgl() / crv * rot;
  } else {
    mP[kZ] += dx * (r2 + f2 * dy2dx) * getTgl();
  }
  mX = xk;
  mP[kY] += dx * dy2dx;
  mP[kSnp] += x2r;
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::propagateParamToDCA(const math_utils::Point3D<value_t>& vtx, value_t b, dim2_t* dca, value_t maxD)
{
  // propagate track to DCA to the vertex
  value_t sn, cs, alp = getAlpha();
  math_utils::detail::sincos(alp, sn, cs);
  value_t x = getX(), y = getY(), snp = getSnp(), csp = gpu::CAMath::Sqrt((1.f - snp) * (1.f + snp));
  value_t xv = vtx.X() * cs + vtx.Y() * sn, yv = -vtx.X() * sn + vtx.Y() * cs, zv = vtx.Z();
  x -= xv;
  y -= yv;
  // Estimate the impact parameter neglecting the track curvature
  value_t d = gpu::CAMath::Abs(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  value_t crv = getCurvature(b);
  value_t tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / gpu::CAMath::Sqrt(1.f + tgfv * tgfv);
  cs = gpu::CAMath::Sqrt((1.f - sn) * (1.f + sn));
  cs = (gpu::CAMath::Abs(tgfv) > constants::math::Almost0) ? sn / tgfv : constants::math::Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  auto tmpT(*this); // operate on the copy to recover after the failure
  alp += gpu::CAMath::ASin(sn);
  if (!tmpT.rotateParam(alp) || !tmpT.propagateParamTo(xv, b)) {
#ifndef GPUCA_ALIGPUCODE
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex "
               << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z() << " | Track is: " << tmpT.asString();
#else
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex " << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z();
#endif
    return false;
  }
  *this = tmpT;
  if (dca) {
    (*dca)[0] = getY() - yv;
    (*dca)[1] = getZ() - zv;
  }
  return true;
}

//____________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::getYZAt(value_t xk, value_t b, value_t& y, value_t& z) const
{
  //----------------------------------------------------------------
  // estimate Y,Z in tracking frame at given X
  //----------------------------------------------------------------
  value_t dx = xk - getX();
  y = mP[kY];
  z = mP[kZ];
  if (gpu::CAMath::Abs(dx) < constants::math::Almost0) {
    return true;
  }
  value_t crv = getCurvature(b);
  value_t x2r = crv * dx;
  value_t f1 = getSnp(), f2 = f1 + x2r;
  if ((gpu::CAMath::Abs(f1) > constants::math::Almost1) || (gpu::CAMath::Abs(f2) > constants::math::Almost1)) {
    return false;
  }
  value_t r1 = gpu::CAMath::Sqrt((1.f - f1) * (1.f + f1));
  if (gpu::CAMath::Abs(r1) < constants::math::Almost0) {
    return false;
  }
  value_t r2 = gpu::CAMath::Sqrt((1.f - f2) * (1.f + f2));
  if (gpu::CAMath::Abs(r2) < constants::math::Almost0) {
    return false;
  }
  double dy2dx = (f1 + f2) / (r1 + r2);
  y += dx * dy2dx;
  if (gpu::CAMath::Abs(x2r) < 0.05f) {
    z += dx * (r2 + f2 * dy2dx) * getTgl();
  } else {
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    //    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    //    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    //    track1 += rot/crv*track3;
    //
    value_t rot = CAMath::ASin(r1 * f2 - r2 * f1);  // more economic version from Yura.
    if (f1 * f1 + f2 * f2 > 1.f && f1 * f2 < 0.f) { // special cases of large rotations or large abs angles
      if (f2 > 0.f) {
        rot = constants::math::PI - rot; //
      } else {
        rot = -constants::math::PI - rot;
      }
    }
    z += getTgl() / crv * rot;
  }
  return true;
}

//______________________________________________________________
template <typename value_T>
GPUd() void TrackParametrization<value_T>::invertParam()
{
  // Transform this track to the local coord. system rotated by 180 deg.
  mX = -mX;
  mAlpha += constants::math::PI;
  math_utils::detail::bringToPMPi<value_t>(mAlpha);
  //
  mP[0] = -mP[0];
  mP[3] = -mP[3];
  mP[4] = -mP[4];
  //
}

//______________________________________________________________
template <typename value_T>
GPUd() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getZAt(value_t xk, value_t b) const
{
  ///< this method is just an alias for obtaining Z @ X in the tree->Draw()
  value_t y, z;
  return getYZAt(xk, b, y, z) ? z : -9999.f;
}

//______________________________________________________________
template <typename value_T>
GPUd() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getYAt(value_t xk, value_t b) const
{
  ///< this method is just an alias for obtaining Z @ X in the tree->Draw()
  value_t y, z;
  return getYZAt(xk, b, y, z) ? y : -9999.f;
}

//______________________________________________________________
template <typename value_T>
GPUd() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getSnpAt(value_t xk, value_t b) const
{
  ///< this method is just an alias for obtaining snp @ X in the tree->Draw()
  value_t dx = xk - getX();
  if (gpu::CAMath::Abs(dx) < constants::math::Almost0) {
    return getSnp();
  }
  value_t crv = (gpu::CAMath::Abs(b) < constants::math::Almost0) ? 0.f : getCurvature(b);
  value_t x2r = crv * dx;
  return mP[kSnp] + x2r;
}

//______________________________________________________________
template <typename value_T>
GPUd() typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getSnpAt(value_t alpha, value_t xk, value_t b) const
{
  ///< this method is just an alias for obtaining snp @ alpha, X in the tree->Draw()
  math_utils::detail::bringToPMPi<value_t>(alpha);
  value_t ca = 0, sa = 0;
  math_utils::detail::sincos(alpha - getAlpha(), sa, ca);
  value_t snp = getSnp(), csp = gpu::CAMath::Sqrt((1.f - snp) * (1.f + snp)); // Improve precision
  // RS: check if rotation does no invalidate track model (cos(local_phi)>=0, i.e. particle direction in local frame is along the X axis
  if ((csp * ca + snp * sa) < 0.) {
    // LOGF(warning,"Rotation failed: local cos(phi) would become {:.2f}", csp * ca + snp * sa);
    return -999;
  }
  value_t tmp = snp * ca - csp * sa;
  if (gpu::CAMath::Abs(tmp) > constants::math::Almost1) {
    LOGP(debug, "Rotation failed: new snp {:.2f}", tmp);
    return -999;
  }
  value_t xrot = getX() * ca + getY() * sa;
  value_t dx = xk - xrot;
  value_t crv = (gpu::CAMath::Abs(b) < constants::math::Almost0) ? 0.f : getCurvature(b);
  value_t x2r = crv * dx;
  return tmp + x2r;
}

#ifndef GPUCA_ALIGPUCODE
//_____________________________________________________________
template <typename value_T>
std::string TrackParametrization<value_T>::asString() const
{
  // print parameters as string
  return fmt::format("X:{:+.4e} Alp:{:+.3e} Par: {:+.4e} {:+.4e} {:+.4e} {:+.4e} {:+.4e} |Q|:{:d} {:s}",
                     getX(), getAlpha(), getY(), getZ(), getSnp(), getTgl(), getQ2Pt(), getAbsCharge(), getPID().getName());
}
#endif

//______________________________________________________________
template <typename value_T>
GPUd() void TrackParametrization<value_T>::printParam() const
{
  // print parameters
#ifndef GPUCA_ALIGPUCODE
  printf("%s\n", asString().c_str());
#else
  printf("X:%+.4e Alp:%+.3e Par: %+.4e %+.4e %+.4e %+.4e %+.4e |Q|:%d",
         getX(), getAlpha(), getY(), getZ(), getSnp(), getTgl(), getQ2Pt(), getAbsCharge());
#endif
}

//______________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::getXatLabR(value_t r, value_t& x, value_t bz, track::DirType dir) const
{
  // Get local X of the track position estimated at the radius lab radius r.
  // The track curvature is accounted exactly
  //
  // The flag "dir" can be used to remove the ambiguity of which intersection to take (out of 2 possible)
  // DirAuto (==0)  - take the intersection closest to the current track position
  // DirOutward (==1) - go along the track (increasing mX)
  // DirInward (==-1) - go backward (decreasing mX)
  //
  const auto fy = mP[0], sn = mP[2];
  const value_t kEps = 1.e-6;
  //
  auto crv = getCurvature(bz);
  while (gpu::CAMath::Abs(crv) > constants::math::Almost0) { // helix ?
    // get center of the track circle
    math_utils::CircleXY<value_t> circle{};
    getCircleParamsLoc(bz, circle);
    if (circle.rC == 0.) {
      crv = 0.;
      break;
    }
    value_t r0 = gpu::CAMath::Sqrt(circle.getCenterD2());
    if (r0 <= constants::math::Almost0) {
      return false; // the track is concentric to circle
    }
    value_t tR2r0 = 1.f, g = 0.f, tmp = 0.f;
    if (gpu::CAMath::Abs(circle.rC - r0) > kEps) {
      tR2r0 = circle.rC / r0;
      g = 0.5f * (r * r / (r0 * circle.rC) - tR2r0 - 1.f / tR2r0);
      tmp = 1.f + g * tR2r0;
    } else {
      tR2r0 = 1.0;
      g = 0.5f * r * r / (r0 * circle.rC) - 1.f;
      tmp = 0.5f * r * r / (r0 * r0);
    }
    value_t det = (1.f - g) * (1.f + g);
    if (det < 0.f) {
      return false; // does not reach raduis r
    }
    det = gpu::CAMath::Sqrt(det);
    //
    // the intersection happens in 2 points: {circle.xC+tR*C,circle.yC+tR*S}
    // with C=f*c0+-|s0|*det and S=f*s0-+c0 sign(s0)*det
    // where s0 and c0 make direction for the circle center (=circle.xC/r0 and circle.yC/r0)
    //
    x = circle.xC * tmp;
    value_t y = circle.yC * tmp;
    if (gpu::CAMath::Abs(circle.yC) > constants::math::Almost0) { // when circle.yC==0 the x,y is unique
      value_t dfx = tR2r0 * gpu::CAMath::Abs(circle.yC) * det;
      value_t dfy = tR2r0 * circle.xC * (circle.yC > 0.f ? det : -det);
      if (dir == DirAuto) {                              // chose the one which corresponds to smallest step
        value_t delta = (x - mX) * dfx - (y - fy) * dfy; // the choice of + in C will lead to smaller step if delta<0
        x += delta < 0.f ? dfx : -dfx;
      } else if (dir == DirOutward) { // along track direction: x must be > mX
        x -= dfx;                     // try the smallest step (dfx is positive)
        value_t dfeps = mX - x;       // handle special case of very small step
        if (dfeps < -kEps) {
          return true;
        }
        if (gpu::CAMath::Abs(dfeps) < kEps && gpu::CAMath::Abs(mX * mX + fy * fy - r * r) < kEps) { // are we already in right r?
          return mX;
        }
        x += dfx + dfx;
        value_t dxm = x - mX;
        if (dxm > 0.f) {
          return true;
        } else if (dxm < -kEps) {
          return false;
        }
        x = mX;                 // don't move
      } else {                  // backward: x must be < mX
        x += dfx;               // try the smallest step (dfx is positive)
        value_t dfeps = x - mX; // handle special case of very small step
        if (dfeps < -kEps) {
          return true;
        }
        if (gpu::CAMath::Abs(dfeps) < kEps && gpu::CAMath::Abs(mX * mX + fy * fy - r * r) < kEps) { // are we already in right r?
          return mX;
        }
        x -= dfx + dfx;
        value_t dxm = x - mX;
        if (dxm < 0.f) {
          return true;
        }
        if (dxm > kEps) {
          return false;
        }
        x = mX; // don't move
      }
    } else { // special case: track touching the circle just in 1 point
      if ((dir == DirOutward && x < mX) || (dir == DirInward && x > mX)) {
        return false;
      }
    }
    return x;
  }
  // this is a straight track
  if (gpu::CAMath::Abs(sn) >= constants::math::Almost1) { // || to Y axis
    value_t det = (r - mX) * (r + mX);
    if (det < 0.f) {
      return false; // does not reach raduis r
    }
    x = mX;
    if (dir == DirAuto) {
      return true;
    }
    det = gpu::CAMath::Sqrt(det);
    if (dir == DirOutward) { // along the track direction
      if (sn > 0.f) {
        if (fy > det) {
          return false; // track is along Y axis and above the circle
        }
      } else {
        if (fy < -det) {
          return false; // track is against Y axis amd belo the circle
        }
      }
    } else if (dir == DirInward) { // against track direction
      if (sn > 0.f) {
        if (fy < -det) {
          return false; // track is along Y axis
        }
      } else if (fy > det) {
        return false; // track is against Y axis
      }
    }
  } else if (gpu::CAMath::Abs(sn) <= constants::math::Almost0) { // || to X axis
    value_t det = (r - fy) * (r + fy);
    if (det < 0.f) {
      return false; // does not reach raduis r
    }
    det = gpu::CAMath::Sqrt(det);
    if (dir == DirAuto) {
      x = mX > 0.f ? det : -det; // choose the solution requiring the smalest step
      return true;
    } else if (dir == DirOutward) { // along the track direction
      if (mX > det) {
        return false; // current point is in on the right from the circle
      } else {
        x = (mX < -det) ? -det : det; // on the left : within the circle
      }
    } else { // against the track direction
      if (mX < -det) {
        return false;
      } else {
        x = mX > det ? det : -det;
      }
    }
  } else { // general case of straight line
    value_t cs = gpu::CAMath::Sqrt((1.f - sn) * (1.f + sn));
    value_t xsyc = mX * sn - fy * cs;
    value_t det = (r - xsyc) * (r + xsyc);
    if (det < 0.f) {
      return false; // does not reach raduis r
    }
    det = gpu::CAMath::Sqrt(det);
    value_t xcys = mX * cs + fy * sn;
    value_t t = -xcys;
    if (dir == DirAuto) {
      t += t > 0.f ? -det : det; // chose the solution requiring the smalest step
    } else if (dir > 0) {        // go in increasing mX direction. ( t+-det > 0)
      if (t >= -det) {
        t += det; // take minimal step giving t>0
      } else {
        return false; // both solutions have negative t
      }
    } else { // go in decreasing mX direction. (t+-det < 0)
      if (t < det) {
        t -= det; // take minimal step giving t<0
      } else {
        return false; // both solutions have positive t
      }
    }
    x = mX + cs * t;
  }
  //
  return true;
}

//______________________________________________
template <typename value_T>
GPUd() bool TrackParametrization<value_T>::correctForELoss(value_t xrho, bool anglecorr, value_t dedx)
{
  //------------------------------------------------------------------
  // This function corrects the track parameters for the energy loss in crossed material.
  // "xrho" - is the product length*density (g/cm^2).
  //     It should be passed as negative when propagating tracks
  //     from the intreaction point to the outside of the central barrel.
  // "dedx" - mean enery loss (GeV/(g/cm^2), if <=kCalcdEdxAuto : calculate on the fly
  // "anglecorr" - switch for the angular correction
  //------------------------------------------------------------------
  constexpr value_t kMaxELossFrac = 0.3f; // max allowed fractional eloss
  constexpr value_t kMinP = 0.01f;        // kill below this momentum

  // Apply angle correction, if requested
  if (anglecorr) {
    value_t csp2 = (1.f - getSnp()) * (1.f + getSnp()); // cos(phi)^2
    value_t cst2I = (1.f + getTgl() * getTgl());        // 1/cos(lambda)^2
    value_t angle = gpu::CAMath::Sqrt(cst2I / (csp2));
    xrho *= angle;
  }
  value_t p = getP();
  value_t p2 = p * p;
  value_t e2 = p2 + getPID().getMass2();
  value_t beta2 = p2 / e2;

  // Calculating the energy loss corrections************************
  if ((xrho != 0.f) && (beta2 < 1.f)) {
    if (dedx < kCalcdEdxAuto + constants::math::Almost1) { // request to calculate dedx on the fly
      dedx = BetheBlochSolid(p / getPID().getMass());
      if (mAbsCharge != 1) {
        dedx *= mAbsCharge * mAbsCharge;
      }
    }

    value_t dE = dedx * xrho;
    value_t e = gpu::CAMath::Sqrt(e2);
    if (gpu::CAMath::Abs(dE) > kMaxELossFrac * e) {
      return false; // 30% energy loss is too much!
    }
    value_t eupd = e + dE;
    value_t pupd2 = eupd * eupd - getPID().getMass2();
    if (pupd2 < kMinP * kMinP) {
      return false;
    }
    setQ2Pt(getQ2Pt() * p / gpu::CAMath::Sqrt(pupd2));
  }

  return true;
}

//______________________________________________
template <typename value_T>
GPUd() typename o2::track::TrackParametrization<value_T>::yzerr_t TrackParametrization<value_T>::getVertexInTrackFrame(const o2::dataformats::VertexBase& v) const
{
  // rotate vertex to track frame and return parameters used by getPredictedChi2 and update of TrackParametrizationWithError
  value_t sn, cs;
  math_utils::detail::sincos(-mAlpha, sn, cs); // use -alpha since we rotate from lab to tracking frame
  value_t sn2 = sn * sn, cs2 = cs * cs, sncs = sn * cs;
  value_t dsxysncs = 2. * v.getSigmaXY() * sncs;
  return {{/*v.getX()*cs-v.getY()*sn,*/ v.getX() * sn + v.getY() * cs, v.getZ()},
          {v.getSigmaX2() * sn2 + dsxysncs + v.getSigmaY2() * cs2, (sn + cs) * v.getSigmaYZ(), v.getSigmaZ2()}};
}

namespace o2::track
{
template class TrackParametrization<float>;
#ifndef GPUCA_GPUCODE_DEVICE
template class TrackParametrization<double>;
#endif
} // namespace o2::track
