// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackParametrization.cxx
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 1, 2020
/// @brief

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "ReconstructionDataFormats/DCA.h"
#include <FairLogger.h>
#include <iostream>
#include "Math/SMatrix.h"
#include <fmt/printf.h>
#include "Framework/Logger.h"

using std::array;
using namespace o2::constants::math;

namespace o2
{
namespace track
{

//______________________________________________________________
template <typename value_T>
TrackParametrization<value_T>::TrackParametrization(const array<value_t, 3>& xyz, const array<value_t, 3>& pxpypz, int charge, bool sectorAlpha)
  : mX{0.f}, mAlpha{0.f}, mP{0.f}
{
  // construct track param from kinematics

  // Alpha of the frame is defined as:
  // sectorAlpha == false : -> angle of pt direction
  // sectorAlpha == true  : -> angle of the sector from X,Y coordinate for r>1
  //                           angle of pt direction for r==0
  //
  //
  constexpr value_t kSafe = 1e-5f;
  value_t radPos2 = xyz[0] * xyz[0] + xyz[1] * xyz[1];
  value_t alp = 0;
  if (sectorAlpha || radPos2 < 1) {
    alp = atan2f(pxpypz[1], pxpypz[0]);
  } else {
    alp = atan2f(xyz[1], xyz[0]);
  }
  if (sectorAlpha) {
    alp = utils::Angle2Alpha(alp);
  }
  //
  value_t sn, cs;
  utils::sincos(alp, sn, cs);
  // protection:  avoid alpha being too close to 0 or +-pi/2
  if (fabs(sn) < 2 * kSafe) {
    if (alp > 0) {
      alp += alp < PIHalf ? 2 * kSafe : -2 * kSafe;
    } else {
      alp += alp > -PIHalf ? -2 * kSafe : 2 * kSafe;
    }
    utils::sincos(alp, sn, cs);
  } else if (fabs(cs) < 2 * kSafe) {
    if (alp > 0) {
      alp += alp > PIHalf ? 2 * kSafe : -2 * kSafe;
    } else {
      alp += alp > -PIHalf ? 2 * kSafe : -2 * kSafe;
    }
    utils::sincos(alp, sn, cs);
  }
  // get the vertex of origin and the momentum
  array<value_t, 3> ver{xyz[0], xyz[1], xyz[2]};
  array<value_t, 3> mom{pxpypz[0], pxpypz[1], pxpypz[2]};
  //
  // Rotate to the local coordinate system
  utils::RotateZ(ver, -alp);
  utils::RotateZ(mom, -alp);
  //
  value_t ptI = 1.f / sqrt(mom[0] * mom[0] + mom[1] * mom[1]);
  mX = ver[0];
  mAlpha = alp;
  mP[kY] = ver[1];
  mP[kZ] = ver[2];
  mP[kSnp] = mom[1] * ptI;
  mP[kTgl] = mom[2] * ptI;
  mAbsCharge = std::abs(charge);
  mP[kQ2Pt] = charge ? ptI * charge : ptI;
  //
  if (fabs(1 - getSnp()) < kSafe) {
    mP[kSnp] = 1. - kSafe; // Protection
  } else if (fabs(-1 - getSnp()) < kSafe) {
    mP[kSnp] = -1. + kSafe; // Protection
  }
  //
}

//_______________________________________________________
template <typename value_T>
bool TrackParametrization<value_T>::getPxPyPzGlo(array<value_t, 3>& pxyz) const
{
  // track momentum
  if (fabs(getQ2Pt()) < Almost0 || fabs(getSnp()) > Almost1) {
    return false;
  }
  value_t cs, sn, pt = getPt();
  value_t r = std::sqrt((1.f - getSnp()) * (1.f + getSnp()));
  utils::sincos(getAlpha(), sn, cs);
  pxyz[0] = pt * (r * cs - getSnp() * sn);
  pxyz[1] = pt * (getSnp() * cs + r * sn);
  pxyz[2] = pt * getTgl();
  return true;
}

//____________________________________________________
template <typename value_T>
bool TrackParametrization<value_T>::getPosDirGlo(array<value_t, 9>& posdirp) const
{
  // fill vector with lab x,y,z,px/p,py/p,pz/p,p,sinAlpha,cosAlpha
  value_t ptI = fabs(getQ2Pt());
  value_t snp = getSnp();
  if (ptI < Almost0 || fabs(snp) > Almost1) {
    return false;
  }
  value_t &sn = posdirp[7], &cs = posdirp[8];
  value_t csp = std::sqrt((1.f - snp) * (1.f + snp));
  value_t cstht = std::sqrt(1.f + getTgl() * getTgl());
  value_t csthti = 1.f / cstht;
  utils::sincos(getAlpha(), sn, cs);
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
bool TrackParametrization<value_T>::rotateParam(value_t alpha)
{
  // rotate to alpha frame
  if (fabs(getSnp()) > Almost1) {
    LOGF(WARNING, "Precondition is not satisfied: |sin(phi)|>1 ! {:f}", getSnp());
    return false;
  }
  //
  utils::BringToPMPi(alpha);
  //
  value_t ca = 0, sa = 0;
  utils::sincos(alpha - getAlpha(), sa, ca);
  value_t snp = getSnp(), csp = std::sqrt((1.f - snp) * (1.f + snp)); // Improve precision
  // RS: check if rotation does no invalidate track model (cos(local_phi)>=0, i.e. particle
  // direction in local frame is along the X axis
  if ((csp * ca + snp * sa) < 0) {
    //LOGF(WARNING,"Rotation failed: local cos(phi) would become {:.2f}", csp * ca + snp * sa);
    return false;
  }
  //
  value_t tmp = snp * ca - csp * sa;
  if (fabs(tmp) > Almost1) {
    LOGF(WARNING, "Rotation failed: new snp {:.2f}", tmp);
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
bool TrackParametrization<value_T>::propagateParamTo(value_t xk, const array<value_t, 3>& b)
{
  //----------------------------------------------------------------
  // Extrapolate this track params (w/o cov matrix) to the plane X=xk in the field b[].
  //
  // X [cm] is in the "tracking coordinate system" of this track.
  // b[]={Bx,By,Bz} [kG] is in the Global coordidate system.
  //----------------------------------------------------------------

  value_t dx = xk - getX();
  if (fabs(dx) < Almost0) {
    return true;
  }
  // Do not propagate tracks outside the ALICE detector
  if (fabs(dx) > 1e5 || fabs(getY()) > 1e5 || fabs(getZ()) > 1e5) {
    LOGF(WARNING, "Anomalous track, target X:{:f}", xk);
    //    print();
    return false;
  }
  value_t crv = getCurvature(b[2]);
  value_t x2r = crv * dx;
  value_t f1 = getSnp(), f2 = f1 + x2r;
  if (fabs(f1) > Almost1 || fabs(f2) > Almost1) {
    return false;
  }
  value_t r1 = std::sqrt((1.f - f1) * (1.f + f1));
  if (fabs(r1) < Almost0) {
    return false;
  }
  value_t r2 = std::sqrt((1.f - f2) * (1.f + f2));
  if (fabs(r2) < Almost0) {
    return false;
  }
  value_t dy2dx = (f1 + f2) / (r1 + r2);
  value_t step = (fabs(x2r) < 0.05f) ? dx * fabs(r2 + f2 * dy2dx)                                           // chord
                                     : 2.f * asinf(0.5f * dx * std::sqrt(1.f + dy2dx * dy2dx) * crv) / crv; // arc
  step *= std::sqrt(1.f + getTgl() * getTgl());
  //
  // get the track x,y,z,px/p,py/p,pz/p,p,sinAlpha,cosAlpha in the Global System
  array<value_t, 9> vecLab{0.f};
  if (!getPosDirGlo(vecLab)) {
    return false;
  }

  // rotate to the system where Bx=By=0.
  value_t bxy2 = b[0] * b[0] + b[1] * b[1];
  value_t bt = std::sqrt(bxy2);
  value_t cosphi = 1.f, sinphi = 0.f;
  if (bt > Almost0) {
    cosphi = b[0] / bt;
    sinphi = b[1] / bt;
  }
  value_t bb = std::sqrt(bxy2 + b[2] * b[2]);
  value_t costet = 1., sintet = 0.;
  if (bb > Almost0) {
    costet = b[2] / bb;
    sintet = bt / bb;
  }
  array<value_t, 7> vect{costet * cosphi * vecLab[0] + costet * sinphi * vecLab[1] - sintet * vecLab[2],
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
  if (fabs(dx) > Almost0) {
    if (fabs(vecLab[3]) < Almost0) {
      return false;
    }
    dx = xk - vecLab[0];
    x += dx;
    y += vecLab[4] / vecLab[3] * dx;
    z += vecLab[5] / vecLab[3] * dx;
  }

  // Calculate the track parameters
  t = 1.f / std::sqrt(vecLab[3] * vecLab[3] + vecLab[4] * vecLab[4]);
  mX = x;
  mP[kY] = y;
  mP[kZ] = z;
  mP[kSnp] = vecLab[4] * t;
  mP[kTgl] = vecLab[5] * t;
  mP[kQ2Pt] = q * t / vecLab[6];

  return true;
}

//____________________________________________________________
template <typename value_T>
bool TrackParametrization<value_T>::propagateParamTo(value_t xk, value_t b)
{
  //----------------------------------------------------------------
  // propagate this track to the plane X=xk (cm) in the field "b" (kG)
  // Only parameters are propagated, not the matrix. To be used for small
  // distances only (<mm, i.e. misalignment)
  //----------------------------------------------------------------
  value_t dx = xk - getX();
  if (fabs(dx) < Almost0) {
    return true;
  }
  value_t crv = (fabs(b) < Almost0) ? 0.f : getCurvature(b);
  value_t x2r = crv * dx;
  value_t f1 = getSnp(), f2 = f1 + x2r;
  if ((fabs(f1) > Almost1) || (fabs(f2) > Almost1)) {
    return false;
  }
  value_t r1 = std::sqrt((1.f - f1) * (1.f + f1));
  if (fabs(r1) < Almost0) {
    return false;
  }
  value_t r2 = std::sqrt((1.f - f2) * (1.f + f2));
  if (fabs(r2) < Almost0) {
    return false;
  }
  mX = xk;
  double dy2dx = (f1 + f2) / (r1 + r2);
  mP[kY] += dx * dy2dx;
  mP[kSnp] += x2r;
  if (fabs(x2r) < 0.05f) {
    mP[kZ] += dx * (r2 + f2 * dy2dx) * getTgl();
  } else {
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    //    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    //    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    //    track1 += rot/crv*track3;
    //
    value_t rot = asinf(r1 * f2 - r2 * f1);         // more economic version from Yura.
    if (f1 * f1 + f2 * f2 > 1.f && f1 * f2 < 0.f) { // special cases of large rotations or large abs angles
      if (f2 > 0.f) {
        rot = PI - rot; //
      } else {
        rot = -PI - rot;
      }
    }
    mP[kZ] += getTgl() / crv * rot;
  }
  return true;
}

//_______________________________________________________________________
template <typename value_T>
bool TrackParametrization<value_T>::propagateParamToDCA(const Point3D<value_t>& vtx, value_t b, std::array<value_t, 2>* dca, value_t maxD)
{
  // propagate track to DCA to the vertex
  value_t sn, cs, alp = getAlpha();
  o2::utils::sincos(alp, sn, cs);
  value_t x = getX(), y = getY(), snp = getSnp(), csp = std::sqrt((1.f - snp) * (1.f + snp));
  value_t xv = vtx.X() * cs + vtx.Y() * sn, yv = -vtx.X() * sn + vtx.Y() * cs, zv = vtx.Z();
  x -= xv;
  y -= yv;
  //Estimate the impact parameter neglecting the track curvature
  Double_t d = std::abs(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  value_t crv = getCurvature(b);
  value_t tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / std::sqrt(1.f + tgfv * tgfv);
  cs = std::sqrt((1. - sn) * (1. + sn));
  cs = (std::abs(tgfv) > Almost0) ? sn / tgfv : Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  auto tmpT(*this); // operate on the copy to recover after the failure
  alp += std::asin(sn);
  if (!tmpT.rotateParam(alp) || !tmpT.propagateParamTo(xv, b)) {
    LOG(WARNING) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex "
                 << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z() << " | Track is: ";
    tmpT.printParam();
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
bool TrackParametrization<value_T>::getYZAt(value_t xk, value_t b, value_t& y, value_t& z) const
{
  //----------------------------------------------------------------
  // estimate Y,Z in tracking frame at given X
  //----------------------------------------------------------------
  value_t dx = xk - getX();
  if (fabs(dx) < Almost0) {
    return true;
  }
  value_t crv = getCurvature(b);
  value_t x2r = crv * dx;
  value_t f1 = getSnp(), f2 = f1 + x2r;
  if ((fabs(f1) > Almost1) || (fabs(f2) > Almost1)) {
    return false;
  }
  value_t r1 = std::sqrt((1.f - f1) * (1.f + f1));
  if (fabs(r1) < Almost0) {
    return false;
  }
  value_t r2 = std::sqrt((1.f - f2) * (1.f + f2));
  if (fabs(r2) < Almost0) {
    return false;
  }
  double dy2dx = (f1 + f2) / (r1 + r2);
  y = mP[kY] + dx * dy2dx;
  z = mP[kZ];
  if (fabs(x2r) < 0.05f) {
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
    value_t rot = asinf(r1 * f2 - r2 * f1);         // more economic version from Yura.
    if (f1 * f1 + f2 * f2 > 1.f && f1 * f2 < 0.f) { // special cases of large rotations or large abs angles
      if (f2 > 0.f) {
        rot = PI - rot; //
      } else {
        rot = -PI - rot;
      }
    }
    z += getTgl() / crv * rot;
  }
  return true;
}

//______________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::invertParam()
{
  // Transform this track to the local coord. system rotated by 180 deg.
  mX = -mX;
  mAlpha += PI;
  utils::BringToPMPi(mAlpha);
  //
  mP[0] = -mP[0];
  mP[3] = -mP[3];
  mP[4] = -mP[4];
  //
}

//______________________________________________________________
template <typename value_T>
typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getZAt(value_t xk, value_t b) const
{
  ///< this method is just an alias for obtaining Z @ X in the tree->Draw()
  value_t y, z;
  return getYZAt(xk, b, y, z) ? z : -9999.;
}

//______________________________________________________________
template <typename value_T>
typename TrackParametrization<value_T>::value_t TrackParametrization<value_T>::getYAt(value_t xk, value_t b) const
{
  ///< this method is just an alias for obtaining Z @ X in the tree->Draw()
  value_t y, z;
  return getYZAt(xk, b, y, z) ? y : -9999.;
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

//______________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::printParam() const
{
  // print parameters
  printf("%s\n", asString().c_str());
}
#endif

//______________________________________________________________
template <typename value_T>
bool TrackParametrization<value_T>::getXatLabR(value_t r, value_t& x, value_t bz, o2::track::DirType dir) const
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
  if (fabs(crv) > o2::constants::math::Almost0) { // helix
    // get center of the track circle
    o2::utils::CircleXY circle;
    getCircleParamsLoc(bz, circle);
    auto r0 = std::sqrt(circle.getCenterD2());
    if (r0 <= o2::constants::math::Almost0) {
      return false; // the track is concentric to circle
    }
    value_t tR2r0 = 1., g = 0., tmp = 0.;
    if (fabs(circle.rC - r0) > kEps) {
      tR2r0 = circle.rC / r0;
      g = 0.5 * (r * r / (r0 * circle.rC) - tR2r0 - 1. / tR2r0);
      tmp = 1. + g * tR2r0;
    } else {
      tR2r0 = 1.0;
      g = 0.5 * r * r / (r0 * circle.rC) - 1.;
      tmp = 0.5 * r * r / (r0 * r0);
    }
    auto det = (1. - g) * (1. + g);
    if (det < 0.) {
      return false; // does not reach raduis r
    }
    det = std::sqrt(det);
    //
    // the intersection happens in 2 points: {circle.xC+tR*C,circle.yC+tR*S}
    // with C=f*c0+-|s0|*det and S=f*s0-+c0 sign(s0)*det
    // where s0 and c0 make direction for the circle center (=circle.xC/r0 and circle.yC/r0)
    //
    x = circle.xC * tmp;
    auto y = circle.yC * tmp;
    if (fabs(circle.yC) > o2::constants::math::Almost0) { // when circle.yC==0 the x,y is unique
      auto dfx = tR2r0 * fabs(circle.yC) * det;
      auto dfy = tR2r0 * circle.xC * (circle.yC > 0. ? det : -det);
      if (dir == DirAuto) {                           // chose the one which corresponds to smallest step
        auto delta = (x - mX) * dfx - (y - fy) * dfy; // the choice of + in C will lead to smaller step if delta<0
        x += delta < 0. ? dfx : -dfx;
      } else if (dir == DirOutward) { // along track direction: x must be > mX
        x -= dfx;                     // try the smallest step (dfx is positive)
        auto dfeps = mX - x;          // handle special case of very small step
        if (dfeps < -kEps) {
          return true;
        }
        if (fabs(dfeps) < kEps && fabs(mX * mX + fy * fy - r * r) < kEps) { // are we already in right r?
          return mX;
        }
        x += dfx + dfx;
        auto dxm = x - mX;
        if (dxm > 0.) {
          return true;
        } else if (dxm < -kEps) {
          return false;
        }
        x = mX;              // don't move
      } else {               // backward: x must be < mX
        x += dfx;            // try the smallest step (dfx is positive)
        auto dfeps = x - mX; // handle special case of very small step
        if (dfeps < -kEps) {
          return true;
        }
        if (fabs(dfeps) < kEps && fabs(mX * mX + fy * fy - r * r) < kEps) { // are we already in right r?
          return mX;
        }
        x -= dfx + dfx;
        auto dxm = x - mX;
        if (dxm < 0.) {
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
  } else {                                          // this is a straight track
    if (fabs(sn) >= o2::constants::math::Almost1) { // || to Y axis
      auto det = (r - mX) * (r + mX);
      if (det < 0.) {
        return false; // does not reach raduis r
      }
      x = mX;
      if (dir == DirAuto) {
        return true;
      }
      det = std::sqrt(det);
      if (dir == DirOutward) { // along the track direction
        if (sn > 0.) {
          if (fy > det) {
            return false; // track is along Y axis and above the circle
          }
        } else {
          if (fy < -det) {
            return false; // track is against Y axis amd belo the circle
          }
        }
      } else if (dir == DirInward) { // against track direction
        if (sn > 0.) {
          if (fy < -det) {
            return false; // track is along Y axis
          }
        } else if (fy > det) {
          return false; // track is against Y axis
        }
      }
    } else if (fabs(sn) <= o2::constants::math::Almost0) { // || to X axis
      auto det = (r - fy) * (r + fy);
      if (det < 0.) {
        return false; // does not reach raduis r
      }
      det = std::sqrt(det);
      if (dir == DirAuto) {
        x = mX > 0. ? det : -det; // choose the solution requiring the smalest step
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
      auto cs = std::sqrt((1 - sn) * (1 + sn));
      auto xsyc = mX * sn - fy * cs;
      auto det = (r - xsyc) * (r + xsyc);
      if (det < 0.) {
        return false; // does not reach raduis r
      }
      det = std::sqrt(det);
      auto xcys = mX * cs + fy * sn;
      auto t = -xcys;
      if (dir == DirAuto) {
        t += t > 0. ? -det : det; // chose the solution requiring the smalest step
      } else if (dir > 0) {       // go in increasing mX direction. ( t+-det > 0)
        if (t >= -det) {
          t += -det; // take minimal step giving t>0
        } else {
          return false; // both solutions have negative t
        }
      } else { // go in increasing mX direction. (t+-det < 0)
        if (t < det) {
          t -= det; // take minimal step giving t<0
        } else {
          return false; // both solutions have positive t
        }
      }
      x = mX + cs * t;
    }
  }
  //
  return true;
}

//______________________________________________
template <typename value_T>
bool TrackParametrization<value_T>::correctForELoss(value_t xrho, value_t mass, bool anglecorr, value_t dedx)
{
  //------------------------------------------------------------------
  // This function corrects the track parameters for the energy loss in crossed material.
  // "xrho" - is the product length*density (g/cm^2).
  //     It should be passed as negative when propagating tracks
  //     from the intreaction point to the outside of the central barrel.
  // "mass" - the mass of this particle (GeV/c^2).
  // "dedx" - mean enery loss (GeV/(g/cm^2), if <=kCalcdEdxAuto : calculate on the fly
  // "anglecorr" - switch for the angular correction
  //------------------------------------------------------------------
  constexpr value_t kMaxELossFrac = 0.3f; // max allowed fractional eloss
  constexpr value_t kMinP = 0.01f;        // kill below this momentum

  // Apply angle correction, if requested
  if (anglecorr) {
    value_t csp2 = (1.f - getSnp()) * (1.f + getSnp()); // cos(phi)^2
    value_t cst2I = (1.f + getTgl() * getTgl());        // 1/cos(lambda)^2
    value_t angle = std::sqrt(cst2I / (csp2));
    xrho *= angle;
  }

  value_t p = getP();
  if (mass < 0) {
    p += p; // q=2 particle
  }
  value_t p2 = p * p, mass2 = mass * mass;
  value_t e2 = p2 + mass2;
  value_t beta2 = p2 / e2;

  // Calculating the energy loss corrections************************
  if ((xrho != 0.f) && (beta2 < 1.f)) {
    if (dedx < kCalcdEdxAuto + Almost1) { // request to calculate dedx on the fly
      dedx = BetheBlochSolid(p / fabs(mass));
      if (mAbsCharge != 1) {
        dedx *= mAbsCharge * mAbsCharge;
      }
    }

    value_t dE = dedx * xrho;
    value_t e = std::sqrt(e2);
    if (fabs(dE) > kMaxELossFrac * e) {
      return false; // 30% energy loss is too much!
    }
    value_t eupd = e + dE;
    value_t pupd2 = eupd * eupd - mass2;
    if (pupd2 < kMinP * kMinP) {
      return false;
    }
    setQ2Pt(getQ2Pt() * p / std::sqrt(pupd2));
  }

  return true;
}

template class TrackParametrization<float>;
//template class TrackParametrization<double>;

} // namespace track
} // namespace o2
