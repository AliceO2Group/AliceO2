// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackparametrizationWithError.cxx
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

#include "ReconstructionDataFormats/TrackParametrizationWithError.h"
#include "ReconstructionDataFormats/Vertex.h"
#include "ReconstructionDataFormats/DCA.h"
#include <GPUCommonLogger.h>

#ifndef GPUCA_GPUCODE_DEVICE
#include <iostream>
#ifndef GPUCA_STANDALONE
#include "Math/SMatrix.h"
#endif
#endif

#ifndef GPUCA_ALIGPUCODE
#include <fmt/printf.h>
#endif

using namespace o2::track;
using namespace o2::gpu;

//______________________________________________________________
template <typename value_T>
GPUd() void TrackParametrizationWithError<value_T>::invert()
{
  // Transform this track to the local coord. system rotated by 180 deg.
  this->invertParam();
  // since the fP1 and fP2 are not inverted, their covariances with others change sign
  mC[kSigZY] = -mC[kSigZY];
  mC[kSigSnpY] = -mC[kSigSnpY];
  mC[kSigTglZ] = -mC[kSigTglZ];
  mC[kSigTglSnp] = -mC[kSigTglSnp];
  mC[kSigQ2PtZ] = -mC[kSigQ2PtZ];
  mC[kSigQ2PtSnp] = -mC[kSigQ2PtSnp];
}

//______________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrizationWithError<value_T>::propagateTo(value_t xk, value_t b)
{
  //----------------------------------------------------------------
  // propagate this track to the plane X=xk (cm) in the field "b" (kG)
  //----------------------------------------------------------------
  value_t dx = xk - this->getX();
  if (gpu::CAMath::Abs(dx) < constants::math::Almost0) {
    return true;
  }
  value_t crv = this->getCurvature(b);
  value_t x2r = crv * dx;
  value_t f1 = this->getSnp(), f2 = f1 + x2r;
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
  this->setX(xk);
  double dy2dx = (f1 + f2) / (r1 + r2);
  value_t dP[kNParams] = {0.f};
  dP[kY] = dx * dy2dx;
  dP[kSnp] = x2r;
  if (gpu::CAMath::Abs(x2r) < 0.05f) {
    dP[kZ] = dx * (r2 + f2 * dy2dx) * this->getTgl();
  } else {
    // for small dx/R the linear apporximation of the arc by the segment is OK,
    // but at large dx/R the error is very large and leads to incorrect Z propagation
    // angle traversed delta = 2*asin(dist_start_end / R / 2), hence the arc is: R*deltaPhi
    // The dist_start_end is obtained from sqrt(dx^2+dy^2) = x/(r1+r2)*sqrt(2+f1*f2+r1*r2)
    //    double chord = dx*TMath::Sqrt(1+dy2dx*dy2dx);   // distance from old position to new one
    //    double rot = 2*TMath::ASin(0.5*chord*crv); // angular difference seen from the circle center
    //    mP1 += rot/crv*mP3;
    //
    value_t rot = gpu::CAMath::ASin(r1 * f2 - r2 * f1); // more economic version from Yura.
    if (f1 * f1 + f2 * f2 > 1.f && f1 * f2 < 0.f) { // special cases of large rotations or large abs angles
      if (f2 > 0.f) {
        rot = constants::math::PI - rot; //
      } else {
        rot = -constants::math::PI - rot;
      }
    }
    dP[kZ] = this->getTgl() / crv * rot;
  }

  this->updateParams(dP); // apply corrections

  value_t &c00 = mC[kSigY2], &c10 = mC[kSigZY], &c11 = mC[kSigZ2], &c20 = mC[kSigSnpY], &c21 = mC[kSigSnpZ],
          &c22 = mC[kSigSnp2], &c30 = mC[kSigTglY], &c31 = mC[kSigTglZ], &c32 = mC[kSigTglSnp], &c33 = mC[kSigTgl2],
          &c40 = mC[kSigQ2PtY], &c41 = mC[kSigQ2PtZ], &c42 = mC[kSigQ2PtSnp], &c43 = mC[kSigQ2PtTgl],
          &c44 = mC[kSigQ2Pt2];

  // evaluate matrix in double prec.
  double rinv = 1. / r1;
  double r3inv = rinv * rinv * rinv;
  double f24 = dx * b * constants::math::B2C; // x2r/mP[kQ2Pt];
  double f02 = dx * r3inv;
  double f04 = 0.5 * f24 * f02;
  double f12 = f02 * this->getTgl() * f1;
  double f14 = 0.5 * f24 * f12; // 0.5*f24*f02*getTgl()*f1;
  double f13 = dx * rinv;

  // b = C*ft
  double b00 = f02 * c20 + f04 * c40, b01 = f12 * c20 + f14 * c40 + f13 * c30;
  double b02 = f24 * c40;
  double b10 = f02 * c21 + f04 * c41, b11 = f12 * c21 + f14 * c41 + f13 * c31;
  double b12 = f24 * c41;
  double b20 = f02 * c22 + f04 * c42, b21 = f12 * c22 + f14 * c42 + f13 * c32;
  double b22 = f24 * c42;
  double b40 = f02 * c42 + f04 * c44, b41 = f12 * c42 + f14 * c44 + f13 * c43;
  double b42 = f24 * c44;
  double b30 = f02 * c32 + f04 * c43, b31 = f12 * c32 + f14 * c43 + f13 * c33;
  double b32 = f24 * c43;

  // a = f*b = f*C*ft
  double a00 = f02 * b20 + f04 * b40, a01 = f02 * b21 + f04 * b41, a02 = f02 * b22 + f04 * b42;
  double a11 = f12 * b21 + f14 * b41 + f13 * b31, a12 = f12 * b22 + f14 * b42 + f13 * b32;
  double a22 = f24 * b42;

  // F*C*Ft = C + (b + bt + a)
  c00 += b00 + b00 + a00;
  c10 += b10 + b01 + a01;
  c20 += b20 + b02 + a02;
  c30 += b30;
  c40 += b40;
  c11 += b11 + b11 + a11;
  c21 += b21 + b12 + a12;
  c31 += b31;
  c41 += b41;
  c22 += b22 + b22 + a22;
  c32 += b32;
  c42 += b42;

  checkCovariance();

  return true;
}

//______________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrizationWithError<value_T>::rotate(value_t alpha)
{
  // rotate to alpha frame
  if (gpu::CAMath::Abs(this->getSnp()) > constants::math::Almost1) {
    LOGP(WARNING, "Precondition is not satisfied: |sin(phi)|>1 ! {:f}", this->getSnp());
    return false;
  }
  //
  math_utils::detail::bringToPMPi<value_t>(alpha);
  //
  value_t ca = 0, sa = 0;
  math_utils::detail::sincos(alpha - this->getAlpha(), sa, ca);
  value_t snp = this->getSnp(), csp = gpu::CAMath::Sqrt((1.f - snp) * (1.f + snp)); // Improve precision
  // RS: check if rotation does no invalidate track model (cos(local_phi)>=0, i.e. particle
  // direction in local frame is along the X axis
  if ((csp * ca + snp * sa) < 0) {
    //LOGP(WARNING,"Rotation failed: local cos(phi) would become {:.2f}", csp * ca + snp * sa);
    return false;
  }
  //

  value_t updSnp = snp * ca - csp * sa;
  if (gpu::CAMath::Abs(updSnp) > constants::math::Almost1) {
    LOGP(WARNING, "Rotation failed: new snp {:.2f}", updSnp);
    return false;
  }
  value_t xold = this->getX(), yold = this->getY();
  this->setAlpha(alpha);
  this->setX(xold * ca + yold * sa);
  this->setY(-xold * sa + yold * ca);
  this->setSnp(updSnp);

  if (gpu::CAMath::Abs(csp) < constants::math::Almost0) {
    LOGP(WARNING, "Too small cosine value {:f}", csp);
    csp = constants::math::Almost0;
  }

  value_t rr = (ca + snp / csp * sa);

  mC[kSigY2] *= (ca * ca);
  mC[kSigZY] *= ca;
  mC[kSigSnpY] *= ca * rr;
  mC[kSigSnpZ] *= rr;
  mC[kSigSnp2] *= rr * rr;
  mC[kSigTglY] *= ca;
  mC[kSigTglSnp] *= rr;
  mC[kSigQ2PtY] *= ca;
  mC[kSigQ2PtSnp] *= rr;

  checkCovariance();
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrizationWithError<value_T>::propagateToDCA(const o2::dataformats::VertexBase& vtx, value_t b, o2::dataformats::DCA* dca, value_t maxD)
{
  // propagate track to DCA to the vertex
  value_t sn, cs, alp = this->getAlpha();
  o2::math_utils::detail::sincos(alp, sn, cs);
  value_t x = this->getX(), y = this->getY(), snp = this->getSnp(), csp = gpu::CAMath::Sqrt((1.f - snp) * (1.f + snp));
  value_t xv = vtx.getX() * cs + vtx.getY() * sn, yv = -vtx.getX() * sn + vtx.getY() * cs, zv = vtx.getZ();
  x -= xv;
  y -= yv;
  //Estimate the impact parameter neglecting the track curvature
  value_t d = gpu::CAMath::Abs(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  value_t crv = this->getCurvature(b);
  value_t tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / gpu::CAMath::Sqrt(1.f + tgfv * tgfv);
  cs = gpu::CAMath::Sqrt((1.f - sn) * (1.f + sn));
  cs = (gpu::CAMath::Abs(tgfv) > constants::math::Almost0) ? sn / tgfv : constants::math::Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  auto tmpT(*this); // operate on the copy to recover after the failure
  alp += gpu::CAMath::ASin(sn);
  if (!tmpT.rotate(alp) || !tmpT.propagateTo(xv, b)) {
    LOG(WARNING) << "failed to propagate to alpha=" << alp << " X=" << xv << vtx << " | Track is: ";
    tmpT.print();
    return false;
  }
  *this = tmpT;
  if (dca) {
    o2::math_utils::detail::sincos(alp, sn, cs);
    auto s2ylocvtx = vtx.getSigmaX2() * sn * sn + vtx.getSigmaY2() * cs * cs - 2. * vtx.getSigmaXY() * cs * sn;
    dca->set(this->getY() - yv, this->getZ() - zv, getSigmaY2() + s2ylocvtx, getSigmaZY(), getSigmaZ2() + vtx.getSigmaZ2());
  }
  return true;
}

//______________________________________________________________
template <typename value_T>
GPUd() TrackParametrizationWithError<value_T>::TrackParametrizationWithError(const dim3_t& xyz, const dim3_t& pxpypz,
                                                                             const gpu::gpustd::array<value_t, kLabCovMatSize>& cv, int charge, bool sectorAlpha)
{
  // construct track param and covariance from kinematics and lab errors

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
  // protection:  avoid alpha being too close to 0 or +-pi/2
  if (gpu::CAMath::Abs(sn) < 2.f * kSafe) {
    if (alp > 0) {
      alp += alp < constants::math::PIHalf ? 2.f * kSafe : -2.f * kSafe;
    } else {
      alp += alp > -constants::math::PIHalf ? -2.f * kSafe : 2.f * kSafe;
    }
    math_utils::detail::sincos(alp, sn, cs);
  } else if (gpu::CAMath::Abs(cs) < 2.f * kSafe) {
    if (alp > 0) {
      alp += alp > constants::math::PIHalf ? 2.f * kSafe : -2.f * kSafe;
    } else {
      alp += alp > -constants::math::PIHalf ? 2.f * kSafe : -2.f * kSafe;
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
  value_t pt = gpu::CAMath::Sqrt(mom[0] * mom[0] + mom[1] * mom[1]);
  value_t ptI = 1.f / pt;
  this->setX(ver[0]);
  this->setAlpha(alp);
  this->setY(ver[1]);
  this->setZ(ver[2]);
  this->setSnp(mom[1] * ptI); // cos(phi)
  this->setTgl(mom[2] * ptI); // tg(lambda)
  this->setAbsCharge(gpu::CAMath::Abs(charge));
  this->setQ2Pt(charge ? ptI * charge : ptI);
  //
  if (gpu::CAMath::Abs(1.f - this->getSnp()) < kSafe) {
    this->setSnp(1.f - kSafe); // Protection
  } else if (gpu::CAMath::Abs(-1.f - this->getSnp()) < kSafe) {
    this->setSnp(-1.f + kSafe); // Protection
  }
  //
  // Covariance matrix (formulas to be simplified)
  value_t r = mom[0] * ptI; // cos(phi)
  value_t cv34 = gpu::CAMath::Sqrt(cv[3] * cv[3] + cv[4] * cv[4]);
  //
  int special = 0;
  value_t sgcheck = r * sn + this->getSnp() * cs;
  if (gpu::CAMath::Abs(sgcheck) > 1 - kSafe) { // special case: lab phi is +-pi/2
    special = 1;
    sgcheck = sgcheck < 0 ? -1.f : 1.f;
  } else if (gpu::CAMath::Abs(sgcheck) < kSafe) {
    sgcheck = cs < 0 ? -1.0f : 1.0f;
    special = 2; // special case: lab phi is 0
  }
  //
  mC[kSigY2] = cv[0] + cv[2];
  mC[kSigZY] = (-cv[3] * sn) < 0 ? -cv34 : cv34;
  mC[kSigZ2] = cv[5];
  //
  value_t ptI2 = ptI * ptI;
  value_t tgl2 = this->getTgl() * this->getTgl();
  if (special == 1) {
    mC[kSigSnpY] = cv[6] * ptI;
    mC[kSigSnpZ] = -sgcheck * cv[8] * r * ptI;
    mC[kSigSnp2] = gpu::CAMath::Abs(cv[9] * r * r * ptI2);
    mC[kSigTglY] = (cv[10] * this->getTgl() - sgcheck * cv[15]) * ptI / r;
    mC[kSigTglZ] = (cv[17] - sgcheck * cv[12] * this->getTgl()) * ptI;
    mC[kSigTglSnp] = (-sgcheck * cv[18] + cv[13] * this->getTgl()) * r * ptI2;
    mC[kSigTgl2] = gpu::CAMath::Abs(cv[20] - 2 * sgcheck * cv[19] * mC[4] + cv[14] * tgl2) * ptI2;
    mC[kSigQ2PtY] = cv[10] * ptI2 / r * charge;
    mC[kSigQ2PtZ] = -sgcheck * cv[12] * ptI2 * charge;
    mC[kSigQ2PtSnp] = cv[13] * r * ptI * ptI2 * charge;
    mC[kSigQ2PtTgl] = (-sgcheck * cv[19] + cv[14] * this->getTgl()) * r * ptI2 * ptI;
    mC[kSigQ2Pt2] = gpu::CAMath::Abs(cv[14] * ptI2 * ptI2);
  } else if (special == 2) {
    mC[kSigSnpY] = -cv[10] * ptI * cs / sn;
    mC[kSigSnpZ] = cv[12] * cs * ptI;
    mC[kSigSnp2] = gpu::CAMath::Abs(cv[14] * cs * cs * ptI2);
    mC[kSigTglY] = (sgcheck * cv[6] * this->getTgl() - cv[15]) * ptI / sn;
    mC[kSigTglZ] = (cv[17] - sgcheck * cv[8] * this->getTgl()) * ptI;
    mC[kSigTglSnp] = (cv[19] - sgcheck * cv[13] * this->getTgl()) * cs * ptI2;
    mC[kSigTgl2] = gpu::CAMath::Abs(cv[20] - 2 * sgcheck * cv[18] * this->getTgl() + cv[9] * tgl2) * ptI2;
    mC[kSigQ2PtY] = sgcheck * cv[6] * ptI2 / sn * charge;
    mC[kSigQ2PtZ] = -sgcheck * cv[8] * ptI2 * charge;
    mC[kSigQ2PtSnp] = -sgcheck * cv[13] * cs * ptI * ptI2 * charge;
    mC[kSigQ2PtTgl] = (-sgcheck * cv[18] + cv[9] * this->getTgl()) * ptI2 * ptI * charge;
    mC[kSigQ2Pt2] = gpu::CAMath::Abs(cv[9] * ptI2 * ptI2);
  } else {
    double m00 = -sn; // m10=cs;
    double m23 = -pt * (sn + this->getSnp() * cs / r), m43 = -pt * pt * (r * cs - this->getSnp() * sn);
    double m24 = pt * (cs - this->getSnp() * sn / r), m44 = -pt * pt * (r * sn + this->getSnp() * cs);
    double m35 = pt, m45 = -pt * pt * this->getTgl();
    //
    if (charge) { // RS: this is a hack, proper treatment to be implemented
      m43 *= charge;
      m44 *= charge;
      m45 *= charge;
    }
    //
    double a1 = cv[13] - cv[9] * (m23 * m44 + m43 * m24) / m23 / m43;
    double a2 = m23 * m24 - m23 * (m23 * m44 + m43 * m24) / m43;
    double a3 = m43 * m44 - m43 * (m23 * m44 + m43 * m24) / m23;
    double a4 = cv[14] + 2. * cv[9];
    double a5 = m24 * m24 - 2. * m24 * m44 * m23 / m43;
    double a6 = m44 * m44 - 2. * m24 * m44 * m43 / m23;
    //
    mC[kSigSnpY] = (cv[10] * m43 - cv[6] * m44) / (m24 * m43 - m23 * m44) / m00;
    mC[kSigQ2PtY] = (cv[6] / m00 - mC[kSigSnpY] * m23) / m43;
    mC[kSigTglY] = (cv[15] / m00 - mC[kSigQ2PtY] * m45) / m35;
    mC[kSigSnpZ] = (cv[12] * m43 - cv[8] * m44) / (m24 * m43 - m23 * m44);
    mC[kSigQ2PtZ] = (cv[8] - mC[kSigSnpZ] * m23) / m43;
    mC[kSigTglZ] = cv[17] / m35 - mC[kSigQ2PtZ] * m45 / m35;
    mC[kSigSnp2] = gpu::CAMath::Abs((a4 * a3 - a6 * a1) / (a5 * a3 - a6 * a2));
    mC[kSigQ2Pt2] = gpu::CAMath::Abs((a1 - a2 * mC[kSigSnp2]) / a3);
    mC[kSigQ2PtSnp] = (cv[9] - mC[kSigSnp2] * m23 * m23 - mC[kSigQ2Pt2] * m43 * m43) / m23 / m43;
    double b1 = cv[18] - mC[kSigQ2PtSnp] * m23 * m45 - mC[kSigQ2Pt2] * m43 * m45;
    double b2 = m23 * m35;
    double b3 = m43 * m35;
    double b4 = cv[19] - mC[kSigQ2PtSnp] * m24 * m45 - mC[kSigQ2Pt2] * m44 * m45;
    double b5 = m24 * m35;
    double b6 = m44 * m35;
    mC[kSigTglSnp] = (b4 - b6 * b1 / b3) / (b5 - b6 * b2 / b3);
    mC[kSigQ2PtTgl] = b1 / b3 - b2 * mC[kSigTglSnp] / b3;
    mC[kSigTgl2] = gpu::CAMath::Abs((cv[20] - mC[kSigQ2Pt2] * (m45 * m45) - mC[kSigQ2PtTgl] * 2.f * m35 * m45) / (m35 * m35));
  }
  checkCovariance();
}

//____________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrizationWithError<value_T>::propagateTo(value_t xk, const dim3_t& b)
{
  //----------------------------------------------------------------
  // Extrapolate this track to the plane X=xk in the field b[].
  //
  // X [cm] is in the "tracking coordinate system" of this track.
  // b[]={Bx,By,Bz} [kG] is in the Global coordidate system.
  //----------------------------------------------------------------

  value_t dx = xk - this->getX();
  if (gpu::CAMath::Abs(dx) < constants::math::Almost0) {
    return true;
  }
  // Do not propagate tracks outside the ALICE detector
  if (gpu::CAMath::Abs(dx) > 1e5 || gpu::CAMath::Abs(this->getY()) > 1e5 || gpu::CAMath::Abs(this->getZ()) > 1e5) {
    LOGP(WARNING, "Anomalous track, target X:{:f}", xk);
    //    print();
    return false;
  }
  value_t crv = (gpu::CAMath::Abs(b[2]) < constants::math::Almost0) ? 0.f : this->getCurvature(b[2]);
  value_t x2r = crv * dx;
  value_t f1 = this->getSnp(), f2 = f1 + x2r;
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

  value_t dy2dx = (f1 + f2) / (r1 + r2);
  value_t step = (gpu::CAMath::Abs(x2r) < 0.05f) ? dx * gpu::CAMath::Abs(r2 + f2 * dy2dx)                                                   // chord
                                                 : 2.f * gpu::CAMath::ASin(0.5f * dx * gpu::CAMath::Sqrt(1.f + dy2dx * dy2dx) * crv) / crv; // arc
  step *= gpu::CAMath::Sqrt(1.f + this->getTgl() * this->getTgl());
  //
  // get the track x,y,z,px/p,py/p,pz/p,p,sinAlpha,cosAlpha in the Global System
  gpu::gpustd::array<value_t, 9> vecLab{0.f};
  if (!this->getPosDirGlo(vecLab)) {
    return false;
  }
  //
  // matrix transformed with Bz component only
  value_t &c00 = mC[kSigY2], &c10 = mC[kSigZY], &c11 = mC[kSigZ2], &c20 = mC[kSigSnpY], &c21 = mC[kSigSnpZ],
          &c22 = mC[kSigSnp2], &c30 = mC[kSigTglY], &c31 = mC[kSigTglZ], &c32 = mC[kSigTglSnp], &c33 = mC[kSigTgl2],
          &c40 = mC[kSigQ2PtY], &c41 = mC[kSigQ2PtZ], &c42 = mC[kSigQ2PtSnp], &c43 = mC[kSigQ2PtTgl],
          &c44 = mC[kSigQ2Pt2];
  // evaluate matrix in double prec.
  double rinv = 1. / r1;
  double r3inv = rinv * rinv * rinv;
  double f24 = dx * b[2] * constants::math::B2C; // x2r/track[kQ2Pt];
  double f02 = dx * r3inv;
  double f04 = 0.5 * f24 * f02;
  double f12 = f02 * this->getTgl() * f1;
  double f14 = 0.5 * f24 * f12; // 0.5*f24*f02*getTgl()*f1;
  double f13 = dx * rinv;

  // b = C*ft
  double b00 = f02 * c20 + f04 * c40, b01 = f12 * c20 + f14 * c40 + f13 * c30;
  double b02 = f24 * c40;
  double b10 = f02 * c21 + f04 * c41, b11 = f12 * c21 + f14 * c41 + f13 * c31;
  double b12 = f24 * c41;
  double b20 = f02 * c22 + f04 * c42, b21 = f12 * c22 + f14 * c42 + f13 * c32;
  double b22 = f24 * c42;
  double b40 = f02 * c42 + f04 * c44, b41 = f12 * c42 + f14 * c44 + f13 * c43;
  double b42 = f24 * c44;
  double b30 = f02 * c32 + f04 * c43, b31 = f12 * c32 + f14 * c43 + f13 * c33;
  double b32 = f24 * c43;

  // a = f*b = f*C*ft
  double a00 = f02 * b20 + f04 * b40, a01 = f02 * b21 + f04 * b41, a02 = f02 * b22 + f04 * b42;
  double a11 = f12 * b21 + f14 * b41 + f13 * b31, a12 = f12 * b22 + f14 * b42 + f13 * b32;
  double a22 = f24 * b42;

  // F*C*Ft = C + (b + bt + a)
  c00 += b00 + b00 + a00;
  c10 += b10 + b01 + a01;
  c20 += b20 + b02 + a02;
  c30 += b30;
  c40 += b40;
  c11 += b11 + b11 + a11;
  c21 += b21 + b12 + a12;
  c31 += b31;
  c41 += b41;
  c22 += b22 + b22 + a22;
  c32 += b32;
  c42 += b42;

  checkCovariance();

  // Rotate to the system where Bx=By=0.
  value_t bxy2 = b[0] * b[0] + b[1] * b[1];
  value_t bt = gpu::CAMath::Sqrt(bxy2);
  value_t cosphi = 1.f, sinphi = 0.f;
  if (bt > constants::math::Almost0) {
    cosphi = b[0] / bt;
    sinphi = b[1] / bt;
  }
  value_t bb = gpu::CAMath::Sqrt(bxy2 + b[2] * b[2]);
  value_t costet = 1., sintet = 0.;
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
  value_t sgn = this->getSign();
  g3helx3(sgn * bb, step, vect);

  // Rotate back to the Global System
  vecLab[0] = cosphi * costet * vect[0] - sinphi * vect[1] + cosphi * sintet * vect[2];
  vecLab[1] = sinphi * costet * vect[0] + cosphi * vect[1] + sinphi * sintet * vect[2];
  vecLab[2] = -sintet * vect[0] + costet * vect[2];

  vecLab[3] = cosphi * costet * vect[3] - sinphi * vect[4] + cosphi * sintet * vect[5];
  vecLab[4] = sinphi * costet * vect[3] + cosphi * vect[4] + sinphi * sintet * vect[5];
  vecLab[5] = -sintet * vect[3] + costet * vect[5];

  // Rotate back to the Tracking System
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
  this->setX(x);
  this->setY(y);
  this->setZ(z);
  this->setSnp(vecLab[4] * t);
  this->setTgl(vecLab[5] * t);
  this->setQ2Pt(sgn * t / vecLab[6]);

  return true;
}

//______________________________________________
template <typename value_T>
GPUd() void TrackParametrizationWithError<value_T>::checkCovariance()
{
  // This function forces the diagonal elements of the covariance matrix to be positive.
  // In case the diagonal element is bigger than the maximal allowed value, it is set to
  // the limit and the off-diagonal elements that correspond to it are set to zero.

  mC[kSigY2] = gpu::CAMath::Abs(mC[kSigY2]);
  if (mC[kSigY2] > kCY2max) {
    value_t scl = gpu::CAMath::Sqrt(kCY2max / mC[kSigY2]);
    mC[kSigY2] = kCY2max;
    mC[kSigZY] *= scl;
    mC[kSigSnpY] *= scl;
    mC[kSigTglY] *= scl;
    mC[kSigQ2PtY] *= scl;
  }
  mC[kSigZ2] = gpu::CAMath::Abs(mC[kSigZ2]);
  if (mC[kSigZ2] > kCZ2max) {
    value_t scl = gpu::CAMath::Sqrt(kCZ2max / mC[kSigZ2]);
    mC[kSigZ2] = kCZ2max;
    mC[kSigZY] *= scl;
    mC[kSigSnpZ] *= scl;
    mC[kSigTglZ] *= scl;
    mC[kSigQ2PtZ] *= scl;
  }
  mC[kSigSnp2] = gpu::CAMath::Abs(mC[kSigSnp2]);
  if (mC[kSigSnp2] > kCSnp2max) {
    value_t scl = gpu::CAMath::Sqrt(kCSnp2max / mC[kSigSnp2]);
    mC[kSigSnp2] = kCSnp2max;
    mC[kSigSnpY] *= scl;
    mC[kSigSnpZ] *= scl;
    mC[kSigTglSnp] *= scl;
    mC[kSigQ2PtSnp] *= scl;
  }
  mC[kSigTgl2] = gpu::CAMath::Abs(mC[kSigTgl2]);
  if (mC[kSigTgl2] > kCTgl2max) {
    value_t scl = gpu::CAMath::Sqrt(kCTgl2max / mC[kSigTgl2]);
    mC[kSigTgl2] = kCTgl2max;
    mC[kSigTglY] *= scl;
    mC[kSigTglZ] *= scl;
    mC[kSigTglSnp] *= scl;
    mC[kSigQ2PtTgl] *= scl;
  }
  mC[kSigQ2Pt2] = gpu::CAMath::Abs(mC[kSigQ2Pt2]);
  if (mC[kSigQ2Pt2] > kC1Pt2max) {
    value_t scl = gpu::CAMath::Sqrt(kC1Pt2max / mC[kSigQ2Pt2]);
    mC[kSigQ2Pt2] = kC1Pt2max;
    mC[kSigQ2PtY] *= scl;
    mC[kSigQ2PtZ] *= scl;
    mC[kSigQ2PtSnp] *= scl;
    mC[kSigQ2PtTgl] *= scl;
  }
}

//______________________________________________
template <typename value_T>
GPUd() void TrackParametrizationWithError<value_T>::resetCovariance(value_t s2)
{
  // Reset the covarince matrix to "something big"
  double d0(kCY2max), d1(kCZ2max), d2(kCSnp2max), d3(kCTgl2max), d4(kC1Pt2max);
  if (s2 > constants::math::Almost0) {
    d0 = getSigmaY2() * s2;
    d1 = getSigmaZ2() * s2;
    d2 = getSigmaSnp2() * s2;
    d3 = getSigmaTgl2() * s2;
    d4 = getSigma1Pt2() * s2;
    if (d0 > kCY2max) {
      d0 = kCY2max;
    }
    if (d1 > kCZ2max) {
      d1 = kCZ2max;
    }
    if (d2 > kCSnp2max) {
      d2 = kCSnp2max;
    }
    if (d3 > kCTgl2max) {
      d3 = kCTgl2max;
    }
    if (d4 > kC1Pt2max) {
      d4 = kC1Pt2max;
    }
  }
  for (int i = 0; i < kCovMatSize; i++) {
    mC[i] = 0;
  }
  mC[kSigY2] = d0;
  mC[kSigZ2] = d1;
  mC[kSigSnp2] = d2;
  mC[kSigTgl2] = d3;
  mC[kSigQ2Pt2] = d4;
}

//______________________________________________
template <typename value_T>
GPUd() typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getPredictedChi2(const dim2_t& p, const dim3_t& cov) const
{
  // Estimate the chi2 of the space point "p" with the cov. matrix "cov"
  auto sdd = static_cast<double>(getSigmaY2()) + static_cast<double>(cov[0]);
  auto sdz = static_cast<double>(getSigmaZY()) + static_cast<double>(cov[1]);
  auto szz = static_cast<double>(getSigmaZ2()) + static_cast<double>(cov[2]);
  auto det = sdd * szz - sdz * sdz;

  if (gpu::CAMath::Abs(det) < constants::math::Almost0) {
    return constants::math::VeryBig;
  }

  value_t d = this->getY() - p[0];
  value_t z = this->getZ() - p[1];

  return (d * (szz * d - sdz * z) + z * (sdd * z - d * sdz)) / det;
}

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // Disable function relying on ROOT SMatrix on GPU

//______________________________________________
template <typename value_T>
void TrackParametrizationWithError<value_T>::buildCombinedCovMatrix(const TrackParametrizationWithError<value_T>& rhs, MatrixDSym5& cov) const
{
  // fill combined cov.matrix (NOT inverted)
  cov(kY, kY) = static_cast<double>(getSigmaY2()) + static_cast<double>(rhs.getSigmaY2());
  cov(kZ, kY) = static_cast<double>(getSigmaZY()) + static_cast<double>(rhs.getSigmaZY());
  cov(kZ, kZ) = static_cast<double>(getSigmaZ2()) + static_cast<double>(rhs.getSigmaZ2());
  cov(kSnp, kY) = static_cast<double>(getSigmaSnpY()) + static_cast<double>(rhs.getSigmaSnpY());
  cov(kSnp, kZ) = static_cast<double>(getSigmaSnpZ()) + static_cast<double>(rhs.getSigmaSnpZ());
  cov(kSnp, kSnp) = static_cast<double>(getSigmaSnp2()) + static_cast<double>(rhs.getSigmaSnp2());
  cov(kTgl, kY) = static_cast<double>(getSigmaTglY()) + static_cast<double>(rhs.getSigmaTglY());
  cov(kTgl, kZ) = static_cast<double>(getSigmaTglZ()) + static_cast<double>(rhs.getSigmaTglZ());
  cov(kTgl, kSnp) = static_cast<double>(getSigmaTglSnp()) + static_cast<double>(rhs.getSigmaTglSnp());
  cov(kTgl, kTgl) = static_cast<double>(getSigmaTgl2()) + static_cast<double>(rhs.getSigmaTgl2());
  cov(kQ2Pt, kY) = static_cast<double>(getSigma1PtY()) + static_cast<double>(rhs.getSigma1PtY());
  cov(kQ2Pt, kZ) = static_cast<double>(getSigma1PtZ()) + static_cast<double>(rhs.getSigma1PtZ());
  cov(kQ2Pt, kSnp) = static_cast<double>(getSigma1PtSnp()) + static_cast<double>(rhs.getSigma1PtSnp());
  cov(kQ2Pt, kTgl) = static_cast<double>(getSigma1PtTgl()) + static_cast<double>(rhs.getSigma1PtTgl());
  cov(kQ2Pt, kQ2Pt) = static_cast<double>(getSigma1Pt2()) + static_cast<double>(rhs.getSigma1Pt2());
}

//______________________________________________
template <typename value_T>
typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getPredictedChi2(const TrackParametrizationWithError<value_T>& rhs) const
{
  MatrixDSym5 cov; // perform matrix operations in double!
  return getPredictedChi2(rhs, cov);
}

//______________________________________________
template <typename value_T>
typename TrackParametrizationWithError<value_T>::value_t TrackParametrizationWithError<value_T>::getPredictedChi2(const TrackParametrizationWithError<value_T>& rhs, MatrixDSym5& covToSet) const
{
  // get chi2 wrt other track, which must be defined at the same parameters X,alpha
  // Supplied non-initialized covToSet matrix is filled by inverse combined matrix for further use

  if (gpu::CAMath::Abs(this->getAlpha() - rhs.getAlpha()) > FLT_EPSILON) {
    LOG(ERROR) << "The reference Alpha of the tracks differ: " << this->getAlpha() << " : " << rhs.getAlpha();
    return 2.f * HugeF;
  }
  if (gpu::CAMath::Abs(this->getX() - rhs.getX()) > FLT_EPSILON) {
    LOG(ERROR) << "The reference X of the tracks differ: " << this->getX() << " : " << rhs.getX();
    return 2.f * HugeF;
  }
  buildCombinedCovMatrix(rhs, covToSet);
  if (!covToSet.Invert()) {
    LOG(WARNING) << "Cov.matrix inversion failed: " << covToSet;
    return 2.f * HugeF;
  }
  double chi2diag = 0., chi2ndiag = 0., diff[kNParams];
  for (int i = kNParams; i--;) {
    diff[i] = this->getParam(i) - rhs.getParam(i);
    chi2diag += diff[i] * diff[i] * covToSet(i, i);
  }
  for (int i = kNParams; i--;) {
    for (int j = i; j--;) {
      chi2ndiag += diff[i] * diff[j] * covToSet(i, j);
    }
  }
  return chi2diag + 2. * chi2ndiag;
}

//______________________________________________
template <typename value_T>
bool TrackParametrizationWithError<value_T>::update(const TrackParametrizationWithError<value_T>& rhs, const MatrixDSym5& covInv)
{
  // update track with other track, the inverted combined cov matrix should be supplied

  // consider skipping this check, since it is usually already done upstream
  if (gpu::CAMath::Abs(this->getAlpha() - rhs.getAlpha()) > FLT_EPSILON) {
    LOG(ERROR) << "The reference Alpha of the tracks differ: " << this->getAlpha() << " : " << rhs.getAlpha();
    return false;
  }
  if (gpu::CAMath::Abs(this->getX() - rhs.getX()) > FLT_EPSILON) {
    LOG(ERROR) << "The reference X of the tracks differ: " << this->getX() << " : " << rhs.getX();
    return false;
  }

  // gain matrix K = Cov0*H*(Cov0+Cov0)^-1 (for measurement matrix H=I)
  MatrixDSym5 matC0;
  matC0(kY, kY) = getSigmaY2();
  matC0(kZ, kY) = getSigmaZY();
  matC0(kZ, kZ) = getSigmaZ2();
  matC0(kSnp, kY) = getSigmaSnpY();
  matC0(kSnp, kZ) = getSigmaSnpZ();
  matC0(kSnp, kSnp) = getSigmaSnp2();
  matC0(kTgl, kY) = getSigmaTglY();
  matC0(kTgl, kZ) = getSigmaTglZ();
  matC0(kTgl, kSnp) = getSigmaTglSnp();
  matC0(kTgl, kTgl) = getSigmaTgl2();
  matC0(kQ2Pt, kY) = getSigma1PtY();
  matC0(kQ2Pt, kZ) = getSigma1PtZ();
  matC0(kQ2Pt, kSnp) = getSigma1PtSnp();
  matC0(kQ2Pt, kTgl) = getSigma1PtTgl();
  matC0(kQ2Pt, kQ2Pt) = getSigma1Pt2();
  MatrixD5 matK = matC0 * covInv;

  // updated state vector: x = K*(x1-x0)
  // RS: why SMatix, SVector does not provide multiplication operators ???
  double diff[kNParams];
  for (int i = kNParams; i--;) {
    diff[i] = rhs.getParam(i) - this->getParam(i);
  }
  for (int i = kNParams; i--;) {
    for (int j = kNParams; j--;) {
      this->updateParam(matK(i, j) * diff[j], i);
    }
  }

  // updated covariance: Cov0 = Cov0 - K*Cov0
  matK *= ROOT::Math::SMatrix<double, kNParams, kNParams, ROOT::Math::MatRepStd<double, kNParams>>(matC0);
  mC[kSigY2] -= matK(kY, kY);
  mC[kSigZY] -= matK(kZ, kY);
  mC[kSigZ2] -= matK(kZ, kZ);
  mC[kSigSnpY] -= matK(kSnp, kY);
  mC[kSigSnpZ] -= matK(kSnp, kZ);
  mC[kSigSnp2] -= matK(kSnp, kSnp);
  mC[kSigTglY] -= matK(kTgl, kY);
  mC[kSigTglZ] -= matK(kTgl, kZ);
  mC[kSigTglSnp] -= matK(kTgl, kSnp);
  mC[kSigTgl2] -= matK(kTgl, kTgl);
  mC[kSigQ2PtY] -= matK(kQ2Pt, kY);
  mC[kSigQ2PtZ] -= matK(kQ2Pt, kZ);
  mC[kSigQ2PtSnp] -= matK(kQ2Pt, kSnp);
  mC[kSigQ2PtTgl] -= matK(kQ2Pt, kTgl);
  mC[kSigQ2Pt2] -= matK(kQ2Pt, kQ2Pt);

  return true;
}

//______________________________________________
template <typename value_T>
bool TrackParametrizationWithError<value_T>::update(const TrackParametrizationWithError<value_T>& rhs)
{
  // update track with other track
  MatrixDSym5 covI; // perform matrix operations in double!
  buildCombinedCovMatrix(rhs, covI);
  if (!covI.Invert()) {
    LOG(WARNING) << "Cov.matrix inversion failed: " << covI;
    return false;
  }
  return update(rhs, covI);
}

#endif

//______________________________________________
template <typename value_T>
GPUd() bool TrackParametrizationWithError<value_T>::update(const dim2_t& p, const dim3_t& cov)
{
  // Update the track parameters with the space point "p" having
  // the covariance matrix "cov"

  value_t &cm00 = mC[kSigY2], &cm10 = mC[kSigZY], &cm11 = mC[kSigZ2], &cm20 = mC[kSigSnpY], &cm21 = mC[kSigSnpZ],
          &cm22 = mC[kSigSnp2], &cm30 = mC[kSigTglY], &cm31 = mC[kSigTglZ], &cm32 = mC[kSigTglSnp], &cm33 = mC[kSigTgl2],
          &cm40 = mC[kSigQ2PtY], &cm41 = mC[kSigQ2PtZ], &cm42 = mC[kSigQ2PtSnp], &cm43 = mC[kSigQ2PtTgl],
          &cm44 = mC[kSigQ2Pt2];

  // use double precision?
  double r00 = static_cast<double>(cov[0]) + static_cast<double>(cm00);
  double r01 = static_cast<double>(cov[1]) + static_cast<double>(cm10);
  double r11 = static_cast<double>(cov[2]) + static_cast<double>(cm11);
  double det = r00 * r11 - r01 * r01;

  if (gpu::CAMath::Abs(det) < constants::math::Almost0) {
    return false;
  }
  double detI = 1. / det;
  double tmp = r00;
  r00 = r11 * detI;
  r11 = tmp * detI;
  r01 = -r01 * detI;

  double k00 = cm00 * r00 + cm10 * r01, k01 = cm00 * r01 + cm10 * r11;
  double k10 = cm10 * r00 + cm11 * r01, k11 = cm10 * r01 + cm11 * r11;
  double k20 = cm20 * r00 + cm21 * r01, k21 = cm20 * r01 + cm21 * r11;
  double k30 = cm30 * r00 + cm31 * r01, k31 = cm30 * r01 + cm31 * r11;
  double k40 = cm40 * r00 + cm41 * r01, k41 = cm40 * r01 + cm41 * r11;

  value_t dy = p[kY] - this->getY(), dz = p[kZ] - this->getZ();
  value_t dsnp = k20 * dy + k21 * dz;
  if (gpu::CAMath::Abs(this->getSnp() + dsnp) > constants::math::Almost1) {
    return false;
  }

  value_t dP[kNParams] = {value_t(k00 * dy + k01 * dz), value_t(k10 * dy + k11 * dz), dsnp, value_t(k30 * dy + k31 * dz),
                          value_t(k40 * dy + k41 * dz)};
  this->updateParams(dP);

  double c01 = cm10, c02 = cm20, c03 = cm30, c04 = cm40;
  double c12 = cm21, c13 = cm31, c14 = cm41;

  cm00 -= k00 * cm00 + k01 * cm10;
  cm10 -= k00 * c01 + k01 * cm11;
  cm20 -= k00 * c02 + k01 * c12;
  cm30 -= k00 * c03 + k01 * c13;
  cm40 -= k00 * c04 + k01 * c14;

  cm11 -= k10 * c01 + k11 * cm11;
  cm21 -= k10 * c02 + k11 * c12;
  cm31 -= k10 * c03 + k11 * c13;
  cm41 -= k10 * c04 + k11 * c14;

  cm22 -= k20 * c02 + k21 * c12;
  cm32 -= k20 * c03 + k21 * c13;
  cm42 -= k20 * c04 + k21 * c14;

  cm33 -= k30 * c03 + k31 * c13;
  cm43 -= k30 * c04 + k31 * c14;

  cm44 -= k40 * c04 + k41 * c14;

  checkCovariance();

  return true;
}

//______________________________________________
template <typename value_T>
GPUd() bool TrackParametrizationWithError<value_T>::correctForMaterial(value_t x2x0, value_t xrho, bool anglecorr, value_t dedx)
{
  //------------------------------------------------------------------
  // This function corrects the track parameters for the crossed material.
  // "x2x0"   - X/X0, the thickness in units of the radiation length.
  // "xrho" - is the product length*density (g/cm^2).
  //     It should be passed as negative when propagating tracks
  //     from the intreaction point to the outside of the central barrel.
  // "dedx" - mean enery loss (GeV/(g/cm^2), if <=kCalcdEdxAuto : calculate on the fly
  // "anglecorr" - switch for the angular correction
  //------------------------------------------------------------------
  constexpr value_t kMSConst2 = 0.0136f * 0.0136f;
  constexpr value_t kMaxELossFrac = 0.3f; // max allowed fractional eloss
  constexpr value_t kMinP = 0.01f;        // kill below this momentum

  value_t& fC22 = mC[kSigSnp2];
  value_t& fC33 = mC[kSigTgl2];
  value_t& fC43 = mC[kSigQ2PtTgl];
  value_t& fC44 = mC[kSigQ2Pt2];
  //
  value_t csp2 = (1.f - this->getSnp()) * (1.f + this->getSnp()); // cos(phi)^2
  value_t cst2I = (1.f + this->getTgl() * this->getTgl());        // 1/cos(lambda)^2
  // Apply angle correction, if requested
  if (anglecorr) {
    value_t angle = gpu::CAMath::Sqrt(cst2I / csp2);
    x2x0 *= angle;
    xrho *= angle;
  }
  value_t p = this->getP();
  value_t p2 = p * p;
  value_t e2 = p2 + this->getPID().getMass2();
  value_t beta2 = p2 / e2;

  // Calculating the multiple scattering corrections******************
  value_t cC22(0.f), cC33(0.f), cC43(0.f), cC44(0.f);
  if (x2x0 != 0.f) {
    value_t theta2 = kMSConst2 / (beta2 * p2) * gpu::CAMath::Abs(x2x0);
    if (this->getAbsCharge() != 1) {
      theta2 *= this->getAbsCharge() * this->getAbsCharge();
    }
    if (theta2 > constants::math::PI * constants::math::PI) {
      return false;
    }
    value_t fp34 = this->getTgl() * this->getCharge2Pt();
    value_t t2c2I = theta2 * cst2I;
    cC22 = t2c2I * csp2;
    cC33 = t2c2I * cst2I;
    cC43 = t2c2I * fp34;
    cC44 = theta2 * fp34 * fp34;
    // optimes this
    //    cC22 = theta2*((1.-getSnp())*(1.+getSnp()))*(1. + this->getTgl()*getTgl());
    //    cC33 = theta2*(1. + this->getTgl()*getTgl())*(1. + this->getTgl()*getTgl());
    //    cC43 = theta2*getTgl()*this->getQ2Pt()*(1. + this->getTgl()*getTgl());
    //    cC44 = theta2*getTgl()*this->getQ2Pt()*getTgl()*this->getQ2Pt();
  }

  // Calculating the energy loss corrections************************
  value_t cP4 = 1.f;
  if ((xrho != 0.f) && (beta2 < 1.f)) {
    if (dedx < kCalcdEdxAuto + constants::math::Almost1) { // request to calculate dedx on the fly
      dedx = BetheBlochSolid(p / this->getPID().getMass());
      if (this->getAbsCharge() != 1) {
        dedx *= this->getAbsCharge() * this->getAbsCharge();
      }
    }

    value_t dE = dedx * xrho;
    value_t e = gpu::CAMath::Sqrt(e2);
    if (gpu::CAMath::Abs(dE) > kMaxELossFrac * e) {
      return false; // 30% energy loss is too much!
    }
    value_t eupd = e + dE;
    value_t pupd2 = eupd * eupd - this->getPID().getMass2();
    if (pupd2 < kMinP * kMinP) {
      return false;
    }
    cP4 = p / gpu::CAMath::Sqrt(pupd2);
    //
    // Approximate energy loss fluctuation (M.Ivanov)
    constexpr value_t knst = 0.07f; // To be tuned.
    value_t sigmadE = knst * gpu::CAMath::Sqrt(gpu::CAMath::Abs(dE)) * e / p2 * this->getCharge2Pt();
    cC44 += sigmadE * sigmadE;
  }

  // Applying the corrections*****************************
  fC22 += cC22;
  fC33 += cC33;
  fC43 += cC43;
  fC44 += cC44;
  this->setQ2Pt(this->getQ2Pt() * cP4);

  checkCovariance();

  return true;
}

//______________________________________________________________
template <typename value_T>
GPUd() bool TrackParametrizationWithError<value_T>::getCovXYZPxPyPzGlo(gpu::gpustd::array<value_t, kLabCovMatSize>& cv) const
{
  //---------------------------------------------------------------------
  // This function returns the global covariance matrix of the track params
  //
  // Cov(x,x) ... :   cv[0]
  // Cov(y,x) ... :   cv[1]  cv[2]
  // Cov(z,x) ... :   cv[3]  cv[4]  cv[5]
  // Cov(px,x)... :   cv[6]  cv[7]  cv[8]  cv[9]
  // Cov(py,x)... :   cv[10] cv[11] cv[12] cv[13] cv[14]
  // Cov(pz,x)... :   cv[15] cv[16] cv[17] cv[18] cv[19] cv[20]
  //
  // Results for (nearly) straight tracks are meaningless !
  //---------------------------------------------------------------------
  if (gpu::CAMath::Abs(this->getQ2Pt()) <= constants::math::Almost0 || gpu::CAMath::Abs(this->getSnp()) > constants::math::Almost1) {
    for (int i = 0; i < 21; i++) {
      cv[i] = 0.;
    }
    return false;
  }

  auto pt = this->getPt();
  value_t sn, cs;
  o2::math_utils::detail::sincos(this->getAlpha(), sn, cs);
  auto r = gpu::CAMath::Sqrt((1. - this->getSnp()) * (1. + this->getSnp()));
  auto m00 = -sn, m10 = cs;
  auto m23 = -pt * (sn + this->getSnp() * cs / r), m43 = -pt * pt * (r * cs - this->getSnp() * sn);
  auto m24 = pt * (cs - this->getSnp() * sn / r), m44 = -pt * pt * (r * sn + this->getSnp() * cs);
  auto m35 = pt, m45 = -pt * pt * this->getTgl();

  if (this->getSign() < 0) {
    m43 = -m43;
    m44 = -m44;
    m45 = -m45;
  }

  cv[0] = mC[0] * m00 * m00;
  cv[1] = mC[0] * m00 * m10;
  cv[2] = mC[0] * m10 * m10;
  cv[3] = mC[1] * m00;
  cv[4] = mC[1] * m10;
  cv[5] = mC[2];
  cv[6] = m00 * (mC[3] * m23 + mC[10] * m43);
  cv[7] = m10 * (mC[3] * m23 + mC[10] * m43);
  cv[8] = mC[4] * m23 + mC[11] * m43;
  cv[9] = m23 * (mC[5] * m23 + mC[12] * m43) + m43 * (mC[12] * m23 + mC[14] * m43);
  cv[10] = m00 * (mC[3] * m24 + mC[10] * m44);
  cv[11] = m10 * (mC[3] * m24 + mC[10] * m44);
  cv[12] = mC[4] * m24 + mC[11] * m44;
  cv[13] = m23 * (mC[5] * m24 + mC[12] * m44) + m43 * (mC[12] * m24 + mC[14] * m44);
  cv[14] = m24 * (mC[5] * m24 + mC[12] * m44) + m44 * (mC[12] * m24 + mC[14] * m44);
  cv[15] = m00 * (mC[6] * m35 + mC[10] * m45);
  cv[16] = m10 * (mC[6] * m35 + mC[10] * m45);
  cv[17] = mC[7] * m35 + mC[11] * m45;
  cv[18] = m23 * (mC[8] * m35 + mC[12] * m45) + m43 * (mC[13] * m35 + mC[14] * m45);
  cv[19] = m24 * (mC[8] * m35 + mC[12] * m45) + m44 * (mC[13] * m35 + mC[14] * m45);
  cv[20] = m35 * (mC[9] * m35 + mC[13] * m45) + m45 * (mC[13] * m35 + mC[14] * m45);

  return true;
}

#ifndef GPUCA_ALIGPUCODE
//______________________________________________________________
template <typename value_T>
std::string TrackParametrizationWithError<value_T>::asString() const
{
  return TrackParametrization<value_t>::asString() +
         fmt::format(
           "\n{:7s} {:+.3e}\n"
           "{:7s} {:+.3e} {:+.3e}\n"
           "{:7s} {:+.3e} {:+.3e} {:+.3e}\n"
           "{:7s} {:+.3e} {:+.3e} {:+.3e} {:+.3e}\n"
           "{:7s} {:+.3e} {:+.3e} {:+.3e} {:+.3e} {:+.3e}",
           "CovMat:", mC[kSigY2], "", mC[kSigZY], mC[kSigZ2], "", mC[kSigSnpY], mC[kSigSnpZ], mC[kSigSnp2], "", mC[kSigTglY],
           mC[kSigTglZ], mC[kSigTglSnp], mC[kSigTgl2], "", mC[kSigQ2PtY], mC[kSigQ2PtZ], mC[kSigQ2PtSnp], mC[kSigQ2PtTgl],
           mC[kSigQ2Pt2]);
}
#endif

//______________________________________________________________
template <typename value_T>
GPUd() void TrackParametrizationWithError<value_T>::print() const
{
  // print parameters
#ifndef GPUCA_ALIGPUCODE
  printf("%s\n", asString().c_str());
#endif
}

namespace o2::track
{
template class TrackParametrizationWithError<float>;
#ifndef GPUCA_GPUCODE_DEVICE
template class TrackParametrizationWithError<double>;
#endif
} // namespace o2::track
