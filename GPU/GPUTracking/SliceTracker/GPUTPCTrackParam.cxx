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

/// \file GPUTPCTrackParam.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCTrackLinearisation.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGeometry.h"

using namespace GPUCA_NAMESPACE::gpu;

//
// Circle in XY:
//
// kCLight = 0.000299792458;
// Kappa = -Bz*kCLight*QPt;
// R  = 1/fabsf(Kappa);
// Xc = X - sin(Phi)/Kappa;
// Yc = Y + cos(Phi)/Kappa;
//

MEM_CLASS_PRE()
GPUd() float MEM_LG(GPUTPCTrackParam)::GetDist2(const MEM_LG(GPUTPCTrackParam) & GPUrestrict() t) const
{
  // get squared distance between tracks

  float dx = GetX() - t.GetX();
  float dy = GetY() - t.GetY();
  float dz = GetZ() - t.GetZ();
  return dx * dx + dy * dy + dz * dz;
}

MEM_CLASS_PRE()
GPUd() float MEM_LG(GPUTPCTrackParam)::GetDistXZ2(const MEM_LG(GPUTPCTrackParam) & GPUrestrict() t) const
{
  // get squared distance between tracks in X&Z

  float dx = GetX() - t.GetX();
  float dz = GetZ() - t.GetZ();
  return dx * dx + dz * dz;
}

MEM_CLASS_PRE()
GPUd() float MEM_LG(GPUTPCTrackParam)::GetS(float x, float y, float Bz) const
{
  //* Get XY path length to the given point

  float k = GetKappa(Bz);
  float ex = GetCosPhi();
  float ey = GetSinPhi();
  x -= GetX();
  y -= GetY();
  float dS = x * ex + y * ey;
  if (CAMath::Abs(k) > 1.e-4f) {
    dS = CAMath::ATan2(k * dS, 1 + k * (x * ey - y * ex)) / k;
  }
  return dS;
}

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCTrackParam)::GetDCAPoint(float x, float y, float z, float& GPUrestrict() xp, float& GPUrestrict() yp, float& GPUrestrict() zp, float Bz) const
{
  //* Get the track point closest to the (x,y,z)

  float x0 = GetX();
  float y0 = GetY();
  float k = GetKappa(Bz);
  float ex = GetCosPhi();
  float ey = GetSinPhi();
  float dx = x - x0;
  float dy = y - y0;
  float ax = dx * k + ey;
  float ay = dy * k - ex;
  float a = sqrt(ax * ax + ay * ay);
  xp = x0 + (dx - ey * ((dx * dx + dy * dy) * k - 2 * (-dx * ey + dy * ex)) / (a + 1)) / a;
  yp = y0 + (dy + ex * ((dx * dx + dy * dy) * k - 2 * (-dx * ey + dy * ex)) / (a + 1)) / a;
  float s = GetS(x, y, Bz);
  zp = GetZ() + GetDzDs() * s;
  if (CAMath::Abs(k) > 1.e-2f) {
    float dZ = CAMath::Abs(GetDzDs() * CAMath::TwoPi() / k);
    if (dZ > .1f) {
      zp += CAMath::Nint((z - zp) / dZ) * dZ;
    }
  }
}

//*
//* Transport routines
//*

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::TransportToX(float x, GPUTPCTrackLinearisation& GPUrestrict() t0, float Bz, float maxSinPhi, float* GPUrestrict() DL)
{
  //* Transport the track parameters to X=x, using linearization at t0, and the field value Bz
  //* maxSinPhi is the max. allowed value for |t0.SinPhi()|
  //* linearisation of trajectory t0 is also transported to X=x,
  //* returns 1 if OK
  //*

  float ex = t0.CosPhi();
  float ey = t0.SinPhi();
  float k = -t0.QPt() * Bz;
  float dx = x - X();

  float ey1 = k * dx + ey;
  float ex1;

  // check for intersection with X=x

  if (CAMath::Abs(ey1) > maxSinPhi) {
    return 0;
  }

  ex1 = CAMath::Sqrt(1 - ey1 * ey1);
  if (ex < 0) {
    ex1 = -ex1;
  }

  float dx2 = dx * dx;
  float ss = ey + ey1;
  float cc = ex + ex1;

  if (CAMath::Abs(cc) < 1.e-4f || CAMath::Abs(ex) < 1.e-4f || CAMath::Abs(ex1) < 1.e-4f) {
    return 0;
  }

  float tg = ss / cc; // tanf((phi1+phi)/2)

  float dy = dx * tg;
  float dl = dx * CAMath::Sqrt(1 + tg * tg);

  if (cc < 0) {
    dl = -dl;
  }
  float dSin = dl * k / 2;
  if (dSin > 1) {
    dSin = 1;
  }
  if (dSin < -1) {
    dSin = -1;
  }
  float dS = (CAMath::Abs(k) > 1.e-4f) ? (2 * CAMath::ASin(dSin) / k) : dl;
  float dz = dS * t0.DzDs();

  if (DL) {
    *DL = -dS * CAMath::Sqrt(1 + t0.DzDs() * t0.DzDs());
  }

  float cci = 1.f / cc;
  float exi = 1.f / ex;
  float ex1i = 1.f / ex1;

  float d[5] = {0, 0, GetPar(2) - t0.SinPhi(), GetPar(3) - t0.DzDs(), GetPar(4) - t0.QPt()};

  // float H0[5] = { 1,0, h2,  0, h4 };
  // float H1[5] = { 0, 1, 0, dS,  0 };
  // float H2[5] = { 0, 0, 1,  0, dxBz };
  // float H3[5] = { 0, 0, 0,  1,  0 };
  // float H4[5] = { 0, 0, 0,  0,  1 };

  float h2 = dx * (1 + ey * ey1 + ex * ex1) * exi * ex1i * cci;
  float h4 = dx2 * (cc + ss * ey1 * ex1i) * cci * cci * (-Bz);
  float dxBz = dx * (-Bz);

  t0.SetCosPhi(ex1);
  t0.SetSinPhi(ey1);

  SetX(X() + dx);
  SetPar(0, Y() + dy + h2 * d[2] + h4 * d[4]);
  SetPar(1, Z() + dz + dS * d[3]);
  SetPar(2, t0.SinPhi() + d[2] + dxBz * d[4]);

  float c00 = mParam.mC[0];
  float c10 = mParam.mC[1];
  float c11 = mParam.mC[2];
  float c20 = mParam.mC[3];
  float c21 = mParam.mC[4];
  float c22 = mParam.mC[5];
  float c30 = mParam.mC[6];
  float c31 = mParam.mC[7];
  float c32 = mParam.mC[8];
  float c33 = mParam.mC[9];
  float c40 = mParam.mC[10];
  float c41 = mParam.mC[11];
  float c42 = mParam.mC[12];
  float c43 = mParam.mC[13];
  float c44 = mParam.mC[14];

  mParam.mC[0] = c00 + h2 * h2 * c22 + h4 * h4 * c44 + 2 * (h2 * c20 + h4 * c40 + h2 * h4 * c42);

  mParam.mC[1] = c10 + h2 * c21 + h4 * c41 + dS * (c30 + h2 * c32 + h4 * c43);
  mParam.mC[2] = c11 + 2 * dS * c31 + dS * dS * c33;

  mParam.mC[3] = c20 + h2 * c22 + h4 * c42 + dxBz * (c40 + h2 * c42 + h4 * c44);
  mParam.mC[4] = c21 + dS * c32 + dxBz * (c41 + dS * c43);
  mParam.mC[5] = c22 + 2 * dxBz * c42 + dxBz * dxBz * c44;

  mParam.mC[6] = c30 + h2 * c32 + h4 * c43;
  mParam.mC[7] = c31 + dS * c33;
  mParam.mC[8] = c32 + dxBz * c43;
  mParam.mC[9] = c33;

  mParam.mC[10] = c40 + h2 * c42 + h4 * c44;
  mParam.mC[11] = c41 + dS * c43;
  mParam.mC[12] = c42 + dxBz * c44;
  mParam.mC[13] = c43;
  mParam.mC[14] = c44;

  return 1;
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::TransportToX(float x, float sinPhi0, float cosPhi0, float Bz, float maxSinPhi)
{
  //* Transport the track parameters to X=x, using linearization at phi0 with 0 curvature,
  //* and the field value Bz
  //* maxSinPhi is the max. allowed value for |t0.SinPhi()|
  //* linearisation of trajectory t0 is also transported to X=x,
  //* returns 1 if OK
  //*

  float ex = cosPhi0;
  float ey = sinPhi0;
  float dx = x - X();

  if (CAMath::Abs(ex) < 1.e-4f) {
    return 0;
  }
  float exi = 1.f / ex;

  float dxBz = dx * (-Bz);
  float dS = dx * exi;
  float h2 = dS * exi * exi;
  float h4 = .5f * h2 * dxBz;

  // float H0[5] = { 1,0, h2,  0, h4 };
  // float H1[5] = { 0, 1, 0, dS,  0 };
  // float H2[5] = { 0, 0, 1,  0, dxBz };
  // float H3[5] = { 0, 0, 0,  1,  0 };
  // float H4[5] = { 0, 0, 0,  0,  1 };

  float sinPhi = SinPhi() + dxBz * QPt();
  if (maxSinPhi > 0 && CAMath::Abs(sinPhi) > maxSinPhi) {
    return 0;
  }

  SetX(X() + dx);
  SetPar(0, GetPar(0) + dS * ey + h2 * (SinPhi() - ey) + h4 * QPt());
  SetPar(1, GetPar(1) + dS * DzDs());
  SetPar(2, sinPhi);

  float c00 = mParam.mC[0];
  float c10 = mParam.mC[1];
  float c11 = mParam.mC[2];
  float c20 = mParam.mC[3];
  float c21 = mParam.mC[4];
  float c22 = mParam.mC[5];
  float c30 = mParam.mC[6];
  float c31 = mParam.mC[7];
  float c32 = mParam.mC[8];
  float c33 = mParam.mC[9];
  float c40 = mParam.mC[10];
  float c41 = mParam.mC[11];
  float c42 = mParam.mC[12];
  float c43 = mParam.mC[13];
  float c44 = mParam.mC[14];

  // This is not the correct propagation!!! The max ensures the positional error does not decrease during track finding.
  // A different propagation method is used for the fit.
  mParam.mC[0] = CAMath::Max(c00, c00 + h2 * h2 * c22 + h4 * h4 * c44 + 2 * (h2 * c20 + h4 * c40 + h2 * h4 * c42));

  mParam.mC[1] = c10 + h2 * c21 + h4 * c41 + dS * (c30 + h2 * c32 + h4 * c43);
  mParam.mC[2] = CAMath::Max(c11, c11 + 2 * dS * c31 + dS * dS * c33);

  mParam.mC[3] = c20 + h2 * c22 + h4 * c42 + dxBz * (c40 + h2 * c42 + h4 * c44);
  mParam.mC[4] = c21 + dS * c32 + dxBz * (c41 + dS * c43);
  mParam.mC[5] = c22 + 2 * dxBz * c42 + dxBz * dxBz * c44;

  mParam.mC[6] = c30 + h2 * c32 + h4 * c43;
  mParam.mC[7] = c31 + dS * c33;
  mParam.mC[8] = c32 + dxBz * c43;
  mParam.mC[9] = c33;

  mParam.mC[10] = c40 + h2 * c42 + h4 * c44;
  mParam.mC[11] = c41 + dS * c43;
  mParam.mC[12] = c42 + dxBz * c44;
  mParam.mC[13] = c43;
  mParam.mC[14] = c44;

  return 1;
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::TransportToX(float x, float Bz, float maxSinPhi)
{
  //* Transport the track parameters to X=x
  GPUTPCTrackLinearisation t0(*this);
  return TransportToX(x, t0, Bz, maxSinPhi);
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::TransportToXWithMaterial(float x, GPUTPCTrackLinearisation& GPUrestrict() t0, GPUTPCTrackFitParam& GPUrestrict() par, float Bz, float maxSinPhi)
{
  //* Transport the track parameters to X=x  taking into account material budget

  static constexpr float kRho = 1.025e-3f;   // [g/cm^3]
  static constexpr float kRadLen = 28811.7f; //[cm]

  static constexpr float kRadLenInv = 1. / kRadLen;
  float dl;

  if (!TransportToX(x, t0, Bz, maxSinPhi, &dl)) {
    return 0;
  }

  CorrectForMeanMaterial(dl * kRadLenInv, dl * kRho, par);
  return 1;
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::TransportToXWithMaterial(float x, GPUTPCTrackFitParam& GPUrestrict() par, float Bz, float maxSinPhi)
{
  //* Transport the track parameters to X=x  taking into account material budget

  GPUTPCTrackLinearisation t0(*this);
  return TransportToXWithMaterial(x, t0, par, Bz, maxSinPhi);
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::TransportToXWithMaterial(float x, float Bz, float maxSinPhi)
{
  //* Transport the track parameters to X=x taking into account material budget

  GPUTPCTrackFitParam par;
  CalculateFitParameters(par);
  return TransportToXWithMaterial(x, par, Bz, maxSinPhi);
}

//*
//*  Multiple scattering and energy losses
//*
MEM_CLASS_PRE()
GPUd() float MEM_LG(GPUTPCTrackParam)::BetheBlochGeant(float bg2, float kp0, float kp1, float kp2, float kp3, float kp4)
{
  //
  // This is the parameterization of the Bethe-Bloch formula inspired by Geant.
  //
  // bg2  - (beta*gamma)^2
  // kp0 - density [g/cm^3]
  // kp1 - density effect first junction point
  // kp2 - density effect second junction point
  // kp3 - mean excitation energy [GeV]
  // kp4 - mean Z/A
  //
  // The default values for the kp* parameters are for silicon.
  // The returned value is in [GeV/(g/cm^2)].
  //

  const float mK = 0.307075e-3f; // [GeV*cm^2/g]
  const float me = 0.511e-3f;    // [GeV/c^2]
  const float rho = kp0;
  const float x0 = kp1 * 2.303f;
  const float x1 = kp2 * 2.303f;
  const float mI = kp3;
  const float mZA = kp4;
  const float maxT = 2 * me * bg2; // neglecting the electron mass

  //*** Density effect
  float d2 = 0.f;
  const float x = 0.5f * CAMath::Log(bg2);
  const float lhwI = CAMath::Log(28.816f * 1e-9f * CAMath::Sqrt(rho * mZA) / mI);
  if (x > x1) {
    d2 = lhwI + x - 0.5f;
  } else if (x > x0) {
    const float r = (x1 - x) / (x1 - x0);
    d2 = lhwI + x - 0.5f + (0.5f - lhwI - x0) * r * r * r;
  }

  return mK * mZA * (1 + bg2) / bg2 * (0.5f * CAMath::Log(2 * me * bg2 * maxT / (mI * mI)) - bg2 / (1 + bg2) - d2);
}

MEM_CLASS_PRE()
GPUd() float MEM_LG(GPUTPCTrackParam)::BetheBlochSolid(float bg)
{
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula,
  // reasonable for solid materials.
  // All the parameters are, in fact, for Si.
  // The returned value is in [GeV]
  //------------------------------------------------------------------

  return BetheBlochGeant(bg);
}

MEM_CLASS_PRE()
GPUd() float MEM_LG(GPUTPCTrackParam)::BetheBlochGas(float bg)
{
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula,
  // reasonable for gas materials.
  // All the parameters are, in fact, for Ne.
  // The returned value is in [GeV]
  //------------------------------------------------------------------

  const float rho = 0.9e-3f;
  const float x0 = 2.f;
  const float x1 = 4.f;
  const float mI = 140.e-9f;
  const float mZA = 0.49555f;

  return BetheBlochGeant(bg, rho, x0, x1, mI, mZA);
}

MEM_CLASS_PRE()
GPUd() float MEM_LG(GPUTPCTrackParam)::ApproximateBetheBloch(float beta2)
{
  //------------------------------------------------------------------
  // This is an approximation of the Bethe-Bloch formula with
  // the density effect taken into account at beta*gamma > 3.5
  // (the approximation is reasonable only for solid materials)
  //------------------------------------------------------------------
  if (beta2 >= 1) {
    return 0;
  }

  if (beta2 / (1 - beta2) > 3.5f * 3.5f) {
    return 0.153e-3f / beta2 * (log(3.5f * 5940) + 0.5f * log(beta2 / (1 - beta2)) - beta2);
  }
  return 0.153e-3f / beta2 * (log(5940 * beta2 / (1 - beta2)) - beta2);
}

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCTrackParam)::CalculateFitParameters(GPUTPCTrackFitParam& par, float mass)
{
  //*!

  float qpt = GetPar(4);
  if (mParam.mC[14] >= 1.f) {
    qpt = 1.f / 0.35f;
  }

  float p2 = (1.f + GetPar(3) * GetPar(3));
  float k2 = qpt * qpt;
  float mass2 = mass * mass;
  float beta2 = p2 / (p2 + mass2 * k2);

  float pp2 = (k2 > 1.e-8f) ? p2 / k2 : 10000; // impuls 2

  // par.bethe = BetheBlochGas( pp2/mass2);
  par.bethe = ApproximateBetheBloch(pp2 / mass2);
  par.e = CAMath::Sqrt(pp2 + mass2);
  par.theta2 = 14.1f * 14.1f / (beta2 * pp2 * 1e6f);
  par.EP2 = par.e / pp2;

  // Approximate energy loss fluctuation (M.Ivanov)

  const float knst = 0.0007f; // To be tuned.
  par.sigmadE2 = knst * par.EP2 * qpt;
  par.sigmadE2 = par.sigmadE2 * par.sigmadE2;

  par.k22 = (1.f + GetPar(3) * GetPar(3));
  par.k33 = par.k22 * par.k22;
  par.k43 = 0;
  par.k44 = GetPar(3) * GetPar(3) * k2;
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::CorrectForMeanMaterial(float xOverX0, float xTimesRho, const GPUTPCTrackFitParam& par)
{
  //------------------------------------------------------------------
  // This function corrects the track parameters for the crossed material.
  // "xOverX0"   - X/X0, the thickness in units of the radiation length.
  // "xTimesRho" - is the product length*density (g/cm^2).
  //------------------------------------------------------------------

  float& mC22 = mParam.mC[5];
  float& mC33 = mParam.mC[9];
  float& mC40 = mParam.mC[10];
  float& mC41 = mParam.mC[11];
  float& mC42 = mParam.mC[12];
  float& mC43 = mParam.mC[13];
  float& mC44 = mParam.mC[14];

  // Energy losses************************

  float dE = par.bethe * xTimesRho;
  if (CAMath::Abs(dE) > 0.3f * par.e) {
    return 0; // 30% energy loss is too much!
  }
  float corr = (1.f - par.EP2 * dE);
  if (corr < 0.3f || corr > 1.3f) {
    return 0;
  }

  SetPar(4, GetPar(4) * corr);
  mC40 *= corr;
  mC41 *= corr;
  mC42 *= corr;
  mC43 *= corr;
  mC44 *= corr * corr;
  mC44 += par.sigmadE2 * CAMath::Abs(dE);

  // Multiple scattering******************

  float theta2 = par.theta2 * CAMath::Abs(xOverX0);
  mC22 += theta2 * par.k22 * (1.f - GetPar(2)) * (1.f + GetPar(2));
  mC33 += theta2 * par.k33;
  mC43 += theta2 * par.k43;
  mC44 += theta2 * par.k44;

  return 1;
}

//*
//* Rotation
//*
MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::Rotate(float alpha, float maxSinPhi)
{
  //* Rotate the coordinate system in XY on the angle alpha

  float cA = CAMath::Cos(alpha);
  float sA = CAMath::Sin(alpha);
  float x = X(), y = Y(), sP = SinPhi(), cP = GetCosPhi();
  float cosPhi = cP * cA + sP * sA;
  float sinPhi = -cP * sA + sP * cA;

  if (CAMath::Abs(sinPhi) > maxSinPhi || CAMath::Abs(cosPhi) < 1.e-2f || CAMath::Abs(cP) < 1.e-2f) {
    return 0;
  }

  float j0 = cP / cosPhi;
  float j2 = cosPhi / cP;

  SetX(x * cA + y * sA);
  SetY(-x * sA + y * cA);
  SetSignCosPhi(cosPhi);
  SetSinPhi(sinPhi);

  // float J[5][5] = { { j0, 0, 0,  0,  0 }, // Y
  //                      {  0, 1, 0,  0,  0 }, // Z
  //                      {  0, 0, j2, 0,  0 }, // SinPhi
  //                    {  0, 0, 0,  1,  0 }, // DzDs
  //                    {  0, 0, 0,  0,  1 } }; // Kappa
  // cout<<"alpha="<<alpha<<" "<<x<<" "<<y<<" "<<sP<<" "<<cP<<" "<<j0<<" "<<j2<<endl;
  // cout<<"      "<<mParam.mC[0]<<" "<<mParam.mC[1]<<" "<<mParam.mC[6]<<" "<<mParam.mC[10]<<" "<<mParam.mC[4]<<" "<<mParam.mC[5]<<" "<<mParam.mC[8]<<" "<<mParam.mC[12]<<endl;
  mParam.mC[0] *= j0 * j0;
  mParam.mC[1] *= j0;
  mParam.mC[3] *= j0;
  mParam.mC[6] *= j0;
  mParam.mC[10] *= j0;

  mParam.mC[3] *= j2;
  mParam.mC[4] *= j2;
  mParam.mC[5] *= j2 * j2;
  mParam.mC[8] *= j2;
  mParam.mC[12] *= j2;

  if (cosPhi < 0) {
    SetSinPhi(-SinPhi());
    SetDzDs(-DzDs());
    SetQPt(-QPt());
    mParam.mC[3] = -mParam.mC[3];
    mParam.mC[4] = -mParam.mC[4];
    mParam.mC[6] = -mParam.mC[6];
    mParam.mC[7] = -mParam.mC[7];
    mParam.mC[10] = -mParam.mC[10];
    mParam.mC[11] = -mParam.mC[11];
  }

  // cout<<"      "<<mParam.mC[0]<<" "<<mParam.mC[1]<<" "<<mParam.mC[6]<<" "<<mParam.mC[10]<<" "<<mParam.mC[4]<<" "<<mParam.mC[5]<<" "<<mParam.mC[8]<<" "<<mParam.mC[12]<<endl;
  return 1;
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::Rotate(float alpha, GPUTPCTrackLinearisation& t0, float maxSinPhi)
{
  //* Rotate the coordinate system in XY on the angle alpha

  float cA = CAMath::Cos(alpha);
  float sA = CAMath::Sin(alpha);
  float x0 = X(), y0 = Y(), sP = t0.SinPhi(), cP = t0.CosPhi();
  float cosPhi = cP * cA + sP * sA;
  float sinPhi = -cP * sA + sP * cA;

  if (CAMath::Abs(sinPhi) > maxSinPhi || CAMath::Abs(cosPhi) < 1.e-2f || CAMath::Abs(cP) < 1.e-2f) {
    return 0;
  }

  // float J[5][5] = { { j0, 0, 0,  0,  0 }, // Y
  //                    {  0, 1, 0,  0,  0 }, // Z
  //                    {  0, 0, j2, 0,  0 }, // SinPhi
  //                  {  0, 0, 0,  1,  0 }, // DzDs
  //                  {  0, 0, 0,  0,  1 } }; // Kappa

  float j0 = cP / cosPhi;
  float j2 = cosPhi / cP;
  float d[2] = {Y() - y0, SinPhi() - sP};

  SetX(x0 * cA + y0 * sA);
  SetY(-x0 * sA + y0 * cA + j0 * d[0]);
  t0.SetCosPhi(cosPhi);
  t0.SetSinPhi(sinPhi);

  SetSinPhi(sinPhi + j2 * d[1]);

  mParam.mC[0] *= j0 * j0;
  mParam.mC[1] *= j0;
  mParam.mC[3] *= j0;
  mParam.mC[6] *= j0;
  mParam.mC[10] *= j0;

  mParam.mC[3] *= j2;
  mParam.mC[4] *= j2;
  mParam.mC[5] *= j2 * j2;
  mParam.mC[8] *= j2;
  mParam.mC[12] *= j2;

  return 1;
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::Filter(float y, float z, float err2Y, float err2Z, float maxSinPhi, bool paramOnly)
{
  //* Add the y,z measurement with the Kalman filter

  float c00 = mParam.mC[0], c11 = mParam.mC[2], c20 = mParam.mC[3], c31 = mParam.mC[7], c40 = mParam.mC[10];

  err2Y += c00;
  err2Z += c11;

  float z0 = y - GetPar(0), z1 = z - GetPar(1);

  if (err2Y < 1.e-8f || err2Z < 1.e-8f) {
    return 0;
  }

  float mS0 = 1.f / err2Y;
  float mS2 = 1.f / err2Z;

  // K = CHtS

  float k00, k11, k20, k31, k40;

  k00 = c00 * mS0;
  k20 = c20 * mS0;
  k40 = c40 * mS0;

  k11 = c11 * mS2;
  k31 = c31 * mS2;

  float sinPhi = GetPar(2) + k20 * z0;

  if (maxSinPhi > 0 && CAMath::Abs(sinPhi) >= maxSinPhi) {
    return 0;
  }

  SetPar(0, GetPar(0) + k00 * z0);
  SetPar(1, GetPar(1) + k11 * z1);
  SetPar(2, sinPhi);
  SetPar(3, GetPar(3) + k31 * z1);
  SetPar(4, GetPar(4) + k40 * z0);
  if (paramOnly) {
    return true;
  }

  mNDF += 2;
  mChi2 += mS0 * z0 * z0 + mS2 * z1 * z1;

  mParam.mC[0] -= k00 * c00;
  mParam.mC[3] -= k20 * c00;
  mParam.mC[5] -= k20 * c20;
  mParam.mC[10] -= k40 * c00;
  mParam.mC[12] -= k40 * c20;
  mParam.mC[14] -= k40 * c40;

  mParam.mC[2] -= k11 * c11;
  mParam.mC[7] -= k31 * c11;
  mParam.mC[9] -= k31 * c31;

  return 1;
}

MEM_CLASS_PRE()
GPUd() bool MEM_LG(GPUTPCTrackParam)::CheckNumericalQuality() const
{
  //* Check that the track parameters and covariance matrix are reasonable

  bool ok = CAMath::Finite(GetX()) && CAMath::Finite(mSignCosPhi) && CAMath::Finite(mChi2);

  const float* c = Cov();
  for (int i = 0; i < 15; i++) {
    ok = ok && CAMath::Finite(c[i]);
  }
  for (int i = 0; i < 5; i++) {
    ok = ok && CAMath::Finite(Par()[i]);
  }

  if (c[0] <= 0 || c[2] <= 0 || c[5] <= 0 || c[9] <= 0 || c[14] <= 0) {
    ok = 0;
  }
  if (c[0] > 5.f || c[2] > 5.f || c[5] > 2.f || c[9] > 2
      //|| ( CAMath::Abs( QPt() ) > 1.e-2 && c[14] > 2. )
  ) {
    ok = 0;
  }

  if (CAMath::Abs(SinPhi()) > GPUCA_MAX_SIN_PHI) {
    ok = 0;
  }
  if (CAMath::Abs(QPt()) > 1.f / 0.05f) {
    ok = 0;
  }
  if (ok) {
    ok = ok && (c[1] * c[1] <= c[2] * c[0]) && (c[3] * c[3] <= c[5] * c[0]) && (c[4] * c[4] <= c[5] * c[2]) && (c[6] * c[6] <= c[9] * c[0]) && (c[7] * c[7] <= c[9] * c[2]) && (c[8] * c[8] <= c[9] * c[5]) && (c[10] * c[10] <= c[14] * c[0]) && (c[11] * c[11] <= c[14] * c[2]) &&
         (c[12] * c[12] <= c[14] * c[5]) && (c[13] * c[13] <= c[14] * c[9]);
  }
  return ok;
}

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCTrackParam)::ConstrainZ(float& z, int sector, float& z0, float& lastZ)
{
  if (sector < GPUCA_NSLICES / 2) {
    if (z < 0) {
      mParam.mZOffset += z;
      mParam.mP[1] -= z;
      z0 -= z;
      lastZ -= z;
      z = 0;
    } else if (z > GPUTPCGeometry::TPCLength()) {
      float shift = z - GPUTPCGeometry::TPCLength();
      mParam.mZOffset += shift;
      mParam.mP[1] -= shift;
      z0 -= shift;
      lastZ -= shift;
      z = GPUTPCGeometry::TPCLength();
    }
  } else {
    if (z > 0) {
      mParam.mZOffset += z;
      mParam.mP[1] -= z;
      z0 -= z;
      lastZ -= z;
      z = 0;
    } else if (z < -GPUTPCGeometry::TPCLength()) {
      float shift = -GPUTPCGeometry::TPCLength() - z;
      mParam.mZOffset -= shift;
      mParam.mP[1] += shift;
      z0 += shift;
      lastZ += shift;
      z = -GPUTPCGeometry::TPCLength();
    }
  }
}

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCTrackParam)::ShiftZ(float z1, float z2, float x1, float x2, float bz, float defaultZOffsetOverR)
{
  const float r1 = CAMath::Max(0.0001f, CAMath::Abs(mParam.mP[4] * bz));

  const float dist2 = mParam.mX * mParam.mX + mParam.mP[0] * mParam.mP[0];
  const float dist1r2 = dist2 * r1 * r1;
  float deltaZ = 0.f;
  bool beamlineReached = false;
  if (dist1r2 < 4) {
    const float alpha = CAMath::ACos(1 - 0.5f * dist1r2); // Angle of a circle, such that |(cosa, sina) - (1,0)| == dist
    const float beta = CAMath::ATan2(mParam.mP[0], mParam.mX);
    const int comp = mParam.mP[2] > CAMath::Sin(beta);
    const float sinab = CAMath::Sin((comp ? 0.5f : -0.5f) * alpha + beta); // Angle of circle through origin and track position, to be compared to Snp
    const float res = CAMath::Abs(sinab - mParam.mP[2]);

    if (res < 0.2) {
      const float r = 1.f / r1;
      const float dS = alpha * r;
      float z0 = dS * mParam.mP[3];
      if (CAMath::Abs(z0) > GPUTPCGeometry::TPCLength()) {
        z0 = z0 > 0 ? GPUTPCGeometry::TPCLength() : -GPUTPCGeometry::TPCLength();
      }
      deltaZ = mParam.mP[1] - z0;
      beamlineReached = true;
    }
  }

  if (!beamlineReached) {
    float basez, basex;
    if (CAMath::Abs(z1) < CAMath::Abs(z2)) {
      basez = z1;
      basex = x1;
    } else {
      basez = z2;
      basex = x2;
    }
    float refZ = ((basez > 0) ? defaultZOffsetOverR : -defaultZOffsetOverR) * basex;
    deltaZ = basez - refZ - mParam.mZOffset;
  }
  mParam.mZOffset += deltaZ;
  mParam.mP[1] -= deltaZ;
  deltaZ = 0;
  float zMax = CAMath::Max(z1, z2);
  float zMin = CAMath::Min(z1, z2);
  if (zMin < 0 && zMin - mParam.mZOffset < -GPUTPCGeometry::TPCLength()) {
    deltaZ = zMin - mParam.mZOffset + GPUTPCGeometry::TPCLength();
  } else if (zMax > 0 && zMax - mParam.mZOffset > GPUTPCGeometry::TPCLength()) {
    deltaZ = zMax - mParam.mZOffset - GPUTPCGeometry::TPCLength();
  }
  if (zMin < 0 && zMax - (mParam.mZOffset + deltaZ) > 0) {
    deltaZ = zMax - mParam.mZOffset;
  } else if (zMax > 0 && zMin - (mParam.mZOffset + deltaZ) < 0) {
    deltaZ = zMin - mParam.mZOffset;
  }
  mParam.mZOffset += deltaZ;
  mParam.mP[1] -= deltaZ;
}

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCTrackParam)::Print() const
{
  //* print parameters

#if !defined(GPUCA_GPUCODE)
  std::cout << "track: x=" << GetX() << " c=" << GetSignCosPhi() << ", P= " << GetY() << " " << GetZ() << " " << GetSinPhi() << " " << GetDzDs() << " " << GetQPt() << std::endl;
  std::cout << "errs2: " << Err2Y() << " " << Err2Z() << " " << Err2SinPhi() << " " << Err2DzDs() << " " << Err2QPt() << std::endl;
#endif
}

MEM_CLASS_PRE()
GPUd() int MEM_LG(GPUTPCTrackParam)::GetPropagatedYZ(float bz, float x, float& projY, float& projZ) const
{
  float k = mParam.mP[4] * bz;
  float dx = x - mParam.mX;
  float ey = mParam.mP[2];
  float ex = CAMath::Sqrt(1 - ey * ey);
  if (SignCosPhi() < 0) {
    ex = -ex;
  }
  float ey1 = ey - k * dx;
  if (CAMath::Abs(ey1) > GPUCA_MAX_SIN_PHI) {
    return 0;
  }
  float ss = ey + ey1;
  float ex1 = CAMath::Sqrt(1.f - ey1 * ey1);
  if (SignCosPhi() < 0) {
    ex1 = -ex1;
  }
  float cc = ex + ex1;
  float dxcci = dx / cc;
  float dy = dxcci * ss;
  float norm2 = 1.f + ey * ey1 + ex * ex1;
  float dl = dxcci * CAMath::Sqrt(norm2 + norm2);
  float dS;
  {
    float dSin = 0.5f * k * dl;
    float a = dSin * dSin;
    const float k2 = 1.f / 6.f;
    const float k4 = 3.f / 40.f;
    dS = dl + dl * a * (k2 + a * (k4));
  }
  float dz = dS * mParam.mP[3];
  projY = mParam.mP[0] + dy;
  projZ = mParam.mP[1] + dz;
  return 1;
}
