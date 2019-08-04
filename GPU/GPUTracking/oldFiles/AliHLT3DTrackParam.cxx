// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliHLT3DTrackParam.cxx
/// \author Sergey Gorbunov

#include "AliHLT3DTrackParam.h"
#include "TMath.h"

ClassImp(AliHLT3DTrackParam);

//* Transport utilities

double AliHLT3DTrackParam::GetDStoPoint(double Bz, const double xyz[3], const double* T0) const
{
  //* Get DS = Path/Momentum to a certain space point for Bz field

  double q = fSignQ;
  if (!T0) {
    T0 = mParam;
  } else {
    q = T0[6];
  }

  const double kCLight = 0.000299792458;
  double bq = Bz * q * kCLight;
  double pt2 = T0[3] * T0[3] + T0[4] * T0[4];
  if (pt2 < 1.e-4) {
    return 0;
  }
  double dx = xyz[0] - T0[0];
  double dy = xyz[1] - T0[1];
  double a = dx * T0[3] + dy * T0[4];
  double dS = 0;
  if (TMath::Abs(bq) < 1.e-8) {
    dS = a / pt2;
  } else {
    dS = TMath::ATan2(bq * a, pt2 + bq * (dy * T0[3] - dx * T0[4])) / bq;
  }
  return dS;
}

void AliHLT3DTrackParam::TransportToDS(double Bz, double DS, double* T0)
{
  //* Transport the particle on DS = Path/Momentum, for Bz field

  double tmp[7];
  if (!T0) {
    T0 = tmp;
    T0[0] = mParam[0];
    T0[1] = mParam[1];
    T0[2] = mParam[2];
    T0[3] = mParam[3];
    T0[4] = mParam[4];
    T0[5] = mParam[5];
    T0[6] = fSignQ;
  }
  const double kCLight = 0.000299792458;
  Bz = Bz * T0[6] * kCLight;
  double bs = Bz * DS;
  double s = TMath::Sin(bs), c = TMath::Cos(bs);
  double sB, cB;
  if (TMath::Abs(bs) > 1.e-10) {
    sB = s / Bz;
    cB = (1 - c) / Bz;
  } else {
    const Double_t kOvSqr6 = 1. / TMath::Sqrt(6.);
    sB = (1. - bs * kOvSqr6) * (1. + bs * kOvSqr6) * DS;
    cB = .5 * sB * bs;
  }

  double px = T0[3];
  double py = T0[4];
  double pz = T0[5];

  double d[6] = {mParam[0] - T0[0], mParam[1] - T0[1], mParam[2] - T0[2], mParam[3] - T0[3], mParam[4] - T0[4], mParam[5] - T0[5]};

  T0[0] = T0[0] + sB * px + cB * py;
  T0[1] = T0[1] - cB * px + sB * py;
  T0[2] = T0[2] + DS * pz;
  T0[3] = c * px + s * py;
  T0[4] = -s * px + c * py;
  T0[5] = T0[5];

  // clang-format off
  double mJ[6][6] = { {1, 0, 0, sB, cB, 0, },
    {0, 1, 0,  -cB, sB,  0, },
    {0, 0, 1,    0,  0, DS, },
    {0, 0, 0,    c,  s,  0, },
    {0, 0, 0,   -s,  c,  0, },
    {0, 0, 0,    0,  0,  1, }};
  // clang-format on

  for (int i = 0; i < 6; i++) {
    mParam[i] = T0[i];
    for (int j = 0; j < 6; j++) {
      mParam[i] += mJ[i][j] * d[j];
    }
  }

  double mA[6][6];
  for (int k = 0, i = 0; i < 6; i++) {
    for (int j = 0; j <= i; j++, k++) {
      mA[i][j] = mA[j][i] = fCov[k];
    }
  }

  double mJC[6][6];
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      mJC[i][j] = 0;
      for (int k = 0; k < 6; k++) {
        mJC[i][j] += mJ[i][k] * mA[k][j];
      }
    }
  }

  for (int k = 0, i = 0; i < 6; i++) {
    for (int j = 0; j <= i; j++, k++) {
      fCov[k] = 0;
      for (int l = 0; l < 6; l++) {
        fCov[k] += mJC[i][l] * mJ[j][l];
      }
    }
  }
}

//* Fit utilities

void AliHLT3DTrackParam::InitializeCovarianceMatrix()
{
  //* Initialization of covariance matrix

  for (int i = 0; i < 21; i++) {
    fCov[i] = 0;
  }
  fSignQ = 0;
  fCov[0] = fCov[2] = fCov[5] = 100.;
  fCov[9] = fCov[14] = fCov[20] = 10000.;
  fChi2 = 0;
  fNDF = -5;
}

void AliHLT3DTrackParam::GetGlueMatrix(const double xyz[3], double G[6], const double* T0) const
{
  //* !

  if (!T0) {
    T0 = mParam;
  }

  double dx = xyz[0] - T0[0], dy = xyz[1] - T0[1], dz = xyz[2] - T0[2];
  double px2 = T0[3] * T0[3], py2 = T0[4] * T0[4], pz2 = T0[5] * T0[5];
  double s2 = (dx * dx + dy * dy + dz * dz);
  double p2 = px2 + py2 + pz2;
  if (p2 > 1.e-4) {
    s2 /= p2;
  }
  double x = T0[3] * s2;
  double xx = px2 * s2, xy = x * T0[4], xz = x * T0[5], yy = py2 * s2, yz = T0[4] * T0[5] * s2;
  G[0] = xx;
  G[1] = xy;
  G[2] = yy;
  G[3] = xz;
  G[4] = yz;
  G[5] = pz2 * s2;
}

void AliHLT3DTrackParam::Filter(const double m[3], const double V[6], const double G[6])
{
  //* !

  // clang-format off
  double
    c00 = fCov[0],
    c10 = fCov[1], c11 = fCov[2],
    c20 = fCov[3], c21 = fCov[4], c22 = fCov[5],
    c30 = fCov[6], c31 = fCov[7], c32 = fCov[8],
    c40 = fCov[10], c41 = fCov[11], c42 = fCov[12],
    c50 = fCov[15], c51 = fCov[16], c52 = fCov[17];
  // clang-format on

  double z0 = m[0] - mParam[0], z1 = m[1] - mParam[1], z2 = m[2] - mParam[2];

  double mS[6] = {c00 + V[0] + G[0], c10 + V[1] + G[1], c11 + V[2] + G[2], c20 + V[3] + G[3], c21 + V[4] + G[4], c22 + V[5] + G[5]};
  double mSi[6];
  mSi[0] = mS[4] * mS[4] - mS[2] * mS[5];
  mSi[1] = mS[1] * mS[5] - mS[3] * mS[4];
  mSi[3] = mS[2] * mS[3] - mS[1] * mS[4];
  double det = 1. / (mS[0] * mSi[0] + mS[1] * mSi[1] + mS[3] * mSi[3]);
  mSi[0] *= det;
  mSi[1] *= det;
  mSi[3] *= det;
  mSi[2] = (mS[3] * mS[3] - mS[0] * mS[5]) * det;
  mSi[4] = (mS[0] * mS[4] - mS[1] * mS[3]) * det;
  mSi[5] = (mS[1] * mS[1] - mS[0] * mS[2]) * det;

  fNDF += 2;
  fChi2 += (+(mSi[0] * z0 + mSi[1] * z1 + mSi[3] * z2) * z0 + (mSi[1] * z0 + mSi[2] * z1 + mSi[4] * z2) * z1 + (mSi[3] * z0 + mSi[4] * z1 + mSi[5] * z2) * z2);

  double k0, k1, k2; // k = CHtS

  k0 = c00 * mSi[0] + c10 * mSi[1] + c20 * mSi[3];
  k1 = c00 * mSi[1] + c10 * mSi[2] + c20 * mSi[4];
  k2 = c00 * mSi[3] + c10 * mSi[4] + c20 * mSi[5];

  mParam[0] += k0 * z0 + k1 * z1 + k2 * z2;
  fCov[0] -= k0 * c00 + k1 * c10 + k2 * c20;

  k0 = c10 * mSi[0] + c11 * mSi[1] + c21 * mSi[3];
  k1 = c10 * mSi[1] + c11 * mSi[2] + c21 * mSi[4];
  k2 = c10 * mSi[3] + c11 * mSi[4] + c21 * mSi[5];

  mParam[1] += k0 * z0 + k1 * z1 + k2 * z2;
  fCov[1] -= k0 * c00 + k1 * c10 + k2 * c20;
  fCov[2] -= k0 * c10 + k1 * c11 + k2 * c21;

  k0 = c20 * mSi[0] + c21 * mSi[1] + c22 * mSi[3];
  k1 = c20 * mSi[1] + c21 * mSi[2] + c22 * mSi[4];
  k2 = c20 * mSi[3] + c21 * mSi[4] + c22 * mSi[5];

  mParam[2] += k0 * z0 + k1 * z1 + k2 * z2;
  fCov[3] -= k0 * c00 + k1 * c10 + k2 * c20;
  fCov[4] -= k0 * c10 + k1 * c11 + k2 * c21;
  fCov[5] -= k0 * c20 + k1 * c21 + k2 * c22;

  k0 = c30 * mSi[0] + c31 * mSi[1] + c32 * mSi[3];
  k1 = c30 * mSi[1] + c31 * mSi[2] + c32 * mSi[4];
  k2 = c30 * mSi[3] + c31 * mSi[4] + c32 * mSi[5];

  mParam[3] += k0 * z0 + k1 * z1 + k2 * z2;
  fCov[6] -= k0 * c00 + k1 * c10 + k2 * c20;
  fCov[7] -= k0 * c10 + k1 * c11 + k2 * c21;
  fCov[8] -= k0 * c20 + k1 * c21 + k2 * c22;
  fCov[9] -= k0 * c30 + k1 * c31 + k2 * c32;

  k0 = c40 * mSi[0] + c41 * mSi[1] + c42 * mSi[3];
  k1 = c40 * mSi[1] + c41 * mSi[2] + c42 * mSi[4];
  k2 = c40 * mSi[3] + c41 * mSi[4] + c42 * mSi[5];

  mParam[4] += k0 * z0 + k1 * z1 + k2 * z2;
  fCov[10] -= k0 * c00 + k1 * c10 + k2 * c20;
  fCov[11] -= k0 * c10 + k1 * c11 + k2 * c21;
  fCov[12] -= k0 * c20 + k1 * c21 + k2 * c22;
  fCov[13] -= k0 * c30 + k1 * c31 + k2 * c32;
  fCov[14] -= k0 * c40 + k1 * c41 + k2 * c42;

  k0 = c50 * mSi[0] + c51 * mSi[1] + c52 * mSi[3];
  k1 = c50 * mSi[1] + c51 * mSi[2] + c52 * mSi[4];
  k2 = c50 * mSi[3] + c51 * mSi[4] + c52 * mSi[5];

  mParam[5] += k0 * z0 + k1 * z1 + k2 * z2;
  fCov[15] -= k0 * c00 + k1 * c10 + k2 * c20;
  fCov[16] -= k0 * c10 + k1 * c11 + k2 * c21;
  fCov[17] -= k0 * c20 + k1 * c21 + k2 * c22;
  fCov[18] -= k0 * c30 + k1 * c31 + k2 * c32;
  fCov[19] -= k0 * c40 + k1 * c41 + k2 * c42;
  fCov[20] -= k0 * c50 + k1 * c51 + k2 * c52;

  // fit charge

  double px = mParam[3];
  double py = mParam[4];
  double pz = mParam[5];

  double p = TMath::Sqrt(px * px + py * py + pz * pz);
  double pi = 1. / p;
  double qp = fSignQ * pi;
  double qp3 = qp * pi * pi;
  double c60 = qp3 * (c30 + c40 + c50), c61 = qp3 * (c31 + c41 + c51), c62 = qp3 * (c32 + c42 + c52);

  k0 = c60 * mSi[0] + c61 * mSi[1] + c62 * mSi[3];
  k1 = c60 * mSi[1] + c61 * mSi[2] + c62 * mSi[4];
  k2 = c60 * mSi[3] + c61 * mSi[4] + c62 * mSi[5];

  qp += k0 * z0 + k1 * z1 + k2 * z2;
  if (qp > 0) {
    fSignQ = 1;
  } else if (qp < 0) {
    fSignQ = -1;
  } else {
    fSignQ = 0;
  }
}

//* Other utilities

void AliHLT3DTrackParam::SetDirection(double Direction[3])
{
  //* Change track direction

  if (mParam[3] * Direction[0] + mParam[4] * Direction[1] + mParam[5] * Direction[2] >= 0) {
    return;
  }

  mParam[3] = -mParam[3];
  mParam[4] = -mParam[4];
  mParam[5] = -mParam[5];
  fSignQ = -fSignQ;

  fCov[6] = -fCov[6];
  fCov[7] = -fCov[7];
  fCov[8] = -fCov[8];
  fCov[10] = -fCov[10];
  fCov[11] = -fCov[11];
  fCov[12] = -fCov[12];
  fCov[15] = -fCov[15];
  fCov[16] = -fCov[16];
  fCov[17] = -fCov[17];
}

void AliHLT3DTrackParam::RotateCoordinateSystem(double alpha)
{
  //* !

  double cA = TMath::Cos(alpha);
  double sA = TMath::Sin(alpha);
  double x = mParam[0], y = mParam[1], px = mParam[3], py = mParam[4];
  mParam[0] = x * cA + y * sA;
  mParam[1] = -x * sA + y * cA;
  mParam[2] = mParam[2];
  mParam[3] = px * cA + py * sA;
  mParam[4] = -px * sA + py * cA;
  mParam[5] = mParam[5];

  // clang-format off
  double mJ[6][6] = { { cA, sA, 0, 0, 0, 0 },
    { -sA, cA, 0,  0,  0,  0 },
    {  0, 0, 1,  0,  0,  0 },
    {  0, 0, 0, cA, sA,  0 },
    {  0, 0, 0, -sA, cA,  0 },
    {  0, 0, 0,  0,  0,  1 }};
  // clang-format on

  double mA[6][6];
  for (int k = 0, i = 0; i < 6; i++) {
    for (int j = 0; j <= i; j++, k++) {
      mA[i][j] = mA[j][i] = fCov[k];
    }
  }

  double mJC[6][6];
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      mJC[i][j] = 0;
      for (int k = 0; k < 6; k++) {
        mJC[i][j] += mJ[i][k] * mA[k][j];
      }
    }
  }

  for (int k = 0, i = 0; i < 6; i++) {
    for (int j = 0; j <= i; j++, k++) {
      fCov[k] = 0;
      for (int l = 0; l < 6; l++) {
        fCov[k] += mJC[i][l] * mJ[j][l];
      }
    }
  }
}

void AliHLT3DTrackParam::Get5Parameters(double alpha, double T[6], double C[15]) const
{
  //* !

  AliHLT3DTrackParam t = *this;
  t.RotateCoordinateSystem(alpha);
  double x = t.mParam[0], y = t.mParam[1], z = t.mParam[2], px = t.mParam[3], py = t.mParam[4], pz = t.mParam[5], q = t.fSignQ;

  double p2 = px * px + py * py + pz * pz;
  if (p2 < 1.e-8) {
    p2 = 1;
  }
  double n2 = 1. / p2;
  double n = sqrt(n2);

  T[5] = x;
  T[0] = y;
  T[1] = z;
  T[2] = py / px;
  T[3] = pz / px;
  T[4] = q * n;

  // clang-format off
  double mJ[5][6] = { { -T[2], 1, 0, 0, 0, 0 },
    { -T[3], 0, 1,  0,  0,  0 },
    { 0, 0, 0,  -T[2] / px,  1. / px,  0 },
    { 0, 0, 0, -T[3] / px,  0,  1. / px },
    { 0, 0, 0, -T[4]*n2*px, -T[4]*n2*py, -T[4]*n2*pz}};
  // clang-format on

  double mA[6][6];
  for (int k = 0, i = 0; i < 6; i++) {
    for (int j = 0; j <= i; j++, k++) {
      mA[i][j] = mA[j][i] = t.fCov[k];
    }
  }

  double mJC[5][6];
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 6; j++) {
      mJC[i][j] = 0;
      for (int k = 0; k < 6; k++) {
        mJC[i][j] += mJ[i][k] * mA[k][j];
      }
    }
  }

  for (int k = 0, i = 0; i < 5; i++) {
    for (int j = 0; j <= i; j++, k++) {
      C[k] = 0;
      for (int l = 0; l < 6; l++) {
        C[k] += mJC[i][l] * mJ[j][l];
      }
    }
  }
}
