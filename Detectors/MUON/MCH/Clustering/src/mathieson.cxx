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

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "mathUtil.h"
#include "mathieson.h"

namespace o2
{
namespace mch
{
// Chamber 1, 2
const double sqrtK3x1_2 = 0.7000; // Pitch= 0.21 cm
const double sqrtK3y1_2 = 0.7550; // Pitch= 0.21 cm
const double pitch1_2 = 0.21;
// Chamber 3, 10
const double sqrtK3x3_10 = 0.7131; // Pitch= 0.25 cm
const double sqrtK3y3_10 = 0.7642; // Pitch= 0.25 cm
const double pitch3_10 = 0.25;

int mathiesonType; // 0 for Station 1 or 1 for station 2-5
static double K1x[2], K1y[2];
static double K2x[2], K2y[2];
static const double sqrtK3x[2] = {sqrtK3x1_2, sqrtK3x3_10},
                    sqrtK3y[2] = {sqrtK3y1_2, sqrtK3y3_10};
static double K4x[2], K4y[2];
static double pitch[2] = {pitch1_2, pitch3_10};
static double invPitch[2];

void initMathieson()
{
  //
  for (int i = 0; i < 2; i++) {
    K2x[i] = M_PI * 0.5 * (1.0 - sqrtK3x[i] * 0.5);
    K2y[i] = M_PI * 0.5 * (1.0 - sqrtK3y[i] * 0.5);
    K1x[i] = K2x[i] * sqrtK3x[i] * 0.25 / (atan(sqrtK3x[i]));
    K1y[i] = K2y[i] * sqrtK3y[i] * 0.25 / (atan(sqrtK3y[i]));
    K4x[i] = K1x[i] / K2x[i] / sqrtK3x[i];
    K4y[i] = K1y[i] / K2y[i] / sqrtK3y[i];
    invPitch[i] = 1.0 / pitch[i];
  }
}

void compute2DPadIntegrals(const double* xInf, const double* xSup,
                           const double* yInf, const double* ySup, int N,
                           int chamberId, double Integrals[])
{
  // Returning array: Charge Integral on all the pads
  //
  if (chamberId <= 2) {
    mathiesonType = 0;
  } else {
    mathiesonType = 1;
  }
  //
  // Select Mathieson coef.
  double curK2x = K2x[mathiesonType];
  double curK2y = K2y[mathiesonType];
  double curSqrtK3x = sqrtK3x[mathiesonType];
  double curSqrtK3y = sqrtK3y[mathiesonType];
  double curK4x = K4x[mathiesonType];
  double curK4y = K4y[mathiesonType];
  double curInvPitch = invPitch[mathiesonType];
  double cst2x = curK2x * curInvPitch;
  double cst2y = curK2y * curInvPitch;
  double cst4 = 4.0 * curK4x * curK4y;
  double uInf, uSup, vInf, vSup;

  for (int i = 0; i < N; i++) {
    // x/u
    uInf = curSqrtK3x * tanh(cst2x * xInf[i]);
    uSup = curSqrtK3x * tanh(cst2x * xSup[i]);
    // y/v
    vInf = curSqrtK3y * tanh(cst2y * yInf[i]);
    vSup = curSqrtK3y * tanh(cst2y * ySup[i]);
    //
    Integrals[i] = cst4 * (atan(uSup) - atan(uInf)) * (atan(vSup) - atan(vInf));
    // printf(" xyInfSup %2d  [%10.6g, %10.6g] x [%10.6g, %10.6g]-> %10.6g\n",
    // i, xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i]);
  }
  // printf(" I[0..%3ld] = %f, %f, ... %f\n", N-1, Integrals[0], Integrals[1],
  // Integrals[N-1]);
  return;
}

void compute1DPadIntegrals(const double* xInf, const double* xSup, int N,
                           int chamberId, bool xAxe, double* Integrals)
{
  // Returning array: Charge Integral on all the pads
  //
  if (chamberId <= 2) {
    mathiesonType = 0;
  } else {
    mathiesonType = 1;
  }
  //
  // Select Mathieson coef.
  double curInvPitch = invPitch[mathiesonType];
  double curK2, curSqrtK3, curK4, cst2;
  if (xAxe) {
    curK2 = K2x[mathiesonType];
    curSqrtK3 = sqrtK3x[mathiesonType];
    curK4 = K4x[mathiesonType];
    cst2 = curK2 * curInvPitch;
  } else {
    curK2 = K2y[mathiesonType];
    curSqrtK3 = sqrtK3y[mathiesonType];
    curK4 = K4y[mathiesonType];
    cst2 = curK2 * curInvPitch;
  }
  double cst4 = 2.0 * curK4;

  double uInf, uSup, vInf, vSup;

  for (int i = 0; i < N; i++) {
    // x/u
    uInf = curSqrtK3 * tanh(cst2 * xInf[i]);
    uSup = curSqrtK3 * tanh(cst2 * xSup[i]);
    //
    Integrals[i] = cst4 * (atan(uSup) - atan(uInf));
    // printf(" xyInfSup %2d  [%10.6g, %10.6g] x [%10.6g, %10.6g]-> %10.6g\n",
    // i, xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i]);
  }
  // printf(" I[0..%3ld] = %f, %f, ... %f\n", N-1, Integrals[0], Integrals[1],
  // Integrals[N-1]);
  return;
}

void compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0,
                                           const double* theta, int N, int K,
                                           int chamberId, double Integrals[])
{
  // Returning array: Charge Integral on all the pads
  // Remarks:
  // - This fct is a cumulative one, as a result it should be set to zero
  //    before calling it
  vectorSetZero(Integrals, N);
  const double* xInf0 = getConstXInf(xyInfSup0, N);
  const double* yInf0 = getConstYInf(xyInfSup0, N);
  const double* xSup0 = getConstXSup(xyInfSup0, N);
  const double* ySup0 = getConstYSup(xyInfSup0, N);
  //
  const double* muX = getConstMuX(theta, K);
  const double* muY = getConstMuY(theta, K);
  const double* w = getConstW(theta, K);

  double z[N];
  double xyInfSup[4 * N];
  double* xInf = getXInf(xyInfSup, N);
  double* yInf = getYInf(xyInfSup, N);
  double* xSup = getXSup(xyInfSup, N);
  double* ySup = getYSup(xyInfSup, N);
  for (int k = 0; k < K; k++) {
    vectorAddScalar(xInf0, -muX[k], N, xInf);
    vectorAddScalar(xSup0, -muX[k], N, xSup);
    vectorAddScalar(yInf0, -muY[k], N, yInf);
    vectorAddScalar(ySup0, -muY[k], N, ySup);
    compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId, z);
    // printf("Vector Sum %g\n", vectorSum(z, N) );
    vectorAddVector(Integrals, w[k], z, N, Integrals);
  }
}

void computeFastCij(const Pads& pads, const Pads& pixel, double Cij[])
{
  // Compute the Charge Integral Cij of pads (j index), considering the
  // center of the Mathieson fct on a pixel (i index)
  // Use the fact that the charge integral CI(x,y) = CI(x) * CI(y)
  // to reduce the computation cost
  // CI(x) is store in PadIntegralX
  // CI(y) is store in PadIntegralY
  // A subsampling of CI(x_i + k*minDx) (or CI(y_i + l*minDY)) is used
  // by taking the mininimun of pads.dx(pads.dy) to discretize the x/y space
  //
  // CI(x)/CI(y) are computed if they are requested.
  //
  // Returning array: Charge Integral on all the pads Cij[]

  if ((pads.mode != Pads::xyInfSupMode) || (pixel.mode != Pads::xydxdyMode)) {
    printf(
      "[computeFastCij] exception: bad representation (mode) of pads in "
      "computeCij (padMode=%d, pixelMode=%d)\n",
      pads.mode, pixel.mode);
    throw std::overflow_error("Bad mode");
    return;
  }
  int N = pads.getNbrOfPads();
  int K = pixel.getNbrOfPads();
  // Pads
  int chId = pads.getChamberId();
  const double* xInf0 = pads.getXInf();
  const double* yInf0 = pads.getYInf();
  const double* xSup0 = pads.getXSup();
  const double* ySup0 = pads.getYSup();
  // Pixels
  const double* muX = pixel.getX();
  const double* muY = pixel.getY();
  //
  double xPixMin = vectorMin(muX, K);
  double xPixMax = vectorMax(muX, K);
  double yPixMin = vectorMin(muY, K);
  double yPixMax = vectorMax(muY, K);
  double dxMinPix = 2 * vectorMin(pixel.getDX(), K);
  double dyMinPix = 2 * vectorMin(pixel.getDY(), K);
  // Sampling of PadIntegralX/PadIntegralY
  int nXPixels = (int)((xPixMax - xPixMin) / dxMinPix + 0.5) + 1;
  int nYPixels = (int)((yPixMax - yPixMin) / dyMinPix + 0.5) + 1;
  //
  // PadIntegralX/PadIntegralY allocation and init with -1
  // ??? double PadIntegralX[nXPixels][N];
  //     double PadIntegralY[nYPixels][N];
  double* PadIntegralX = new double[nXPixels * N];
  double* PadIntegralY = new double[nYPixels * N];
  // Inv. printf("??? nXPixels=%d, xPixMin=%f, xPixMax=%f, dxMinPix=%f, nPads=%d\n", nXPixels, xPixMin, xPixMax, dxMinPix, N);
  // printf("??? nYPixels=%d, yPixMin=%f, yPixMax=%f, dyMinPix=%f, nPads=%d\n", nYPixels, yPixMin, yPixMax, dyMinPix, N);
  vectorSet((double*)PadIntegralX, -1.0, nXPixels * N);
  vectorSet((double*)PadIntegralY, -1.0, nYPixels * N);
  double zInf[N];
  double zSup[N];
  bool xAxe;
  /*
  for (int kx=0; kx < nXPixels; kx++) {
    double x = xPixMin + kx * dxPix;
    vectorAddScalar( xInf0, - x, N, zInf );
    vectorAddScalar( xSup0, - x, N, zSup );
    compute1DPadIntegrals( zInf, zSup, N, chId, xAxe, PadIntegralX[kx] );
  }
  xAxe = false;
  for (int ky=0; ky < nYPixels; ky++) {
    double y = yPixMin + ky * dyPix;
    vectorAddScalar( yInf0, - y, N, zInf );
    vectorAddScalar( ySup0, - y, N, zSup );
    compute1DPadIntegrals( zInf, zSup, N, chId, xAxe, PadIntegralY[ky] );
  }
  */

  // Loop on Pixels
  for (int k = 0; k < K; k++) {
    // Calculate the indexes in the 1D charge integral
    // PadIntegralX:PadIntegralY
    int xIdx = (int)((muX[k] - xPixMin) / dxMinPix + 0.5);
    int yIdx = (int)((muY[k] - yPixMin) / dyMinPix + 0.5);
    // compute2DPadIntegrals( xInf, xSup, yInf, ySup, N, chId, &Cij[N*k] );
    // Cij[ N*k + p] = PadIntegralX( k, xIdx) * PadIntegralY( k, yIdx);
    // printf("k=%d, mu[k]=(%f, %f) Sum_pads Ck = %g\n", k, muX[k], muY[k],
    // vectorSum( &Cij[N*k], N) );
    if (PadIntegralX[xIdx * N + 0] == -1) {
      // Not yet computed
      vectorAddScalar(xInf0, -muX[k], N, zInf);
      vectorAddScalar(xSup0, -muX[k], N, zSup);
      xAxe = true;
      compute1DPadIntegrals(zInf, zSup, N, chId, xAxe, &PadIntegralX[xIdx * N + 0]);
    }
    if (PadIntegralY[yIdx * N + 0] == -1) {
      // Not yet computed
      vectorAddScalar(yInf0, -muY[k], N, zInf);
      vectorAddScalar(ySup0, -muY[k], N, zSup);
      xAxe = false;
      compute1DPadIntegrals(zInf, zSup, N, chId, xAxe, &PadIntegralY[yIdx * N + 0]);
    }
    // Compute IC(xy) = IC(x) * IC(y)
    vectorMultVector(&PadIntegralX[xIdx * N + 0], &PadIntegralY[yIdx * N + 0], N, &Cij[N * k]);
  }
  delete[] PadIntegralX;
  delete[] PadIntegralY;
}

void computeCij(const Pads& pads, const Pads& pixel, double Cij[])
{
  // Compute the Charge Integral Cij of pads (j index), considering the
  // center of the Mathieson fct on a pixel (i index)
  //
  // Returning array: Charge Integral on all the pads Cij[]

  if ((pads.mode != Pads::xyInfSupMode) || (pixel.mode != Pads::xydxdyMode)) {
    printf(
      "computeFastCij] exception: bad representation (mode) of pads in "
      "computeCij (padMode=%d, pixelMode=%d)\n",
      pads.mode, pixel.mode);
    throw std::overflow_error("Bad mode");
    return;
  }
  int N = pads.getNbrOfPads();
  int K = pixel.getNbrOfPads();
  int chId = pads.getChamberId();
  const double* xInf0 = pads.getXInf();
  const double* yInf0 = pads.getYInf();
  const double* xSup0 = pads.getXSup();
  const double* ySup0 = pads.getXSup();

  //
  const double* muX = pixel.getX();
  const double* muY = pixel.getY();

  double xInf[N];
  double yInf[N];
  double xSup[N];
  double ySup[N];

  for (int k = 0; k < K; k++) {
    vectorAddScalar(xInf0, -muX[k], N, xInf);
    vectorAddScalar(xSup0, -muX[k], N, xSup);
    vectorAddScalar(yInf0, -muY[k], N, yInf);
    vectorAddScalar(ySup0, -muY[k], N, ySup);
    compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chId, &Cij[N * k]);
    // printf("k=%d, mu[k]=(%f, %f) Sum_pads Ck = %g\n", k, muX[k], muY[k],
    // vectorSum( &Cij[N*k], N) );
  }
}
// theta
double* getVarX(double* theta, int K) { return &theta[0 * K]; };
double* getVarY(double* theta, int K) { return &theta[1 * K]; };
double* getMuX(double* theta, int K) { return &theta[2 * K]; };
double* getMuY(double* theta, int K) { return &theta[3 * K]; };
double* getW(double* theta, int K) { return &theta[4 * K]; };
double* getMuAndW(double* theta, int K) { return &theta[2 * K]; };
//
const double* getConstVarX(const double* theta, int K)
{
  return &theta[0 * K];
};
const double* getConstVarY(const double* theta, int K)
{
  return &theta[1 * K];
};
const double* getConstMuX(const double* theta, int K) { return &theta[2 * K]; };
const double* getConstMuY(const double* theta, int K) { return &theta[3 * K]; };
const double* getConstW(const double* theta, int K) { return &theta[4 * K]; };
const double* getConstMuAndW(const double* theta, int K)
{
  return &theta[2 * K];
};

// xyDxy
double* getX(double* xyDxy, int N) { return &xyDxy[0 * N]; };
double* getY(double* xyDxy, int N) { return &xyDxy[1 * N]; };
double* getDX(double* xyDxy, int N) { return &xyDxy[2 * N]; };
double* getDY(double* xyDxy, int N) { return &xyDxy[3 * N]; };
//
const double* getConstX(const double* xyDxy, int N) { return &xyDxy[0 * N]; };
const double* getConstY(const double* xyDxy, int N) { return &xyDxy[1 * N]; };
const double* getConstDX(const double* xyDxy, int N) { return &xyDxy[2 * N]; };
const double* getConstDY(const double* xyDxy, int N) { return &xyDxy[3 * N]; };

// xySupInf
double* getXInf(double* xyInfSup, int N) { return &xyInfSup[0 * N]; };
double* getYInf(double* xyInfSup, int N) { return &xyInfSup[1 * N]; };
double* getXSup(double* xyInfSup, int N) { return &xyInfSup[2 * N]; };
double* getYSup(double* xyInfSup, int N) { return &xyInfSup[3 * N]; };
const double* getConstXInf(const double* xyInfSup, int N)
{
  return &xyInfSup[0 * N];
};
const double* getConstYInf(const double* xyInfSup, int N)
{
  return &xyInfSup[1 * N];
};
const double* getConstXSup(const double* xyInfSup, int N)
{
  return &xyInfSup[2 * N];
};
const double* getConstYSup(const double* xyInfSup, int N)
{
  return &xyInfSup[3 * N];
};

void copyTheta(const double* theta0, int K0, double* theta, int K1, int K)
{
  const double* w = getConstW(theta0, K0);
  const double* muX = getConstMuX(theta0, K0);
  const double* muY = getConstMuY(theta0, K0);
  const double* varX = getConstVarX(theta0, K0);
  const double* varY = getConstVarY(theta0, K0);
  double* wm = getW(theta, K1);
  double* muXm = getMuX(theta, K1);
  double* muYm = getMuY(theta, K1);
  double* varXm = getVarX(theta, K1);
  double* varYm = getVarY(theta, K1);
  vectorCopy(w, K, wm);
  vectorCopy(muX, K, muXm);
  vectorCopy(muY, K, muYm);
  vectorCopy(varX, K, varXm);
  vectorCopy(varY, K, varYm);
}

void copyXYdXY(const double* xyDxy0, int N0, double* xyDxy, int N1, int N)
{
  const double* X0 = getConstX(xyDxy0, N0);
  const double* Y0 = getConstY(xyDxy0, N0);
  const double* DX0 = getConstDX(xyDxy0, N0);
  const double* DY0 = getConstDY(xyDxy0, N0);

  double* X = getX(xyDxy, N1);
  double* Y = getY(xyDxy, N1);
  double* DX = getDX(xyDxy, N1);
  double* DY = getDY(xyDxy, N1);

  vectorCopy(X0, N, X);
  vectorCopy(Y0, N, Y);
  vectorCopy(DX0, N, DX);
  vectorCopy(DY0, N, DY);
}

void printTheta(const char* str, double meanCharge, const double* theta, int K)
{
  const double* varX = getConstVarX(theta, K);
  const double* varY = getConstVarY(theta, K);
  const double* muX = getConstMuX(theta, K);
  const double* muY = getConstMuY(theta, K);
  const double* w = getConstW(theta, K);

  printf("%s \n", str);
  printf("    k      charge      w      muX      muY     sigX   sigY\n");
  for (int k = 0; k < K; k++) {
    printf("  %.2d-th: %8.2g %6.3g %8.3g %8.3g %8.3g %8.3g\n", k, w[k] * meanCharge, w[k], muX[k],
           muY[k], sqrt(varX[k]), sqrt(varY[k]));
  }
}
void xyDxyToxyInfSup(const double* xyDxy, int nxyDxy, double* xyInfSup)
{
  const double* X = getConstX(xyDxy, nxyDxy);
  const double* Y = getConstY(xyDxy, nxyDxy);
  const double* DX = getConstDX(xyDxy, nxyDxy);
  const double* DY = getConstDY(xyDxy, nxyDxy);
  double* XInf = getXInf(xyInfSup, nxyDxy);
  double* XSup = getXSup(xyInfSup, nxyDxy);
  double* YInf = getYInf(xyInfSup, nxyDxy);
  double* YSup = getYSup(xyInfSup, nxyDxy);
  for (int k = 0; k < nxyDxy; k++) {
    // To avoid overwritting
    double xInf = X[k] - DX[k];
    double xSup = X[k] + DX[k];
    double yInf = Y[k] - DY[k];
    double ySup = Y[k] + DY[k];
    //
    XInf[k] = xInf;
    YInf[k] = yInf;
    XSup[k] = xSup;
    YSup[k] = ySup;
  }
}

// Mask operations
void maskedCopyXYdXY(const double* xyDxy, int nxyDxy, const Mask_t* mask,
                     int nMask, double* xyDxyMasked, int nxyDxyMasked)
{
  const double* X = getConstX(xyDxy, nxyDxy);
  const double* Y = getConstY(xyDxy, nxyDxy);
  const double* DX = getConstDX(xyDxy, nxyDxy);
  const double* DY = getConstDY(xyDxy, nxyDxy);
  double* Xm = getX(xyDxyMasked, nxyDxyMasked);
  double* Ym = getY(xyDxyMasked, nxyDxyMasked);
  double* DXm = getDX(xyDxyMasked, nxyDxyMasked);
  double* DYm = getDY(xyDxyMasked, nxyDxyMasked);
  vectorGather(X, mask, nMask, Xm);
  vectorGather(Y, mask, nMask, Ym);
  vectorGather(DX, mask, nMask, DXm);
  vectorGather(DY, mask, nMask, DYm);
}

void maskedCopyToXYInfSup(const double* xyDxy, int ndxyDxy, const Mask_t* mask,
                          int nMask, double* xyDxyMasked, int ndxyDxyMasked)
{
  const double* X = getConstX(xyDxy, ndxyDxy);
  const double* Y = getConstY(xyDxy, ndxyDxy);
  const double* DX = getConstDX(xyDxy, ndxyDxy);
  const double* DY = getConstDY(xyDxy, ndxyDxy);
  double* Xm = getX(xyDxyMasked, ndxyDxyMasked);
  double* Ym = getY(xyDxyMasked, ndxyDxyMasked);
  double* DXm = getDX(xyDxyMasked, ndxyDxyMasked);
  double* DYm = getDY(xyDxyMasked, ndxyDxyMasked);
  double* XmInf = getXInf(xyDxyMasked, ndxyDxyMasked);
  double* XmSup = getXSup(xyDxyMasked, ndxyDxyMasked);
  double* YmInf = getYInf(xyDxyMasked, ndxyDxyMasked);
  double* YmSup = getYSup(xyDxyMasked, ndxyDxyMasked);
  vectorGather(X, mask, nMask, Xm);
  vectorGather(Y, mask, nMask, Ym);
  vectorGather(DX, mask, nMask, DXm);
  vectorGather(DY, mask, nMask, DYm);
  for (int k = 0; k < nMask; k++) {
    // To avoid overwritting
    double xInf = Xm[k] - DXm[k];
    double xSup = Xm[k] + DXm[k];
    double yInf = Ym[k] - DYm[k];
    double ySup = Ym[k] + DYm[k];
    //
    XmInf[k] = xInf;
    YmInf[k] = yInf;
    XmSup[k] = xSup;
    YmSup[k] = ySup;
  }
}

void maskedCopyTheta(const double* theta, int K, const Mask_t* mask, int nMask,
                     double* maskedTheta, int maskedK)
{
  const double* w = getConstW(theta, K);
  const double* muX = getConstMuX(theta, K);
  const double* muY = getConstMuY(theta, K);
  const double* varX = getConstVarX(theta, K);
  const double* varY = getConstVarY(theta, K);
  double* wm = getW(maskedTheta, maskedK);
  double* muXm = getMuX(maskedTheta, maskedK);
  double* muYm = getMuY(maskedTheta, maskedK);
  double* varXm = getVarX(maskedTheta, maskedK);
  double* varYm = getVarY(maskedTheta, maskedK);
  vectorGather(w, mask, nMask, wm);
  vectorGather(muX, mask, nMask, muXm);
  vectorGather(muY, mask, nMask, muYm);
  vectorGather(varX, mask, nMask, varXm);
  vectorGather(varY, mask, nMask, varYm);
}

void printXYdXY(const char* str, const double* xyDxy, int NMax, int N,
                const double* val1, const double* val2)
{
  const double* X = getConstX(xyDxy, NMax);
  const double* Y = getConstY(xyDxy, NMax);
  const double* DX = getConstDX(xyDxy, NMax);
  const double* DY = getConstDY(xyDxy, NMax);

  printf("%s\n", str);
  int nPrint = 0;
  if (val1 != nullptr) {
    nPrint++;
  }
  if (val2 != nullptr) {
    nPrint++;
  }
  if ((nPrint == 1) && (val2 != nullptr)) {
    val1 = val2;
  }

  if (nPrint == 0) {
    for (PadIdx_t i = 0; i < N; i++) {
      printf("  pad %2d: %9.3g %9.3g %9.3g %9.3g \n", i, X[i], Y[i], DX[i],
             DY[i]);
    }
  } else if (nPrint == 1) {
    for (PadIdx_t i = 0; i < N; i++) {
      printf("  pad %2d: %9.3g %9.3g %9.3g %9.3g - %9.3g \n", i, X[i], Y[i],
             DX[i], DY[i], val1[i]);
    }
  } else {
    for (PadIdx_t i = 0; i < N; i++) {
      printf("  pad %d: %9.3g %9.3g %9.3g %9.3g - %9.3g %9.3g \n", i, X[i],
             Y[i], DX[i], DY[i], val1[i], val2[i]);
    }
  }
}

} // namespace mch
} // namespace o2

// C Wrapper
void o2_mch_initMathieson() { o2::mch::initMathieson(); }

void o2_mch_compute2DPadIntegrals(const double* xInf, const double* xSup,
                                  const double* yInf, const double* ySup, int N,
                                  int chamberId, double Integrals[])
{
  o2::mch::compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId,
                                 Integrals);
}

void o2_mch_computeCij(const double* xyInfSup0, const double* pixel, int N,
                       int K, int chamberId, double Cij[])
{
  // Returning array: Charge Integral on all the pads
  // Remarks:
  // - This fct is a cumulative one, as a result it should be set to zero
  //    before calling it
  const double* xInf0 = o2::mch::getConstXInf(xyInfSup0, N);
  const double* yInf0 = o2::mch::getConstYInf(xyInfSup0, N);
  const double* xSup0 = o2::mch::getConstXSup(xyInfSup0, N);
  const double* ySup0 = o2::mch::getConstYSup(xyInfSup0, N);
  //
  const double* muX = o2::mch::getConstMuX(pixel, K);
  const double* muY = o2::mch::getConstMuY(pixel, K);
  const double* w = o2::mch::getConstW(pixel, K);

  double z[N];
  double xyInfSup[4 * N];
  double* xInf = o2::mch::getXInf(xyInfSup, N);
  double* yInf = o2::mch::getYInf(xyInfSup, N);
  double* xSup = o2::mch::getXSup(xyInfSup, N);
  double* ySup = o2::mch::getYSup(xyInfSup, N);
  for (int k = 0; k < K; k++) {
    o2::mch::vectorAddScalar(xInf0, -muX[k], N, xInf);
    o2::mch::vectorAddScalar(xSup0, -muX[k], N, xSup);
    o2::mch::vectorAddScalar(yInf0, -muY[k], N, yInf);
    o2::mch::vectorAddScalar(ySup0, -muY[k], N, ySup);
    o2_mch_compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId,
                                 &Cij[N * k]);
    // printf("Vector Sum %g\n", vectorSum(z, N) );
  }
}

void o2_mch_compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0,
                                                  const double* theta, int N,
                                                  int K, int chamberId,
                                                  double Integrals[])
{
  o2::mch::compute2DMathiesonMixturePadIntegrals(xyInfSup0, theta, N, K,
                                                 chamberId, Integrals);
}
