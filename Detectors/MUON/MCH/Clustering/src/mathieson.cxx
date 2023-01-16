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
#include <map>
#include <limits>

#include "MCHClustering/ClusterConfig.h"
#include "mathUtil.h"
#include "mathieson.h"

namespace o2
{
namespace mch
{
extern ClusterConfig clusterConfig;

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
static double K3x[2], K3y[2];
static double K4x[2], K4y[2];
static double pitch[2] = {pitch1_2, pitch3_10};
static double invPitch[2];

// Spline Coef
int useSpline = 0;
SplineCoef* splineCoef[2][2];
static double splineXYStep = 1.0e-3;
static double splineXYLimit = 3.0;
static int nSplineSampling = 0;
double* splineXY = nullptr;

//
int useCache = 0;

SplineCoef::SplineCoef(int N)
{
  a = new double[N];
  b = new double[N];
  c = new double[N];
  d = new double[N];
}

SplineCoef::~SplineCoef()
{
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
}
void initMathieson(int useSpline_, int useCache_)
{
  useSpline = useSpline_;
  useCache = useCache_;
  //
  for (int i = 0; i < 2; i++) {
    K3x[i] = sqrtK3x[i] * sqrtK3x[i];
    K3y[i] = sqrtK3y[i] * sqrtK3y[i];
    K2x[i] = M_PI * 0.5 * (1.0 - sqrtK3x[i] * 0.5);
    K2y[i] = M_PI * 0.5 * (1.0 - sqrtK3y[i] * 0.5);
    K1x[i] = K2x[i] * sqrtK3x[i] * 0.25 / (atan(sqrtK3x[i]));
    K1y[i] = K2y[i] * sqrtK3y[i] * 0.25 / (atan(sqrtK3y[i]));
    K4x[i] = K1x[i] / K2x[i] / sqrtK3x[i];
    K4y[i] = K1y[i] / K2y[i] / sqrtK3y[i];
    invPitch[i] = 1.0 / pitch[i];
  }
  if (useSpline) {
    initSplineMathiesonPrimitive();
  }
}

void initSplineMathiesonPrimitive()
{
  // x/y Interval and positive x/y limit
  double xyStep = splineXYStep;
  double xyLimit = splineXYLimit;
  // X/Y Sampling
  nSplineSampling = int(xyLimit / xyStep) + 1;
  int N = nSplineSampling;

  splineXY = new double[N];
  for (int i = 0; i < N; i++) {
    splineXY[i] = xyStep * i;
  }
  double* xy = splineXY;

  // Spline coef allocation for the 4 functions
  splineCoef[0][0] = new SplineCoef(N);
  splineCoef[0][1] = new SplineCoef(N);
  splineCoef[1][0] = new SplineCoef(N);
  splineCoef[1][1] = new SplineCoef(N);

  // Compute the spline Coef. for the 4 Mathieson primitives
  double mathPrimitive[N];
  double rightDerivative(0.0), leftDerivative;
  // X and Y primitives on chambers <= 2 (Mathieson Type = 0)
  int mathiesonType = 0;
  int axe = 0;
  mathiesonPrimitive(xy, N, axe, 2, mathPrimitive);
  leftDerivative = 2.0 * K4x[mathiesonType] * sqrtK3x[mathiesonType] * K2x[mathiesonType] * invPitch[mathiesonType];
  computeSplineCoef(xy, xyStep, mathPrimitive, N, leftDerivative, rightDerivative, o2::mch::splineCoef[mathiesonType][axe]);
  axe = 1;
  mathiesonPrimitive(xy, N, axe, 2, mathPrimitive);
  leftDerivative = 2.0 * K4y[mathiesonType] * sqrtK3y[mathiesonType] * K2y[mathiesonType] * invPitch[mathiesonType];
  computeSplineCoef(xy, xyStep, mathPrimitive, N, leftDerivative, rightDerivative, splineCoef[mathiesonType][axe]);
  mathiesonType = 1;
  axe = 0;
  mathiesonPrimitive(xy, N, axe, 3, mathPrimitive);
  leftDerivative = 2.0 * K4x[mathiesonType] * sqrtK3x[mathiesonType] * K2x[mathiesonType] * invPitch[mathiesonType];
  computeSplineCoef(xy, xyStep, mathPrimitive, N, leftDerivative, rightDerivative, splineCoef[mathiesonType][axe]);
  axe = 1;
  mathiesonPrimitive(xy, N, axe, 3, mathPrimitive);
  leftDerivative = 2.0 * K4y[mathiesonType] * sqrtK3y[mathiesonType] * K2y[mathiesonType] * invPitch[mathiesonType];
  computeSplineCoef(xy, xyStep, mathPrimitive, N, leftDerivative, rightDerivative, splineCoef[mathiesonType][axe]);
}

// Spline implementation of the book "Numerical Analysis" - 9th edition
// Richard L Burden, J Douglas Faires
// Section 3.5, p. 146
// Restrictions : planed with a regular sampling (dx = cst)
// spline(x) :[-inf, +inf] -> [-1/2, +1/2]
// Error < 7.0 e-11 for 1001 sampling between [0, 3.0]
void computeSplineCoef(const double* xy, double xyStep, const double* f, int N,
                       double leftDerivative, double rightDerivative, SplineCoef* splineCoef)
{
  double* a = splineCoef->a;
  double* b = splineCoef->b;
  double* c = splineCoef->c;
  double* d = splineCoef->d;

  // a coef : the sampled function
  vectorCopy(f, N, a);

  // Step 1
  double h = xyStep;

  // Step 2 & 3 : Compute alpha
  double alpha[N];
  alpha[0] = 3.0 / h * (f[1] - f[0]) - 3 * leftDerivative;
  alpha[N - 1] = 3 * rightDerivative - 3.0 / h * (f[N - 1] - f[N - 2]);
  for (int i = 1; i < N - 1; i++) {
    // To keep the general case if h is not constant
    alpha[i] = 3.0 / h * (f[i + 1] - f[i]) - 3.0 / h * (f[i] - f[i - 1]);
  }

  // Step 4 to 6 solve a tridiagonal linear system
  //
  // Step 4
  double l[N], mu[N], z[N];
  l[0] = 2.0 * h;
  mu[0] = 0.5;
  z[0] = alpha[0] / l[0];
  //
  // Step 5
  for (int i = 1; i < N - 1; i++) {
    l[i] = 2.0 * (xy[i + 1] - xy[i - 1]) - h * mu[i - 1];
    mu[i] = h / l[i];
    z[i] = (alpha[i] - h * z[i - 1]) / l[i];
  }
  //
  // Step 6
  l[N - 1] = h * (2.0 - mu[N - 2]);
  z[N - 1] = (alpha[N - 1] - h * z[N - 2]) / l[N - 1];
  c[N - 1] = z[N - 1];

  // Step 7 : calculate cubic coefficients
  for (int j = N - 2; j >= 0; j--) {
    c[j] = z[j] - mu[j] * c[j + 1];
    b[j] = (f[j + 1] - f[j]) / h - h / 3.0 * (c[j + 1] + 2 * c[j]);
    d[j] = (c[j + 1] - c[j]) / (3 * h);
  }
}

void splineMathiesonPrimitive(const double* x, int N, int axe, int chamberId, double* mPrimitive)
{
  int mathiesonType = (chamberId <= 2) ? 0 : 1;
  double* a = splineCoef[mathiesonType][axe]->a;
  double* b = splineCoef[mathiesonType][axe]->b;
  double* c = splineCoef[mathiesonType][axe]->c;
  double* d = splineCoef[mathiesonType][axe]->d;
  double dx = splineXYStep;
  // printf("dx=%f nSplineSampling=%d\n", dx, nSplineSampling);
  double signX[N];
  // x without sign
  double uX[N];
  for (int i = 0; i < N; i++) {
    signX[i] = (x[i] >= 0) ? 1 : -1;
    uX[i] = signX[i] * x[i];
    /*
    if( uX[i] > (2.0 * splineXYLimit)) {
      // x >> 0, f(x) = 0.0
      signX[i] = 0.0;
      uX[i] = 0.5;
    }
    */
  }

  double cst = 1.0 / dx;
  // Get indexes in the sample function
  int idx;
  double h;
  for (int i = 0; i < N; i++) {
    // int k = int( uX[i] * cst + dx*0.1 );
    //  if ( k < nSplineSampling) {
    if (uX[i] < splineXYLimit) {
      idx = int(uX[i] * cst + dx * 0.1);
      h = uX[i] - idx * dx;
    } else {
      idx = nSplineSampling - 1;
      h = 0;
    }
    mPrimitive[i] = signX[i] * (a[idx] + h * (b[idx] + h * (c[idx] + h * (d[idx]))));
    // printf("x[i]=%f, signX[i]=%f uX[i]=%f idx=%d, h=%f, prim=%f, splineXYLimit=%f\n", x[i], signX[i], uX[i], idx, h, mPrimitive[i], splineXYLimit );
  }
  // print ("uX ",  uX)
  //     print ("h ",  h)
  //     print ("f(x0) ",  a[idx])
  //     print ("df|dx0",  h*( b[idx] + h*( c[idx] + h *(d[idx]))))
  //     print ("f, ",  a[idx] + h*( b[idx] + h*( c[idx] + h *(d[idx]))))
}

// Return the Mathieson primitive at x or y
void mathiesonPrimitive(const double* xy, int N,
                        int axe, int chamberId, double mPrimitive[])
{
  mathiesonType = (chamberId <= 2) ? 0 : 1;
  //
  // Select Mathieson coef.
  double curK2xy = (axe == 0) ? K2x[mathiesonType] : K2y[mathiesonType];
  double curSqrtK3xy = (axe == 0) ? sqrtK3x[mathiesonType] : sqrtK3y[mathiesonType];
  double curInvPitch = invPitch[mathiesonType];
  double cst2xy = curK2xy * curInvPitch;
  double curK4xy = (axe == 0) ? K4x[mathiesonType] : K4y[mathiesonType];

  for (int i = 0; i < N; i++) {
    double u = curSqrtK3xy * tanh(cst2xy * xy[i]);
    mPrimitive[i] = 2 * curK4xy * atan(u);
  }
}

void compute1DMathieson(const double* xy, int N,
                        int axe, int chamberId, double mathieson[])
{
  // Returning array: Charge Integral on all the pads
  //
  mathiesonType = (chamberId <= 2) ? 0 : 1;

  //
  // Select Mathieson coef.

  double curK1xy = (axe == 0) ? K1x[mathiesonType] : K1y[mathiesonType];
  double curK2xy = (axe == 0) ? K2x[mathiesonType] : K2y[mathiesonType];
  double curK3xy = (axe == 0) ? K3x[mathiesonType] : K3y[mathiesonType];
  double curInvPitch = invPitch[mathiesonType];
  double cst2xy = curK2xy * curInvPitch;

  for (int i = 0; i < N; i++) {
    //  tanh(x) & tanh(y)
    double xTanh = tanh(cst2xy * xy[i]);
    double xTanh2 = xTanh * xTanh;
    mathieson[i] = curK1xy * (1.0 - xTanh2) / (1.0 + curK3xy * xTanh2);
  }
  return;
}
void compute1DPadIntegrals(const double* xyInf, const double* xySup, int N,
                           double xy0, int axe, int chamberId, double* integrals)
{
  double zInf[N], zSup[N];
  vectorAddScalar(xyInf, -xy0, N, zInf);
  vectorAddScalar(xySup, -xy0, N, zSup);
  compute1DPadIntegrals(zInf, zSup, N, axe, chamberId, integrals);
}

void compute1DPadIntegrals(const double* xyInf, const double* xySup, int N,
                           int axe, int chamberId, double* Integrals)
{
  // Returning array: Charge Integral on all the pads
  //
  mathiesonType = (chamberId <= 2) ? 0 : 1;

  //
  // Select Mathieson coef.
  double curInvPitch = invPitch[mathiesonType];
  double curK2 = (axe == 0) ? K2x[mathiesonType] : K2y[mathiesonType];
  double curSqrtK3 = (axe == 0) ? sqrtK3x[mathiesonType] : sqrtK3y[mathiesonType];
  double curK4 = (axe == 0) ? K4x[mathiesonType] : K4y[mathiesonType];
  double cst2 = curK2 * curInvPitch;
  double cst4 = 2.0 * curK4;

  double uInf, uSup;
  for (int i = 0; i < N; i++) {
    // x/u
    uInf = curSqrtK3 * tanh(cst2 * xyInf[i]);
    uSup = curSqrtK3 * tanh(cst2 * xySup[i]);
    //
    Integrals[i] = cst4 * (atan(uSup) - atan(uInf));
    // printf(" xyInfSup %2d  [%10.6g, %10.6g] x [%10.6g, %10.6g]-> %10.6g\n",
    // i, xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i]);
  }
  // printf(" I[0..%3ld] = %f, %f, ... %f\n", N-1, Integrals[0], Integrals[1],
  // Integrals[N-1]);
  return;
}

int compressSameValues(const double* x1, const double* x2, int* map1, int* map2, int N, double* xCompress)
{
  // map1[0..N-1]: i in [0..N-1] -> integral index for x1 [0..nCompressed-1]
  // map2[0..N-1]: the same for x2
  // xCompress[0..nCompressed]: values of x1 & x2 compressed (unique values)
  // The xCompress values will be used to compute the primitive
  // The map1/2 will be used to find the corresponding index in the xCompress or primitive arrays
  // Return nCompressed

  // Transform to integer to avoid comparison on close x values
  const double* x[2] = {x1, x2};
  int* xCode = new int[2 * N];
  for (int i = 0; i < N; i++) {
    for (int b = 0; b < 2; b++) {
      // Calculate the indexes in the 1D charge integral
      // Error on pad position > 10-3 cm
      xCode[i + b * N] = (int)(x[b][i] * 1000 + 0.5);
    }
  }
  // Sort the code
  int sIdx[2 * N];
  for (int k = 0; k < 2 * N; k++) {
    sIdx[k] = k;
  }
  std::sort(sIdx, &sIdx[2 * N], [=](int a, int b) {
    return (xCode[a] < xCode[b]);
  });

  // printf("sort  xCode[sIdx[0]]=%d xCode[sIdx[2*N-1]]=%d\n", xCode[sIdx[0]], xCode[sIdx[2*N-1]]);
  // vectorPrintInt("xCode",xCode, 2*N);
  // vectorPrintInt("sIdx",sIdx, 2*N);

  // Renumber and compress
  int nCompress = 0;
  int prevCode = std::numeric_limits<int>::max();

  // Map1
  for (int i = 0; i < 2 * N; i++) {
    int idx = sIdx[i];
    if (xCode[idx] != prevCode) {
      if (idx < N) {
        // Store the compress value in map1
        xCompress[nCompress] = x1[idx];
        map1[idx] = nCompress;
        // printf("i=%d sIdx[i]=%d nCompress=%d idx=%d map1[idx]=%d\n", i, idx, nCompress, idx, map1[idx]);
      } else {
        // Store the compress value in map2
        xCompress[nCompress] = x2[idx - N];
        map2[idx - N] = nCompress;
        // printf("i=%d sIdx[i]=%d nCompress=%d idx-N=%d map2[idx]=%d\n", i, idx, nCompress,  idx-N, map2[idx-N]);
      }
      nCompress++;
    } else {
      // the code is the same (same values)
      if (idx < N) {
        map1[idx] = nCompress - 1;
        // printf("identical i=%d sIdx[i]=%d nCompress-1=%d idx=%d\n", i, idx, nCompress-1, idx);
      } else {
        map2[idx - N] = nCompress - 1;
        // printf("identical i=%d sIdx[i]=%d nCompress-1=%d idx=%d\n", i, idx, nCompress-1, idx-N);
      }
    }
    prevCode = xCode[idx];
  }
  // printf(" compress nCompress/N=%d/%d \n", nCompress, N);
  // vectorPrint("x1", x1, N);
  // vectorPrintInt("map1",map1, N);
  // vectorPrint("x2", x2, N);
  // vectorPrintInt("map2",map2, N);
  // vectorPrint("xCompress", xCompress, nCompress);
  delete[] xCode;
  return nCompress;
}

CompressedPads_t* compressPads(const double* xInf, const double* xSup,
                               const double* yInf, const double* ySup, int N)
{
  CompressedPads_t* compressedPads = new CompressedPads_t;
  // On x axe
  compressedPads->xCompressed = new double[2 * N];
  compressedPads->mapXInf = new int[N];
  compressedPads->mapXSup = new int[N];
  compressedPads->nXc = compressSameValues(xInf, xSup, compressedPads->mapXInf, compressedPads->mapXSup, N, compressedPads->xCompressed);
  compressedPads->yCompressed = new double[2 * N];
  compressedPads->mapYInf = new int[N];
  compressedPads->mapYSup = new int[N];
  compressedPads->nYc = compressSameValues(yInf, ySup, compressedPads->mapYInf, compressedPads->mapYSup, N, compressedPads->yCompressed);
  return compressedPads;
}

void deleteCompressedPads(CompressedPads_t* compressedPads)
{
  delete[] compressedPads->mapXInf;
  delete[] compressedPads->mapXSup;
  delete[] compressedPads->mapYInf;
  delete[] compressedPads->mapYSup;
  delete[] compressedPads->xCompressed;
  delete[] compressedPads->yCompressed;
}

void computeCompressed2DPadIntegrals(
  /* const double* xInf, const double* xSup,
                             const double* yInf, const double* ySup,
  */
  CompressedPads_t* compressedPads, double xShift, double yShift, int N,
  int chamberId, double Integrals[])
{

  int nXc = compressedPads->nXc;
  int nYc = compressedPads->nYc;
  // Compute the integrals on Compressed pads
  double xy[N];
  double xPrimitives[nXc];
  double yPrimitives[nYc];
  // X axe
  int axe = 0;
  // x Translation (seed location)
  vectorAddScalar(compressedPads->xCompressed, -xShift, nXc, xy);
  // Primitives on compressed pads
  mathiesonPrimitive(xy, nXc, axe, chamberId, xPrimitives);
  // Y axe
  axe = 1;
  // x Translation (seed location)
  vectorAddScalar(compressedPads->yCompressed, -yShift, nYc, xy);
  // Primitives on compressed pads
  mathiesonPrimitive(xy, nYc, axe, chamberId, yPrimitives);

  // Compute all the integrals
  int* mapXInf = compressedPads->mapXInf;
  int* mapXSup = compressedPads->mapXSup;
  int* mapYInf = compressedPads->mapYInf;
  int* mapYSup = compressedPads->mapYSup;
  for (int i = 0; i < N; i++) {
    Integrals[i] = (xPrimitives[mapXSup[i]] - xPrimitives[mapXInf[i]]) * (yPrimitives[mapYSup[i]] - yPrimitives[mapYInf[i]]);
    // printf(" i=%d mapXInf=%d mapXSup=%d mapYInf=%d mapYSup=%d xyIntegrals=%f, %f \n", i,
    //        mapXInf[i], mapXSup[i], mapYInf[i], mapYSup[i], xPrimitives[mapXSup[i]] - xPrimitives[mapXInf[i]],
    //        yPrimitives[mapYSup[i]] - yPrimitives[mapYInf[i]]);
  }

  // vectorPrint("xPrimitives", xPrimitives, nXc);
  // vectorPrint("yPrimitives", yPrimitives, nYc);
}

void compute2DPadIntegrals(const double* xInf, const double* xSup,
                           const double* yInf, const double* ySup, int N,
                           int chamberId, double Integrals[])
{
  if (1) {
    int mapXInf[N], mapXSup[N];
    int mapYInf[N], mapYSup[N];
    double xy[2 * N];
    // Primitives on x axe
    int nXc = compressSameValues(xInf, xSup, mapXInf, mapXSup, N, xy);
    // vectorPrint("x map", xy, nXc);
    int axe = 0;
    double xPrimitives[nXc];
    mathiesonPrimitive(xy, nXc, axe, chamberId, xPrimitives);
    // Primitives on y axe
    int nYc = compressSameValues(yInf, ySup, mapYInf, mapYSup, N, xy);
    // vectorPrint("y map", xy, nYc);
    double yPrimitives[nYc];
    axe = 1;
    mathiesonPrimitive(xy, nYc, axe, chamberId, yPrimitives);

    for (int i = 0; i < N; i++) {
      Integrals[i] = (xPrimitives[mapXSup[i]] - xPrimitives[mapXInf[i]]) * (yPrimitives[mapYSup[i]] - yPrimitives[mapYInf[i]]);
      // printf(" i=%d mapXInf=%d mapXSup=%d mapYInf=%d mapYSup=%d xyIntegrals=%f, %f \n", i,
      //        mapXInf[i], mapXSup[i], mapYInf[i], mapYSup[i], xPrimitives[mapXSup[i]] - xPrimitives[mapXInf[i]],
      //        yPrimitives[mapYSup[i]] - yPrimitives[mapYInf[i]]);
    }

    // vectorPrint("xPrimitives", xPrimitives, nXc);
    // vectorPrint("yPrimitives", yPrimitives, nYc);

  } else {

    if (useSpline) {
      double lBoundPrim[N], uBoundPrim[N], xIntegrals[N], yIntegrals[N];
      int axe = 0;
      // mathiesonPrimitive(xInf, N, axe, chamberId, lBoundPrim);
      splineMathiesonPrimitive(xInf, N, axe, chamberId, lBoundPrim);
      // mathiesonPrimitive(xSup, N, axe, chamberId, uBoundPrim);
      splineMathiesonPrimitive(xSup, N, axe, chamberId, uBoundPrim);
      vectorAddVector(uBoundPrim, -1.0, lBoundPrim, N, xIntegrals);
      // vectorPrint("xIntegrals analytics ", xIntegrals, N);
      for (int i = 0; i < N; i++) {
        if (xIntegrals[i] < 0.0) {
          printf("??? %d x (%f %f) lInt=%f uInt%f xInt=%f\n", i, xInf[i], xSup[i], lBoundPrim[i], uBoundPrim[i], xIntegrals[i]);
          throw std::out_of_range(
            "[findLocalMaxWithPEM] ????");
        }
      }
      axe = 1;
      // mathiesonPrimitive(yInf, N, axe, chamberId, lBoundPrim);
      splineMathiesonPrimitive(yInf, N, axe, chamberId, lBoundPrim);
      // mathiesonPrimitive(ySup, N, axe, chamberId, uBoundPrim);
      splineMathiesonPrimitive(ySup, N, axe, chamberId, uBoundPrim);
      vectorAddVector(uBoundPrim, -1.0, lBoundPrim, N, yIntegrals);
      // vectorPrint("yIntegrals analytics ", yIntegrals, N);
      vectorMultVector(xIntegrals, yIntegrals, N, Integrals);
      // Invald ????

      for (int i = 0; i < N; i++) {
        if (yIntegrals[i] < 0.0) {
          printf("??? %d y (%f %f) lInt=%f uInt%f yInt=%f\n", i, yInf[i], ySup[i], lBoundPrim[i], uBoundPrim[i], yIntegrals[i]);
          throw std::out_of_range(
            "[findLocalMaxWithPEM] ????");
        }
      } // vectorPrint("Integrals analytics", Integrals, N);

      /* ??????????????????????
      axe = 0;
      splineMathiesonPrimitive( xInf, N, axe, chamberId, lBoundPrim );
      // vectorPrint("x lBoundPrim spline ", lBoundPrim, N);
      splineMathiesonPrimitive( xSup, N, axe, chamberId, uBoundPrim );
      vectorAddVector( uBoundPrim, -1.0, lBoundPrim, N, xIntegrals);
      // vectorPrint("xIntegrals spline", xIntegrals, N);
      axe = 1;
      splineMathiesonPrimitive( yInf, N, axe, chamberId, lBoundPrim );
      splineMathiesonPrimitive( ySup, N, axe, chamberId, uBoundPrim );
      vectorAddVector( uBoundPrim, -1.0, lBoundPrim, N, yIntegrals);

      // vectorPrint("yIntegrals spline", yIntegrals, N);
      */

      vectorMultVector(xIntegrals, yIntegrals, N, Integrals);
      // vectorPrint("Integrals spline", Integrals, N);

    } else {
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
        // printf(" Ix=%10.6g Iy=%10.6g\n", 2*curK4x * (atan(uSup) - atan(uInf)),  2*curK4y * (atan(vSup) - atan(vInf)));
        // printf(" xyInfSup %2d  [%10.6g, %10.6g] x [%10.6g, %10.6g]-> %10.6g * %10.6g = %10.6g\n",
        // i, xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i], 2.0 * curK4x*(atan(uSup) - atan(uInf)), 2.0 * curK4y*(atan(vSup) - atan(vInf)) ) ;
      }
      // printf(" I[0..%3ld] = %f, %f, ... %f\n", N-1, Integrals[0], Integrals[1],
      // Integrals[N-1]);
    }
  }
  // CHECK
  if (clusterConfig.mathiesonCheck) {
    checkIntegrals(xInf, xSup, yInf, ySup, Integrals, chamberId, N);
  }
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

bool checkIntegrals(const double* xInf, const double* xSup, const double* yInf, const double* ySup,
                    const double* integralsToCheck, int chId, int N)
{
  double lBoundPrim[N], uBoundPrim[N];
  double xIntegrals[N], yIntegrals[N], Integrals[N];
  // ??? find the reason for high value
  double precision = 5.e-5;
  int axe = 0;
  mathiesonPrimitive(xInf, N, axe, chId, lBoundPrim);
  mathiesonPrimitive(xSup, N, axe, chId, uBoundPrim);
  vectorAddVector(uBoundPrim, -1.0, lBoundPrim, N, xIntegrals);
  /*
  for (int i=0; i < N; i++) {
    if ( xIntegrals[i] >= 0.0) {
      printf("i=%d xInf=%f xSup=%f, uBoundPrim=%f lBoundPrim=%f\n", i,
              xInf[i], xSup[i], uBoundPrim[i], lBoundPrim[i]);
    }
  }
  */
  axe = 1;
  mathiesonPrimitive(yInf, N, axe, chId, lBoundPrim);
  mathiesonPrimitive(ySup, N, axe, chId, uBoundPrim);
  vectorAddVector(uBoundPrim, -1.0, lBoundPrim, N, yIntegrals);
  // vectorPrint("yIntegrals analytics ", yIntegrals, N);
  vectorMultVector(xIntegrals, yIntegrals, N, Integrals);
  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (std::fabs(integralsToCheck[i] - Integrals[i]) > precision) {
      printf("i=%d xInf=%f xSup=%f, yInf=%f ySup=%f, reference=%f check value=%f\n", i,
             xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i], integralsToCheck[i]);
      ok = false;
      throw std::out_of_range("[checkIntegral] bad integral value");
    }
  }

  return ok;
}

void computeFastCij(const Pads& pads, const Pads& pixel, double Cij[])
{
  // Compute the Charge Integral Cij of pads (j index), considering the
  // center of the Mathieson fct on a pixel (i index)
  // Use the fact that the charge integral CI(x,y) = CI(x) * CI(y)
  // to reduce the computation cost
  // CI(x) is store in PadIntegralX
  // CI(y) is store in PadIntegralY
  // A sub-sampling of CI(x_i + k*minDx) (or CI(y_i + l*minDY)) is used
  // by taking the mininimun of pads.dx(pads.dy) to discretize the x/y space
  //
  // CI(x)/CI(y) are computed if they are requested.
  //
  // Returning array: Charge Integral on all the pads Cij[]

  if ((pads.mode != Pads::PadMode::xyInfSupMode) || (pixel.mode != Pads::PadMode::xydxdyMode)) {
    printf(
      "[computeFastCij] exception: bad representation (mode) of pads in "
      "computeCij (padMode=%d, pixelMode=%d)\n",
      (int)pads.mode, (int)pixel.mode);
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

  double zInf[N];
  double zSup[N];
  int axe;

  // Loop on Pixels
  std::map<int, double*> xMap;
  std::map<int, double*> yMap;
  for (int k = 0; k < K; k++) {
    // Calculate the indexes in the 1D charge integral
    // Error on pad position > 10-3 cm
    int xCode = (int)(muX[k] * 1000 + 0.5);
    int yCode = (int)(muY[k] * 1000 + 0.5);
    if (xMap.find(xCode) == xMap.end()) {
      // Not yet computed
      vectorAddScalar(xInf0, -muX[k], N, zInf);
      vectorAddScalar(xSup0, -muX[k], N, zSup);
      axe = 0;
      double* xIntegrals = new double[N];
      compute1DPadIntegrals(zInf, zSup, N, axe, chId, xIntegrals);
      xMap[xCode] = xIntegrals;
    }
    if (yMap.find(yCode) == yMap.end()) {
      // Not yet computed
      vectorAddScalar(yInf0, -muY[k], N, zInf);
      vectorAddScalar(ySup0, -muY[k], N, zSup);
      axe = 1;
      double* yIntegrals = new double[N];
      compute1DPadIntegrals(zInf, zSup, N, axe, chId, yIntegrals);
      yMap[yCode] = yIntegrals;
    }
    // Compute IC(xy) = IC(x) * IC(y)
    vectorMultVector(xMap[xCode], yMap[yCode], N, &Cij[N * k]);
    //
    // Check
    if (clusterConfig.mathiesonCheck) {
      double xInf[N], xSup[N];
      double yInf[N], ySup[N];
      double lBoundPrim[N], uBoundPrim[N];
      double xIntegrals[N], yIntegrals[N], Integrals[N];
      // printf("pad xyPad[0]= %f %f \n", (xSup0[0] - xInf0[0])*0.5, (ySup0[0] - yInf0[0])*0.5);
      // printf("pad xyPad[0]= %f %f \n", xSup0[0], ySup0[0]);
      // printf("pad xyPix[0]= %f %f \n", muX[k], muY[k]);
      vectorAddScalar(xInf0, -muX[k], N, xInf);
      vectorAddScalar(xSup0, -muX[k], N, xSup);
      vectorAddScalar(yInf0, -muY[k], N, yInf);
      vectorAddScalar(ySup0, -muY[k], N, ySup);
      checkIntegrals(xInf, xSup, yInf, ySup, &Cij[N * k], chId, N);
    }
  }
  // Free map
  for (auto it = xMap.begin(); it != xMap.end(); ++it) {
    delete[] it->second;
  }
  for (auto it = yMap.begin(); it != yMap.end(); ++it) {
    delete[] it->second;
  }
}

void computeFastCijV0(const Pads& pads, const Pads& pixel, double Cij[])
{
  // Compute the Charge Integral Cij of pads (j index), considering the
  // center of the Mathieson fct on a pixel (i index)
  // Use the fact that the charge integral CI(x,y) = CI(x) * CI(y)
  // to reduce the computation cost
  // CI(x) is store in PadIntegralX
  // CI(y) is store in PadIntegralY
  // A sub-sampling of CI(x_i + k*minDx) (or CI(y_i + l*minDY)) is used
  // by taking the mininimun of pads.dx(pads.dy) to discretize the x/y space
  //
  // CI(x)/CI(y) are computed if they are requested.
  //
  // Returning array: Charge Integral on all the pads Cij[]

  if ((pads.mode != Pads::PadMode::xyInfSupMode) || (pixel.mode != Pads::PadMode::xydxdyMode)) {
    printf(
      "[computeFastCij] exception: bad representation (mode) of pads in "
      "computeCij (padMode=%d, pixelMode=%d)\n",
      (int)pads.mode, (int)pixel.mode);
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
  double dxMinPix = vectorMin(pixel.getDX(), K);
  double dyMinPix = vectorMin(pixel.getDY(), K);
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
  int axe;
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
      axe = 0;
      compute1DPadIntegrals(zInf, zSup, N, axe, chId, &PadIntegralX[xIdx * N + 0]);
    }
    if (PadIntegralY[yIdx * N + 0] == -1) {
      // Not yet computed
      vectorAddScalar(yInf0, -muY[k], N, zInf);
      vectorAddScalar(ySup0, -muY[k], N, zSup);
      axe = 1;
      compute1DPadIntegrals(zInf, zSup, N, axe, chId, &PadIntegralY[yIdx * N + 0]);
    }
    // Compute IC(xy) = IC(x) * IC(y)
    vectorMultVector(&PadIntegralX[xIdx * N + 0], &PadIntegralY[yIdx * N + 0], N, &Cij[N * k]);

    double xInf[N], xSup[N];
    double yInf[N], ySup[N];
    double lBoundPrim[N], uBoundPrim[N];
    double xIntegrals[N], yIntegrals[N], Integrals[N];

    vectorAddScalar(xInf0, -muX[k], N, xInf);
    vectorAddScalar(xSup0, -muX[k], N, xSup);
    vectorAddScalar(yInf0, -muY[k], N, yInf);
    vectorAddScalar(ySup0, -muY[k], N, ySup);
    double integral;
    int axe = 0;
    mathiesonPrimitive(xInf, N, axe, chId, lBoundPrim);
    mathiesonPrimitive(xSup, N, axe, chId, uBoundPrim);
    vectorAddVector(uBoundPrim, -1.0, lBoundPrim, N, xIntegrals);
    axe = 1;
    mathiesonPrimitive(yInf, N, axe, chId, lBoundPrim);
    mathiesonPrimitive(ySup, N, axe, chId, uBoundPrim);
    vectorAddVector(uBoundPrim, -1.0, lBoundPrim, N, yIntegrals);
    // vectorPrint("yIntegrals analytics ", yIntegrals, N);
    vectorMultVector(xIntegrals, yIntegrals, N, Integrals);
    for (int i = 0; i < N; i++) {
      // compute2DPadIntegrals(xInf[i], xSup, yInf, ySup, 1, chId, &integral);
      if (std::fabs(Cij[N * k + i] - Integrals[i]) > 1.0e-6) {
        printf("i(pixel)=%d j(pad)=%d cij=%f xInt=%f yInt=%f fastcij=%f xFast=%f yFast=%f\n", k, i,
               Integrals[i], xIntegrals[i], yIntegrals[i], Cij[N * k + i], PadIntegralX[xIdx * N + i], PadIntegralY[yIdx * N + i]);
      }
    }
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

  if ((pads.mode != Pads::PadMode::xyInfSupMode) || (pixel.mode != Pads::PadMode::xydxdyMode)) {
    printf(
      "computeFastCij] exception: bad representation (mode) of pads in "
      "computeCij (padMode=%d, pixelMode=%d)\n",
      (int)pads.mode, (int)pixel.mode);
    throw std::overflow_error("Bad mode");
    return;
  }
  int N = pads.getNbrOfPads();
  int K = pixel.getNbrOfPads();
  int chId = pads.getChamberId();
  const double* xInf0 = pads.getXInf();
  const double* yInf0 = pads.getYInf();
  const double* xSup0 = pads.getXSup();
  const double* ySup0 = pads.getYSup();

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

void checkCij(const Pads& pads, const Pads& pixels, const double* checkCij, int mode)
{
  // Mode : 0 (nothing), 1 (info), 2 (detail), -1 (exception)
  int nPads = pads.getNbrOfPads();
  int nPixels = pixels.getNbrOfPads();
  double* Cij = new double[nPads * nPixels];
  double* diffCij = new double[nPads * nPixels];
  double precision = 2.0e-5;
  computeCij(pads, pixels, Cij);
  vectorAddVector(Cij, -1, checkCij, nPads * nPixels, diffCij);
  vectorAbs(diffCij, nPads * nPixels, diffCij);
  double minDiff = vectorMin(diffCij, nPads * nPixels);
  double maxDiff = vectorMax(diffCij, nPads * nPixels);
  int argMax = vectorArgMax(diffCij, nPads * nPixels);
  // printf("\n\n nPads, nPixels %d %d\n", nPads, nPixels);
  int iIdx = argMax / nPads;
  int jIdx = argMax % nPads;
  if ((maxDiff > precision) && (mode != 0)) {
    printf("\n\n[checkCij] min/max(checkCij-Cij)=(%f, %f) argmin/max=(i=%d, j=%d)\n",
           minDiff, maxDiff, iIdx, jIdx);
    printf("\n checkCij=%f differ from  Cij=%f\n", checkCij[iIdx * nPads + jIdx], Cij[iIdx * nPads + jIdx]);
  }

  if ((maxDiff > precision) && (mode > 1)) {
    for (int k = 0; k < nPixels; k++) {
      for (int l = 0; l < nPads; l++) {
        if (diffCij[k * nPads + l] > precision) {
          printf("pad=%d pixel=%d checkCij=%f Cij=%f diff=%f\n", l, k, checkCij[k * nPads + l], Cij[k * nPads + l], diffCij[k * nPads + l]);
        }
      }
    }
    // printf("findLocalMaxWithPEM: WARNING maxDiff(Cij)=%f\n", maxDiff);
  }
  if ((maxDiff > precision) && (mode == -1)) {
    throw std::out_of_range("[checkCij] bad Cij value");
  }
  delete[] Cij;
  delete[] diffCij;
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
void o2_mch_initMathieson()
{
  o2::mch::initMathieson(o2::mch::clusterConfig.useSpline, 0);
}

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
