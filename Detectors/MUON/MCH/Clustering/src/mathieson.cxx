#include <stdio.h>
#include <stdlib.h>

#include "MCHClustering/dataStructure.h"
#include "MCHClustering/mathUtil.h"
#include "MCHClustering/mathieson.h"

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
// ??? The Pad Integrals are store here:
// double *I;

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

void compute2DPadIntegrals(const double* xyInfSup,
                           int N, int chamberId, double Integrals[])
{
  // Returning array: Charge Integral on all the pads
  const double* xInf = getConstXInf(xyInfSup, N);
  const double* xSup = getConstXSup(xyInfSup, N);
  const double* yInf = getConstYInf(xyInfSup, N);
  const double* ySup = getConstYSup(xyInfSup, N);
  //
  if (chamberId <= 2)
    mathiesonType = 0;
  else
    mathiesonType = 1;
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
    // printf(" xyInfSup %2d  [%10.6g, %10.6g] x [%10.6g, %10.6g]-> %10.6g\n", i, xInf[i], xSup[i], yInf[i], ySup[i], Integrals[i]);
  }
  // printf(" I[0..%3ld] = %f, %f, ... %f\n", N-1, Integrals[0], Integrals[1], Integrals[N-1]);
  return;
}

void compute2DMathiesonMixturePadIntegrals(const double* xyInfSup0, const double* theta,
                                           int N, int K, int chamberId, double Integrals[])
{
  // Returning array: Charge Integral on all the pads
  // Remarks:
  // - This fct is a cumulative one, as a result it should be set to zero
  //    before calling it
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
    compute2DPadIntegrals(xyInfSup, N, chamberId, z);
    // printf("Vector Sum %g\n", vectorSum(z, N) );
    vectorAddVector(Integrals, w[k], z, N, Integrals);
  }
}
