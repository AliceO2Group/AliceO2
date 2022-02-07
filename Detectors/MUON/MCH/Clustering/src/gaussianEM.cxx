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

#include <cmath>
#include <cfloat>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include "MCHClustering/dataStructure.h"
#include "MCHClustering/mathUtil.h"
#include "gaussianEM.h"

//
// TODO : Optimization,  generateMixedGaussians2D computed twice.
//

static int nIterMin = 10;
static int nIterMax = 400;

void printEMState(int it, double logL, double dLogL)
{
  printf("EM step %d, logL =%15.9g, dLogL =%15.9g\n", it, logL, dLogL);
}

void printArrayState(const char* str, const double* array, int K, int N)
{
  printf("%s\n min/max/sum %15.9g, %15.9g, %15.9g\n",
         str, vectorMin(array, K * N), vectorMax(array, K * N), vectorSum(array, K * N));
}
double crossEntropy(const double* zObs, const double* zPredicted, int N)
{
  double cross = 0.0;
  for (int i = 0; i < N; i++) {
    cross += (zObs[i] * log(zPredicted[i]));
  }
  return cross;
}

void computeDiscretizedGaussian2D(const double* xyInfSup, const double* theta,
                                  int K, int N, int k, double* z)
{
  // compute the k-component of the mixture
  const double* xInf = getConstXInf(xyInfSup, N);
  const double* yInf = getConstYInf(xyInfSup, N);
  const double* xSup = getConstXSup(xyInfSup, N);
  const double* ySup = getConstYSup(xyInfSup, N);

  const double* varX = getConstVarX(theta, K);
  const double* varY = getConstVarY(theta, K);
  const double* muX = getConstMuX(theta, K);
  const double* muY = getConstMuY(theta, K);
  // const double *w   = getConstW(theta);;

  // Compute
  /*
    ISupSup[:] = 0.25*( 1.0 + vErf( (xsup[:]-muX) * cstx ) ) * ( 1.0 + vErf( (ysup[:]-muY) * csty ) )
    IInfInf[:] = 0.25*( 1.0 + vErf( (xinf[:]-muX) * cstx ) ) * ( 1.0 + vErf( (yinf[:]-muY) * csty ) )
    IInfSup[:] = 0.25*( 1.0 + vErf( (xinf[:]-muX) * cstx ) ) * ( 1.0 + vErf( (ysup[:]-muY) * csty ) )
    ISupInf[:] = 0.25*( 1.0 + vErf( (xsup[:]-muX) * cstx ) ) * ( 1.0 + vErf( (yinf[:]-muY) * csty ) )
    */
  double IxInf[N];
  double IyInf[N];
  double IxSup[N];
  double IySup[N];
  double sigX, sigY;
  double cstx, csty;

  if (varX[k] < 1.e-15) {
    for (int i = 0; i < N; i++) {
      IxInf[i] = ((xInf[i] - muX[k]) < 0.0) ? 0.0 : 2.0;
      IxSup[i] = ((xSup[i] - muX[k]) < 0.0) ? 0.0 : 2.0;
    }
  } else {
    sigX = sqrt(varX[k]);
    cstx = 1.0 / (sqrt(2.0) * sigX);
    for (int i = 0; i < N; i++) {
      IxInf[i] = 1.0 + erf((xInf[i] - muX[k]) * cstx);
      IxSup[i] = 1.0 + erf((xSup[i] - muX[k]) * cstx);
    }
  }
  if (varY[k] < 1.e-15) {
    for (int i = 0; i < N; i++) {
      IyInf[i] = ((yInf[i] - muY[k]) < 0.0) ? 0.0 : 2.0;
      IySup[i] = ((ySup[i] - muY[k]) < 0.0) ? 0.0 : 2.0;
    }
  } else {
    sigY = sqrt(varY[k]);
    csty = 1.0 / (sqrt(2.0) * sigY);
    for (int i = 0; i < N; i++) {
      IyInf[i] = 1.0 + erf((yInf[i] - muY[k]) * csty);
      IySup[i] = 1.0 + erf((ySup[i] - muY[k]) * csty);
    }
  }

  // Compute
  //
  // ISupSup[:] = 0.25 * IxSup[:] * IySup[:]
  // IInfInf[:] = 0.25 * IXInf[:] * IyInf[:]
  // IInfSup[:] = 0.25 * IXInf[:] * IYSup[:]
  // ISupInf[:] = 0.25 * IXSup[:] * IYInf[:]
  // z[:] = ISupSup[:] - IInfSup[:] - ISupInf[:] + IInfInf[:];

  for (int i = 0; i < N; i++) {
    z[i] = 0.25 * IxSup[i] * IySup[i];
    z[i] += 0.25 * IxInf[i] * IyInf[i];
    z[i] -= 0.25 * IxInf[i] * IySup[i];
    z[i] -= 0.25 * IxSup[i] * IyInf[i];
  }
  return;
}

void generateMixedGaussians2D(const double* xyInfSup, const double* theta, int K, int N, double* z)
{
  double zk[N];
  const double* w = getConstW(theta, K);
  vectorSetZero(z, N);
  for (int k = 0; k < K; k++) {
    if (w[k] > 0.0) {
      computeDiscretizedGaussian2D(xyInfSup, theta, K, N, k, zk);
      // z[i] = w[k] * zk[i]
      vectorAddVector(z, w[k], zk, N, z);
    }
  }
}

double computeWeightedLogLikelihood(const double* xyInfSup, const double* theta, const double* z, int K, int N)
{
  // w is assumed to be normalized
  double li[N];
  double LogLikelihood = 0;

  generateMixedGaussians2D(xyInfSup, theta, K, N, li);
  // Avoiding computing log( small values ) replace by 0.0 (meaning no contribution)
  for (int i = 0; i < N; i++) {
    // TODO: avoid writing in li
    li[i] = (li[i] < DBL_EPSILON) ? 0.0 : log(li[i]) * z[i];
    LogLikelihood += li[i];
  }
  return LogLikelihood;
}

void EStep(const double* xyInfSup, const double* theta, int K, int N, int verbose, double* eta, double* zEval)
{
  /*
          Compute new eta(i,k), fraction/proba that point i
          belongs to kth gaussian
*/
  double kSum[N];
  // ??? To remove : double zEval[N];
  vectorSetZero(eta, K * N);
  vectorSetZero(kSum, N);
  const double* w = getConstW(theta, K);

  for (int k = 0; k < K; k++) {
    // eta[:, k] = w[k] * computeDiscretizedGaussian2D( xy[:,:], dxy, mu[k,:], var[k,:] )
    // kSum += eta[:,k]
    if (w[k] != 0.0) {
      computeDiscretizedGaussian2D(xyInfSup, theta, K, N, k, zEval);
      vectorMultScalar(zEval, w[k], N, &eta[k * N]);
      vectorAddVector(kSum, 1.0, &eta[k * N], N, kSum);
    }
  }
  // eta normalisation
  for (int i = 0; i < N; i++) {
    if (kSum[i] < DBL_EPSILON) {
      // print( "WARNING sum eta(i,:) is null, i=",i , "xy=", xy[:,i], "dxy=",dxy[:,i])
      // eta[ i, :] = eta[i, :] / kSum[i]
      for (int k = 0; k < K; k++) {
        eta[k * N + i] = 0.;
      }
    } else {
      // eta[ i, :] = eta[i, :] / kSum[i]
      for (int k = 0; k < K; k++) {
        eta[k * N + i] = eta[k * N + i] / kSum[i];
      }
    }
  }
  if (verbose >= 2) {
    double nSum[K];
    vectorSetZero(kSum, N);
    vectorSetZero(nSum, K);
    for (int i = 0; i < N; i++) {
      for (int k = 0; k < K; k++) {
        kSum[i] += eta[k * N + i];
        nSum[k] += eta[k * N + i];
      }
    }
    printf("  E-Step check: sum_k{eta}[0:N] min=%f max=%f\n", vectorMin(kSum, N), vectorMax(kSum, N));
    printf("  E-Step check: sum_n{eta}[0:K] min=%f max=%f\n", vectorMin(nSum, K), vectorMax(nSum, K));
    printf("  E-Step check: sum_{eta} =%f\n", vectorSum(nSum, K));
    vectorPrint("  E-Step check: sum_k{eta}[0:N]", kSum, N);
  }
  return;
}

void maskedEStep(const double* xyInfSup, const double* theta, const Mask_t* maskTheta, int K, int N, int verbose, double* eta, double* zEval)
{
  /*
          Compute new eta(i,k), fraction/proba that point i
          belongs to kth gaussian
*/
  double kSum[N];
  // ??? To remove : double zEval[N];
  vectorSetZero(eta, K * N);
  vectorSetZero(kSum, N);
  const double* w = getConstW(theta, K);

  for (int k = 0; k < K; k++) {
    // eta[:, k] = w[k] * computeDiscretizedGaussian2D( xy[:,:], dxy, mu[k,:], var[k,:] )
    // kSum += eta[:,k]
    if (maskTheta[k]) {
      computeDiscretizedGaussian2D(xyInfSup, theta, K, N, k, zEval);
      vectorMultScalar(zEval, w[k], N, &eta[k * N]);
      vectorAddVector(kSum, 1.0, &eta[k * N], N, kSum);
    }
  }
  // eta normalisation
  for (int i = 0; i < N; i++) {
    if (kSum[i] < DBL_EPSILON) {
      // print( "WARNING sum eta(i,:) is null, i=",i , "xy=", xy[:,i], "dxy=",dxy[:,i])
      // eta[ i, :] = eta[i, :] / kSum[i]
      for (int k = 0; k < K; k++) {
        eta[k * N + i] = 0.;
      }
    } else {
      // eta[ i, :] = eta[i, :] / kSum[i]
      for (int k = 0; k < K; k++) {
        // if (maskTheta[k]) eta[ k*N + i] = eta[ k*N + i]/kSum[i];
        eta[k * N + i] = eta[k * N + i] / kSum[i];
      }
    }
  }
  if (verbose >= 2) {
    double nSum[K];
    vectorSetZero(kSum, N);
    vectorSetZero(nSum, K);
    for (int i = 0; i < N; i++) {
      for (int k = 0; k < K; k++) {
        kSum[i] += eta[k * N + i];
        nSum[k] += eta[k * N + i];
      }
    }
    printf("  EMask-Step check: sum_k{eta}[0:N] min=%f max=%f\n", vectorMin(kSum, N), vectorMax(kSum, N));
    printf("  EMask-Step check: sum_n{eta}[0:K] min=%f max=%f\n", vectorMin(nSum, K), vectorMax(nSum, K));
    printf("  EMask-Step check: sum_{eta} =%f\n", vectorSum(nSum, K));
    vectorPrint("  EMask-Step check: sum_k{eta}[0:N]", kSum, N);
  }
  return;
}

void weightedMStep(const double* xyDxy, const double* z, const double* eta, int K, int N, int cstVar, double* theta)
{
  // xyDxy[N,4]
  // z[N]
  // eta[K, N]

  const double* x = getConstX(xyDxy, N);
  const double* y = getConstY(xyDxy, N);

  double* varX = getVarX(theta, K);
  double* varY = getVarY(theta, K);
  double* muX = getMuX(theta, K);
  double* muY = getMuY(theta, K);
  double* w = getW(theta, K);

  // Compute unnormalized wk
  // w[k] = sum_i{ eta[ k, i] * z[i] }
  double wk[K];
  gsl_vector_const_view gsl_z = gsl_vector_const_view_array(z, N);
  gsl_vector_view gsl_wk = gsl_vector_view_array(wk, K);
  gsl_matrix_const_view gsl_eta = gsl_matrix_const_view_array(eta, K, N);
  gsl_blas_dgemv(CblasNoTrans, 1.0, &gsl_eta.matrix, &gsl_z.vector, 0.0, &gsl_wk.vector);

  // vectorPrint( "??? MStep ", wk, K);

  double u[N];
  gsl_vector_view gsl_u = gsl_vector_view_array(u, N);
  gsl_vector_view gsl_muX = gsl_vector_view_array(muX, K);
  gsl_vector_view gsl_muY = gsl_vector_view_array(muY, K);
  //
  //  Compute muX
  //
  // u[i] = x[i]*z[i]
  vectorMultVector(x, z, N, u);
  // muX[k] = sum_i{ eta[ k, i] * x[i] * z[i] }
  gsl_blas_dgemv(CblasNoTrans, 1.0, &gsl_eta.matrix, &gsl_u.vector, 0.0, &gsl_muX.vector);
  //
  //  Compute muY
  //
  // u[i] = x[i]*z[i]
  vectorMultVector(y, z, N, u);
  // muX[k] = sum_i{ eta[ k, i] * y[i] * z[i] }
  gsl_blas_dgemv(CblasNoTrans, 1.0, &gsl_eta.matrix, &gsl_u.vector, 0.0, &gsl_muY.vector);
  //
  // Normalization muX, muY
  // Wk[k] = 0, the component is meaningless
  for (int k = 0; k < K; k++) {
    muX[k] = (w[k] < DBL_EPSILON) ? 0.0 : muX[k] / wk[k];
    muY[k] = (w[k] < DBL_EPSILON) ? 0.0 : muY[k] / wk[k];
  }

  // Compute varX, varY
  if (cstVar == 0) {
    double x2[N], y2[N];
    gsl_vector_view gsl_x2 = gsl_vector_view_array(x2, N);
    gsl_vector_view gsl_y2 = gsl_vector_view_array(y2, N);
    for (int k = 0; k < K; k++) {
      for (int i = 0; i < N; i++) {
        x2[i] = x[i] - muX[k];
        x2[i] = x2[i] * x2[i];
        x2[i] = x2[i] * z[i];
        y2[i] = y[i] - muY[k];
        y2[i] = y2[i] * y2[i];
        y2[i] = y2[i] * z[i];
      }
      gsl_vector_const_view gsl_etaK = gsl_vector_const_view_array(&eta[k * N], N);
      gsl_blas_ddot(&gsl_etaK.vector, &gsl_x2.vector, &varX[k]);
      gsl_blas_ddot(&gsl_etaK.vector, &gsl_y2.vector, &varY[k]);
      // varX, varY normalization
      if (wk[k] < DBL_EPSILON) {
        varX[k] = 0.0;
        varY[k] = 0.0;
      } else {
        varX[k] = varX[k] / wk[k];
        varY[k] = varY[k] / wk[k];
        // To avoid numerical pb
        varX[k] = (varX[k] < DBL_EPSILON) ? 0.0 : varX[k];
        varY[k] = (varY[k] < DBL_EPSILON) ? 0.0 : varY[k];
      }
    }
  }
  //
  // Calculate w (normalization)
  //  w = wk / N
  //
  // Build the mask from eta
  // because of the  points which don't
  // participate eta[k, :] = 0
  double etaN[N];
  vectorSetZero(etaN, N);
  for (int k = 0; k < K; k++) {
    for (int i = 0; i < N; i++) {
      etaN[i] += eta[k * N + i];
    }
  }
  double zMasked[N];
  vectorMultVector(z, etaN, N, zMasked);
  double sum = vectorSum(zMasked, N);
  // verbose ???
  if (0) {
    vectorPrint("MStep z", z, N);
    vectorPrint("MStep wk", wk, K);
  }
  vectorMultScalar(wk, 1.0 / sum, K, w);
  return;
}

double weightedEMLoop(const double* xyDxy, const Mask_t* saturated, const double* zObs,
                      const double* theta0, const Mask_t* maskTheta, int K, int N,
                      int mode, double LConvergence, int verbose, double* theta)
{

  // Mode of computation
  // TODO make a function
  int m = mode;
  int cstVarMode = m & 0x1;
  // Saturated
  // Not selected way ??? int nbrSaturatedPads = (m >> 1) & 0x1;
  int nbrSaturatedPads = vectorSumShort(saturated, N);
  if (verbose >= 2) {
    printf("  wheightedEMLoop : cstVarMode =%d\n", cstVarMode);
    printf("  wheightedEMLoop : nbrSaturatedPads =%d\n", nbrSaturatedPads);
  }
  // vectorPrint("  zObs", zObs, N);
  // vectorPrint("  xydxy", xyDxy, N*4);
  // Define theta
  vectorCopy(theta0, K * 5, theta);
  double* varX = getVarX(theta, K);
  double* varY = getVarY(theta, K);
  double* muX = getMuX(theta, K);
  double* muY = getMuY(theta, K);
  double* w = getW(theta, K);

  // Define eta
  double eta[N * K];

  // define x, y, dx, dy description
  const double* x = getConstX(xyDxy, N);
  const double* y = getConstY(xyDxy, N);
  const double* dX = getConstDX(xyDxy, N);
  const double* dY = getConstDY(xyDxy, N);

  // Compute boundary of each pads
  double xyInfSup[4 * N];
  vectorAddVector(x, -1.0, dX, N, getXInf(xyInfSup, N));
  vectorAddVector(y, -1.0, dY, N, getYInf(xyInfSup, N));
  vectorAddVector(x, +1.0, dX, N, getXSup(xyInfSup, N));
  vectorAddVector(y, +1.0, dY, N, getYSup(xyInfSup, N));
  //
  // z normalization
  double zSum = vectorSum(zObs, N);
  double zObsNorm[N];
  vectorMultScalar(zObs, 1.0 / zSum, N, zObsNorm);
  //
  // Initial Likelihood
  if (maskTheta) {
    for (int k = 0; k < K; k++) {
      w[k] = (maskTheta[k] == 1) ? w[k] : 0;
    }
  }
  double logL = computeWeightedLogLikelihood(xyInfSup, theta0, zObsNorm, K, N);
  if (verbose >= 1) {
    printEMState(-1, logL, 0.0);
  }
  //
  // EM Loop
  //
  double zEval[N];
  double prevLogL = logL;
  logL = 1.0 + prevLogL;
  int it = 0;
  // for( ; (( fabs((logL - prevLogL)/logL) > LConvergence) || (it < nIterMin)) && ( it < nIterMax ); ) {
  for (; ((fabs((logL - prevLogL) / logL) > LConvergence)) && (it < nIterMax);) {
    if (verbose >= 1) {
      printEMState(it, logL, logL - prevLogL);
    }
    prevLogL = logL;
    //
    // EM Step
    //
    // E-Step
    if (maskTheta == nullptr) {
      EStep(xyInfSup, theta, K, N, verbose, eta, zEval);
    } else {
      maskedEStep(xyInfSup, theta, maskTheta, K, N, verbose, eta, zEval);
      //
    }
    if (nbrSaturatedPads > 0) {
      // Set (or update) saturated pads to the estimate
      // values i.e. zEval
      // Remark
      // saturated is a constant array and
      // the zObsNorm[ saturated == 0 )] are not modified
      // and take the zObsNorm Values
      vectorMaskedUpdate(saturated, zEval, N, zObsNorm);
    }
    // M-Step
    weightedMStep(xyDxy, zObsNorm, eta, K, N, cstVarMode, theta);
    // Log-Lilelihood
    logL = computeWeightedLogLikelihood(xyInfSup, theta, zObsNorm, K, N);

    if (verbose >= 2) {
      printTheta("  EM new theta", theta, K);
    }
    it += 1;
  }
  if (verbose >= 1)
    printEMState(it, logL, logL - prevLogL);
  if (verbose >= 2)
    printf("End GaussianEM\n");

  // Return BIC criterion
  int kSignificant = vectorSumOfGreater(w, 10.e-5, K);
  printf("EMLoop # parameters %d, log( N -saturated)= %f \n", (3 * kSignificant - 1), log(N - nbrSaturatedPads));
  printf("EMLoop LogL=%f\n", logL);
  double dof = (3 * kSignificant - 1);
  double nData = N - nbrSaturatedPads;
  double BIC = -2 * logL + (3 * kSignificant - 1) * log(N - nbrSaturatedPads);
  double AIC = 2 * (3 * kSignificant - 1) - 2 * logL;
  double AICc = 2 * (3 * kSignificant - 1) - 2 * logL + 2 * dof * (dof + 1) / (nData - dof - 1);
  double zPredict[N];
  generateMixedGaussians2D(xyInfSup, theta, K, N, zPredict);
  double crossE = crossEntropy(zObs, zPredict, N);
  printf("EMLoop BIC=%f, AIC=%f, AICc=%f crossEntropy=%f\n", BIC, AIC, AICc, crossE);

  return logL;
}

double weightedEMLoopWithMuCriterion(const double* xyDxy, const Mask_t* saturated, const double* zObs,
                                     const double* theta0, const Mask_t* maskTheta, int K, int N,
                                     int mode, double LConvergence, int verbose, double* theta)
{

  // Mode of computation
  // TODO make a function
  int m = mode;
  int cstVarMode = m & 0x1;
  // Saturated
  // Not selected way ??? int nbrSaturatedPads = (m >> 1) & 0x1;
  int nbrSaturatedPads = vectorSumShort(saturated, N);
  if (verbose >= 2) {
    printf("  wheightedEMLoop : cstVarMode =%d\n", cstVarMode);
    printf("  wheightedEMLoop : nbrSaturatedPads =%d\n", nbrSaturatedPads);
  }
  // vectorPrint("  zObs", zObs, N);
  // vectorPrint("  xydxy", xyDxy, N*4);
  // Define theta
  vectorCopy(theta0, K * 5, theta);
  double* varX = getVarX(theta, K);
  double* varY = getVarY(theta, K);
  double* muX = getMuX(theta, K);
  double* muY = getMuY(theta, K);
  double* w = getW(theta, K);

  // Define eta
  double eta[N * K];

  // define x, y, dx, dy description
  const double* x = getConstX(xyDxy, N);
  const double* y = getConstY(xyDxy, N);
  const double* dX = getConstDX(xyDxy, N);
  const double* dY = getConstDY(xyDxy, N);

  // Compute boundary of each pads
  double xyInfSup[4 * N];
  vectorAddVector(x, -1.0, dX, N, getXInf(xyInfSup, N));
  vectorAddVector(y, -1.0, dY, N, getYInf(xyInfSup, N));
  vectorAddVector(x, +1.0, dX, N, getXSup(xyInfSup, N));
  vectorAddVector(y, +1.0, dY, N, getYSup(xyInfSup, N));
  double minDxPad = vectorMin(dX, N);
  double minDyPad = vectorMin(dY, N);
  //
  // z normalization
  double zSum = vectorSum(zObs, N);
  double zObsNorm[N];
  vectorMultScalar(zObs, 1.0 / zSum, N, zObsNorm);
  //
  // Initial Likelihood
  if (maskTheta) {
    for (int k = 0; k < K; k++) {
      w[k] = (maskTheta[k] == 1) ? w[k] : 0;
    }
  }
  double logL = computeWeightedLogLikelihood(xyInfSup, theta0, zObsNorm, K, N);
  if (verbose >= 1) {
    printEMState(-1, logL, 0.0);
  }
  //
  // EM Loop
  //
  double prevTheta[5 * K];
  double zEval[N];
  double prevLogL = logL;
  vectorCopy(theta, K * 5, prevTheta);
  double* prevMuX = getMuX(prevTheta, K);
  double* prevMuY = getMuY(prevTheta, K);
  double* prevW = getW(prevTheta, K);

  logL = 1.0 + prevLogL;
  int it = 0;
  int muChanged = 0;
  // for( ; (( fabs((logL - prevLogL)/logL) > LConvergence) || (it < nIterMin)) && ( it < nIterMax ); ) {
  for (; (muChanged == 0) && (fabs((logL - prevLogL) / logL) > LConvergence) && (it < nIterMax);) {
    if (verbose >= 1)
      printEMState(it, logL, logL - prevLogL);
    prevLogL = logL;
    //
    // EM Step
    //
    // E-Step
    if (maskTheta == nullptr) {
      EStep(xyInfSup, theta, K, N, verbose, eta, zEval);
    } else {
      maskedEStep(xyInfSup, theta, maskTheta, K, N, verbose, eta, zEval);
    }
    //
    if (nbrSaturatedPads > 0) {
      // Set (or update) saturated pads to the estimate
      // values i.e. zEval
      // Remark
      // saturated is a constant array and
      // the zObsNorm[ saturated == 0 )] are not modified
      // and take the zObsNorm Values
      vectorMaskedUpdate(saturated, zEval, N, zObsNorm);
    }
    // M-Step
    weightedMStep(xyDxy, zObsNorm, eta, K, N, cstVarMode, theta);
    // Log-Lilelihood
    logL = computeWeightedLogLikelihood(xyInfSup, theta, zObsNorm, K, N);

    if (verbose >= 2) {
      printTheta("  EM new theta", theta, K);
    }
    // Mu Changed
    // Coresponding pad in projection ???
    Mask_t significantMuChange[N];
    for (int k = 0; k < K; k++) {
      significantMuChange[k] = 0;
      double diffX = fabs(prevMuX[k] - muX[k]);
      double diffY = fabs(prevMuY[k] - muY[k]);
      if (maskTheta[k] & ((diffX > minDxPad) || (diffY > minDyPad))) {
        significantMuChange[k] = 1;
        muChanged = 1;
      }
    }
    //
    vectorCopy(theta, K * 5, prevTheta);
    it += 1;
  }
  if (verbose >= 1) {
    printEMState(it, logL, logL - prevLogL);
  }
  if (verbose >= 2) {
    printf("End GaussianEM\n");
  }
  // Return BIC criterion
  int kSignificant = vectorSumOfGreater(w, 10.e-5, K);
  // printf("EMLoop # parameters %d, log( N -saturated)= %f \n", (3*kSignificant-1), log(N - nbrSaturatedPads) );
  // printf("EMLoop LogL=%f\n", logL );
  double dof = (3 * kSignificant - 1);
  double nData = N - nbrSaturatedPads;
  double BIC = -2 * logL + (3 * kSignificant - 1) * log(N - nbrSaturatedPads);
  double AIC = 2 * (3 * kSignificant - 1) - 2 * logL;
  double AICc = 2 * (3 * kSignificant - 1) - 2 * logL + 2 * dof * (dof + 1) / (nData - dof - 1);
  double zPredict[N];
  generateMixedGaussians2D(xyInfSup, theta, K, N, zPredict);
  double crossE = crossEntropy(zObs, zPredict, N);
  // printf("EMLoop BIC=%f, AIC=%f, AICc=%f crossEntropy=%f\n", BIC, AIC, AICc, crossE );

  return logL;
}
