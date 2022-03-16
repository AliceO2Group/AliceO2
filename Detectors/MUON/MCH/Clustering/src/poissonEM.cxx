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

#include "MCHClustering/PadsPEM.h"
#include "MCHClustering/dataStructure.h"
#include "MCHClustering/mathUtil.h"
#include "MCHClustering/mathieson.h"

#define VERBOSE 0

//
// TODO : Optimization,  generateMixedGaussians2D computed twice.
//

static int nIterMin = 10;
static int nIterMax = 400;

namespace o2
{
namespace mch
{

void iterateEMPoisson(const double* Cij, const double* Ci, const Mask_t* maskCij, const double* qPixels, const double* qPad, double* qPadPrediction,
                      int nPixels, int nPads, double* newQPixels)
{

  double residu = 0.0;

  // Compute charge prediction on pad j based on pixel charges
  //  qPadPrediction[j] = Sum_i{ Cij[i,j].qPixels[i] }
  for (int j = 0; j < nPads; j++) {
    // ??? qPadPrediction[i] = np.dot( Cij[ i, 0:nPixels], qPixels[0:nPixels]*maskCij[i,:] )
    qPadPrediction[j] = 0;
    for (int i = 0; i < nPixels; i++) {
      qPadPrediction[j] += maskCij[nPads * i + j] * Cij[nPads * i + j] * qPixels[i];
    }
    // Prevent  zero division
    if (qPadPrediction[j] < 1.0e-6) {
      if (qPad[j] < 1.0e-6) {
        qPadPrediction[j] = 1.0;
      } else {
        qPadPrediction[j] = 0.1 * qPad[j];
      }
    }
    residu += fabs(qPadPrediction[j] - qPad[j]);
  }

  // Update the pixel charge qPixels with the
  // the new predicted charges qPadPrediction
  // qPixels[i] = qPixels[i] / Ci[i] * Sum_j { Cij[i,j]*qPad[j] / qPadPrediction[j] }
  for (int i = 0; i < nPixels; i++) {
    // Normalization term
    // r = np.sum( Cij[:,j]*qPad[0:nPads] / qPadPrediction[0:nPads] )
    if (Ci[i] > 1.0e-10) {
      double s_i = 0;
      for (int j = 0; j < nPads; j++) {
        s_i += maskCij[nPads * i + j] * Cij[nPads * i + j] * qPad[j] / qPadPrediction[j];
      }
      newQPixels[i] = s_i * qPixels[i] / Ci[i];
      // printf("new pixel i=%d s_i=%f, Ci[i]=%f\n", i, s_i,  Ci[i]);
    } else {
      newQPixels[i] = 0;
    }
  }
}

double computeChiSquare(const Pads& pads, const double* qPredictedPads)
{
  double chi2 = 0.0;
  for (int i = 0; i < pads.nPads; i++) {
    double var = (1 - pads.saturate[i]) * (pads.q[i] - qPredictedPads[i]);
    chi2 += var * var;
  }
  return chi2;
}

double PoissonEMLoop(const Pads& pads, Pads& pixels, double* Cij, Mask_t* maskCij,
                     int qCutMode, double minPadResidu, int nItMax, int verbose)
{
  // The array pixels return the last state
  //
  // Init.
  //
  // ??? To put in cst module
  double qPixMin = 0.5;
  int nPads = pads.nPads;
  int nPixels = pixels.nPads;

  //
  double* x = pixels.x;
  double* y = pixels.y;
  double* dx = pixels.dx;
  double* dy = pixels.dy;
  double* qPixels = pixels.q;
  //
  const double* qPads = pads.q;
  // Ci, Cj: sum of Cij by row or by column
  double Ci[nPixels];

  double qPadPrediction[nPads];
  double previousQPixels[nPixels];
  double qPixCut;
  // Init convergence criteria
  bool converge = false;
  int it = 0;
  if (VERBOSE > 1) {
    printf("Poisson EM\n");
    printf("   it.  <Pixels_residu>   <Pad_residu>   max(Pad_residu)   sum(Pad_residu)/sum(qPad)\n");
  }
  double meanPixelsResidu = 0.0;
  double maxPixelsResidu = 0.0;
  //
  while (!converge) {
    //
    // Filter pixels
    //
    if (qCutMode == -1) {
      // Percent of the min charge
      qPixCut = 1.02 * vectorMin(qPixels, nPixels);
    } else if (qCutMode == 0) {
      // No filtering
      qPixCut = 0.0;
    } else {
      // Default mode
      qPixCut = fmax(0.01 * vectorMax(qPixels, nPixels), qPixMin);
    }
    // ??? inv printf("  EMPoisson qPixCut      = %14.6f\n", qPixCut);

    // Disable pixels
    // vectorBuildMask ???
    for (int i = 0; i < (nPixels); i++) {
      if (qPixels[i] < qPixCut) {
        // ?? not optimal waste cpu time
        for (int j = 0; j < nPads; j++) {
          maskCij[nPads * i + j] = 0;
        }
      }
    }
    // Update Ci
    for (int i = 0; i < nPixels; i++) {
      Ci[i] = 0;
      for (int j = 0; j < nPads; j++) {
        Ci[i] += Cij[nPads * i + j] * maskCij[nPads * i + j];
      }
    }
    /* inv ???
    vectorPrint("Ci", Ci, nPixels);
    printf("Sum Ci= %f\n",vectorSum( Ci, nPixels));
    */
    double Cj[nPads];
    for (int j = 0; j < nPads; j++) {
      Cj[j] = 0;
      for (int i = 0; i < nPixels; i++) {
        Cj[j] += Cij[nPads * i + j] * maskCij[nPads * i + j];
      }
    }
    /* Inv ???
    vectorPrint("Cj", Cj, nPads);
    printf("Sum Cj= %f\n",vectorSum( Cj, nPads));
    */
    // Store previous qPixels
    vectorCopy(qPixels, nPixels, previousQPixels);
    //
    //
    // Poisson EM Iterations
    //
    // Convergence acceleration process
    // ??? doc
    if (0) {
      double qPixels1[nPixels], qPixels2[nPixels];
      // Speed-up factors
      double r[nPixels], v[nPixels];
      // Perform 2 iterations
      // Test simple iteration if(1) {
      iterateEMPoisson(Cij, Ci, maskCij, qPixels, qPads, qPadPrediction, nPixels, nPads, qPixels1);
      iterateEMPoisson(Cij, Ci, maskCij, qPixels1, qPads, qPadPrediction, nPixels, nPads, qPixels2);
      // ??? To optimize : loop fusion
      // Compute r[:] = (qPixels1[:] - qPixels[:])
      vectorAddVector(qPixels1, -1.0, qPixels, nPixels, r);
      // Compute v[:] = (qPixels2[:] - qPixels[:]) - r[:]
      vectorAddVector(qPixels2, -1.0, qPixels1, nPixels, v);
      vectorAddVector(v, -1.0, r, nPixels, v);
      double rNorm = vectorNorm(r, nPixels);
      double vNorm = vectorNorm(v, nPixels);
      // printf("rNorm=%f vNorm=%f\n", rNorm, vNorm);
      if ((rNorm < 1.0e-12) || (vNorm < 1.0e-12)) {
        converge = true;
      } else {
        double alpha = -rNorm / vNorm;
        // qPixels[:] = qPixels[:] - 2.0*alpha*r[:] + alpha*alpha*v[:]
        vectorAddVector(qPixels, -2.0 * alpha, r, nPixels, qPixels);
        vectorAddVector(qPixels, alpha * alpha, v, nPixels, qPixels);
        iterateEMPoisson(Cij, Ci, maskCij, qPixels, qPads, qPadPrediction, nPixels, nPads, qPixels);
      }

    } else {
      iterateEMPoisson(Cij, Ci, maskCij, qPixels, qPads, qPadPrediction, nPixels, nPads, qPixels);
      // printf("  Total Predicted Charge = %14.6f\n", vectorSum(qPadPrediction, nPads) );
    }

    // Compute pixel residues: pixResidu[:] = abs( previousQPixels[:] - qPixels[:] )
    double pixResidu[nPixels];
    vectorAddVector(previousQPixels, -1.0, qPixels, nPixels, pixResidu);
    vectorAbs(pixResidu, nPixels, pixResidu);
    meanPixelsResidu = vectorSum(pixResidu, nPixels) / nPixels;
    maxPixelsResidu = vectorMax(pixResidu, nPixels);
    if (meanPixelsResidu < 1.0e-12) {
      converge = true;
    }
    // Compute pad residues: padResidu[:] = abs( qPads - qPadPrediction[:] )
    double padResidu[nPads];
    vectorAddVector(qPads, -1.0, qPadPrediction, nPads, padResidu);
    vectorAbs(padResidu, nPads, padResidu);
    double meanPadResidu = vectorSum(padResidu, nPads) / nPads;
    if (VERBOSE > 1) {
      printf(" %4d    %10.6f      %10.6f      %10.6f             %10.6f\n",
             it, meanPixelsResidu, meanPadResidu, vectorMax(padResidu, nPads), vectorSum(padResidu, nPads) / vectorSum(qPads, nPads));
      int u = vectorArgMax(padResidu, nPads);
      // printf("max pad residu:    qPads=%10.6f, qPadPrediction=%10.6f \n", qPads[u], qPadPrediction[u]);
    }
    converge = converge || (meanPadResidu < minPadResidu) || (it > nItMax);
    it += 1;
  }

  // Update pixels
  // Remove small charged pixels (<qPixCut)
  int k = 0;
  for (int i = 0; i < nPixels; i++) {
    if (qPixels[i] > qPixCut) {
      // if ( qPixels[i] > -1 ) {
      qPixels[k] = qPixels[i];
      // printf("k=%d <- i=%d\n", k, i);
      x[k] = x[i];
      y[k] = y[i];
      dx[k] = dx[i];
      dy[k] = dy[i];
      k++;
    } // else {
      // printf("i=%d, qPixels[i]=%f = \n", i, qPixels[i]);

    // }
  }
  int oldValueNPads = pixels.nPads;
  pixels.nPads = k;
  double chi2 = computeChiSquare(pads, qPadPrediction);

  // Take care to the leadind dimension is nPads
  if (VERBOSE > 0) {
    printf("End poisson EM :\n");
    printf("  Total Pad Charge       = %14.6f\n", vectorSum(qPads, nPads));
    printf("  Total Predicted Charge = %14.6f\n", vectorSum(qPadPrediction, nPads));
    printf("  Total Pixel Charge     = %14.6f\n", vectorSum(qPixels, nPixels));
    printf("  EMPoisson qPixCut      = %14.6f\n", qPixCut);
    printf("  Chi2                   = %14.6f\n", chi2);
    printf("  # of iterations        = %d\n", it);
    printf("  Max Pixel variation    = %14.6f\n", maxPixelsResidu);

    printf("  Nbr of removed pixels  = %d/%d\n", oldValueNPads - k, oldValueNPads);
  }
  return chi2;
}

} // namespace mch
} // namespace o2