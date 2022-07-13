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

#include <cfloat>
#include <cmath>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "MCHClustering/ClusterConfig.h"
#include "MCHClustering/PadsPEM.h"
#include "mathUtil.h"
#include "mathieson.h"

//
// TODO : Optimization,  generateMixedGaussians2D computed twice.
//

static int nIterMin = 10;
static int nIterMax = 400;

namespace o2
{
namespace mch
{

void iterateEMPoisson(const double* Cij, const double* Ci,
                      const Mask_t* maskCij, const double* qPixels,
                      const double* qPad, double* qPadPrediction, int nPixels,
                      int nPads, double* newQPixels)
{

  double residu = 0.0;

  // Compute charge prediction on pad j based on pixel charges
  //  qPadPrediction[j] = Sum_i{ Cij[i,j].qPixels[i] }
  for (int j = 0; j < nPads; j++) {
    qPadPrediction[j] = 0;
    for (int i = 0; i < nPixels; i++) {
      qPadPrediction[j] +=
        maskCij[nPads * i + j] * Cij[nPads * i + j] * qPixels[i];
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
  // qPixels[i] = qPixels[i] / Ci[i] * Sum_j { Cij[i,j]*qPad[j] /
  // qPadPrediction[j] }
  for (int i = 0; i < nPixels; i++) {
    // Normalization term
    // r = np.sum( Cij[:,j]*qPad[0:nPads] / qPadPrediction[0:nPads] )
    if (Ci[i] > 1.0e-10) {
      double s_i = 0;
      for (int j = 0; j < nPads; j++) {
        s_i += maskCij[nPads * i + j] * Cij[nPads * i + j] * qPad[j] /
               qPadPrediction[j];
      }
      newQPixels[i] = s_i * qPixels[i] / Ci[i];
    } else {
      newQPixels[i] = 0;
    }
  }
}

void fastIterateEMPoissonV0(const double* Cij, const double* Ci,
                            const double* qPixels, const double* qPad,
                            double* qPadPrediction, int nPixels, int nPads,
                            double* newQPixels)
{

  double residu = 0.0;
  double* qRatio = new double[nPads];

  // Compute charge prediction on pad j based on pixel charges
  //  qPadPrediction[j] = Sum_i{ Cij[i,j].qPixels[i] }
  for (int j = 0; j < nPads; j++) {
    qPadPrediction[j] = 0;
    for (int i = 0; i < nPixels; i++) {
      qPadPrediction[j] += Cij[nPads * i + j] * qPixels[i];
    }
    // Prevent  zero division

    if (qPadPrediction[j] < 1.0e-10) {
      qPadPrediction[j] = qPadPrediction[j] + 1.0e-10;
    }
    /*
      if (qPad[j] < 1.0e-6) {
         qPadPrediction[j] = 1.0;
      } else {
        qPadPrediction[j] = 0.1 * qPad[j];
      }
    }
    */
    qRatio[j] = qPad[j] / qPadPrediction[j];
    residu += fabs(qPadPrediction[j] - qPad[j]);
  }

  // Update the pixel charge qPixels with the
  // the new predicted charges qPadPrediction
  // qPixels[i] = qPixels[i] / Ci[i] * Sum_j { Cij[i,j]*qPad[j] /
  // qPadPrediction[j] }
  for (int i = 0; i < nPixels; i++) {
    // Normalization term
    // r = np.sum( Cij[:,j]*qPad[0:nPads] / qPadPrediction[0:nPads] )
    if (Ci[i] > 1.0e-10) {
      double s_i = 0;
      for (int j = 0; j < nPads; j++) {
        s_i += Cij[nPads * i + j] * qRatio[j];
      }
      newQPixels[i] = s_i * qPixels[i] / Ci[i];
    } else {
      newQPixels[i] = 0;
    }
  }
  delete[] qRatio;
}

void fastIterateEMPoisson(const double* Cij, const double* Ci,
                          const double* qPixels, const double* qPad,
                          double* qPadPrediction, int nPixels, int nPads,
                          double* newQPixels)
{

  double residu = 0.0;
  double* qRatio = new double[nPads];

  // Compute charge prediction on pad j based on pixel charges
  //  qPadPrediction[j] = Sum_i{ Cij[i,j].qPixels[i] }
  gsl_matrix_const_view Cij_gsl = gsl_matrix_const_view_array(Cij, nPixels, nPads);
  gsl_vector_const_view qPixels_gsl = gsl_vector_const_view_array(qPixels, nPixels);
  gsl_vector_view qPadPrediction_gsl = gsl_vector_view_array(qPadPrediction, nPads);

  gsl_blas_dgemv(CblasTrans, 1.0, &Cij_gsl.matrix, &qPixels_gsl.vector, 0.0, &qPadPrediction_gsl.vector);
  for (int j = 0; j < nPads; j++) {
    // Prevent division by zero
    if (qPadPrediction[j] < 1.0e-10) {
      qPadPrediction[j] = qPadPrediction[j] + 1.0e-10;
    }
    qRatio[j] = qPad[j] / qPadPrediction[j];
    residu += fabs(qPadPrediction[j] - qPad[j]);
  }

  // Update the pixel charge qPixels with the
  // the new predicted charges qPadPrediction
  // qPixels[i] = qPixels[i] / Ci[i] * Sum_j { Cij[i,j]*qPad[j] /
  // qPadPrediction[j] }
  gsl_vector_view qRatio_gsl = gsl_vector_view_array(qRatio, nPads);
  gsl_vector_view newQPixels_gsl = gsl_vector_view_array(newQPixels, nPixels);
  if (1) {
    gsl_blas_dgemv(CblasNoTrans, 1.0, &Cij_gsl.matrix, &qRatio_gsl.vector, 0.0, &newQPixels_gsl.vector);

    for (int i = 0; i < nPixels; i++) {
      if (Ci[i] > 1.0e-10) {
        newQPixels[i] = newQPixels[i] * qPixels[i] / Ci[i];
      } else {
        newQPixels[i] = 0;
      }
    }
  } else if (0) {
    for (int i = 0; i < nPixels; i++) {
      // Normalization term
      // r = np.sum( Cij[:,j]*qPad[0:nPads] / qPadPrediction[0:nPads] )
      double s_i = 0;
      newQPixels[i] = 0.;
      for (int j = 0; j < nPads; j++) {
        s_i = s_i + Cij[nPads * i + j] * qRatio[j];
      }
      if (Ci[i] > 1.0e-10) {
        newQPixels[i] = s_i * qPixels[i] / Ci[i];
      } else {
        newQPixels[i] = 0;
      }
    }
  } else {
    for (int i = 0; i < nPixels; i++) {
      // Normalization term
      // r = np.sum( Cij[:,j]*qPad[0:nPads] / qPadPrediction[0:nPads] )
      double s_i = 0;
      // newQPixels[i] = 0.;

      for (int j = 0; j < nPads; j++) {
        s_i = s_i + Cij[nPads * i + j] * qRatio[j];
      }
      if (Ci[i] > 1.0e-10) {
        newQPixels[i] = s_i * qPixels[i] / Ci[i];
      } else {
        newQPixels[i] = 0;
      }

      /*
      double s_i = 0;
      if (Ci[i] > 1.0e-10) {
        for (int j = 0; j < nPads; j++) {
          s_i += Cij[nPads * i + j] * qRatio[j];
        }
        newQPixels[i] = s_i * qPixels[i] / Ci[i];
      } else {
        newQPixels[i] = 0;
      }
       */
    }
  }

  delete[] qRatio;
}

double computeChiSquare(const Pads& pads, const double* qPredictedPads,
                        int iStart, int iEnd)
{
  // Compute Chi2 on unsaturated pads
  double chi2 = 0.0;
  const double* q = pads.getCharges();
  const Mask_t* sat = pads.getSaturates();
  for (int i = iStart; i < iEnd; i++) {
    double var = (1 - sat[i]) * (q[i] - qPredictedPads[i]);
    chi2 += var * var;
  }
  return chi2;
}

std::pair<double, double> PoissonEMLoop(const Pads& pads, Pads& pixels,
                                        const double* Cij, Mask_t* maskCij,
                                        int qCutMode, double minPadResidu,
                                        int nItMax, int n0)
{
  // The array pixels return the last state
  //
  // Init.
  //
  int nPads = pads.getNbrOfPads();
  int nPixels = pixels.getNbrOfPads();
  //
  const double* x = pixels.getX();
  const double* y = pixels.getY();
  const double* dx = pixels.getDX();
  const double* dy = pixels.getDY();
  double* qPixels = new double[pixels.getNbrOfPads()];
  vectorCopy(pixels.getCharges(), pixels.getNbrOfPads(), qPixels);
  //
  const double* qPads = pads.getCharges();
  //
  // MaskCij: Used to disable Cij contribution (disable pixels)
  vectorSetShort(maskCij, 1, nPads * nPixels);
  // Ci, Cj: sum of Cij by row or by column
  double Ci[nPixels];
  double qPadPrediction[nPads];
  double previousQPixels[nPixels];
  double qPixCut;
  // Init convergence criteria
  bool converge = false;
  int it = 0;
  if (ClusterConfig::EMLocalMaxLog > ClusterConfig::detail) {
    printf("Poisson EM\n");
    printf(
      "   it.  <Pixels_residu>   <Pad_residu>   max(Pad_residu)   "
      "sum(Pad_residu)/sum(qPad)\n");
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
    } else {
      // qCutMode = 0
      // No filtering
      qPixCut = 0.0;
    }

    // Disable pixels
    if (qPixCut > 0.0) {
      for (int i = 0; i < (nPixels); i++) {
        if (qPixels[i] < qPixCut) {
          vectorSetShort(&maskCij[nPads * i], 0, nPads);
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
    double Cj[nPads];
    for (int j = 0; j < nPads; j++) {
      Cj[j] = 0;
      for (int i = 0; i < nPixels; i++) {
        Cj[j] += Cij[nPads * i + j] * maskCij[nPads * i + j];
      }
    }
    // Store previous qPixels state
    vectorCopy(qPixels, nPixels, previousQPixels);
    //
    //
    // Poisson EM Iterations
    //

    // Convergence acceleration process
    // Not used
    /*
    if(0) {
    double qPixels1[nPixels], qPixels2[nPixels];
    // Speed-up factors
    double r[nPixels], v[nPixels];
    // Perform 2 iterations
    // Test simple iteration if(1) {
    iterateEMPoisson( Cij, Ci, maskCij, qPixels, qPads, qPadPrediction, nPixels,
    nPads, qPixels1); iterateEMPoisson( Cij, Ci, maskCij, qPixels1, qPads,
    qPadPrediction, nPixels, nPads, qPixels2);
    // ??? To optimize : loop fusion
    // Compute r[:] = (qPixels1[:] - qPixels[:])
    vectorAddVector( qPixels1, -1.0, qPixels,nPixels, r );
    // Compute v[:] = (qPixels2[:] - qPixels[:]) - r[:]
    vectorAddVector( qPixels2, -1.0, qPixels1, nPixels, v );
    vectorAddVector( v, -1.0, r, nPixels, v );
    double rNorm = vectorNorm(r, nPixels);
    double vNorm = vectorNorm(v, nPixels);
    // printf("rNorm=%f vNorm=%f\n", rNorm, vNorm);
    if (( rNorm < 1.0e-12 ) || (vNorm < 1.0e-12 )) {
      converge = true;
    } else {
      double alpha = - rNorm / vNorm;
      // qPixels[:] = qPixels[:] - 2.0*alpha*r[:] + alpha*alpha*v[:]
      vectorAddVector( qPixels, -2.0*alpha, r, nPixels, qPixels);
      vectorAddVector( qPixels, alpha*alpha, v, nPixels, qPixels);
      iterateEMPoisson( Cij, Ci, maskCij, qPixels, qPads, qPadPrediction,
    nPixels, nPads, qPixels);
    }

    } else {
    */
    // iterateEMPoisson( Cij, Ci, maskCij, qPixels, qPads, qPadPrediction,
    // nPixels, nPads, qPixels);
    // fastIterateEMPoisson(Cij, Ci, qPixels, qPads, qPadPrediction, nPixels,
    //                     nPads, qPixels);
    fastIterateEMPoisson(Cij, Ci, previousQPixels, qPads, qPadPrediction, nPixels,
                         nPads, qPixels);
    // }

    // Compute pixel residues: pixResidu[:] = abs( previousQPixels[:] -
    // qPixels[:] )
    double pixResidu[nPixels];
    vectorAddVector(previousQPixels, -1.0, qPixels, nPixels, pixResidu);
    vectorAbs(pixResidu, nPixels, pixResidu);
    meanPixelsResidu = vectorSum(pixResidu, nPixels) / nPixels;
    maxPixelsResidu = vectorMax(pixResidu, nPixels);
    // Compute pad residues: padResidu[:] = abs( qPads - qPadPrediction[:] )
    double padResidu[nPads];
    vectorAddVector(qPads, -1.0, qPadPrediction, nPads, padResidu);
    vectorAbs(padResidu, nPads, padResidu);
    double meanPadResidu = vectorSum(padResidu, nPads) / nPads;
    if (ClusterConfig::EMLocalMaxLog > ClusterConfig::detail) {
      printf(" %4d    %10.6f      %10.6f      %10.6f             %10.6f\n", it,
             meanPixelsResidu, meanPadResidu, vectorMax(padResidu, nPads),
             vectorSum(padResidu, nPads) / vectorSum(qPads, nPads));
      int u = vectorArgMax(padResidu, nPads);
      // printf("max pad residu:    qPads=%10.6f, qPadPrediction=%10.6f \n",
      // qPads[u], qPadPrediction[u]);
    }
    converge = (meanPixelsResidu < 1.0e-12) || (meanPadResidu < minPadResidu) ||
               (it > nItMax);
    it += 1;
  }

  // Update pixels charge
  // Remove small charged pixels (<qPixCut)
  int oldValueNPads = pixels.getNbrOfPads();
  pixels.setCharges(qPixels, nPixels);
  int k = 0;
  if (qPixCut > 0.0) {
    k = pixels.removePads(qPixCut);
  }

  // Chi2 on cathode0
  double chi20 = computeChiSquare(pads, qPadPrediction, 0, n0);
  // Chi2 on cathode1
  double chi21 = computeChiSquare(pads, qPadPrediction, n0, nPads);

  // Take care to the leadind dimension is getNbrOfPads()
  if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) {
    printf("End poisson EM :\n");
    printf("  Total Pad Charge       = %14.6f\n", vectorSum(qPads, nPads));
    printf("  Total Predicted Charge = %14.6f\n",
           vectorSum(qPadPrediction, nPads));
    printf("  Total Pixel Charge     = %14.6f\n", vectorSum(qPixels, nPixels));
    printf("  EMPoisson qPixCut      = %14.6f\n", qPixCut);
    printf("  Chi20                   = %14.6f\n", chi20);
    printf("  Chi21                   = %14.6f\n", chi21);
    printf("  # of iterations        = %d\n", it);
    printf("  Max Pixel variation    = %14.6f\n", maxPixelsResidu);
    printf("  Nbr of removed pixels  = %d/%d\n", oldValueNPads - k,
           oldValueNPads);
  }
  delete[] qPixels;
  return std::make_pair(chi20, chi21);
}

} // namespace mch
} // namespace o2