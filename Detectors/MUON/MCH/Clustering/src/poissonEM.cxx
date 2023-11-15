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

namespace o2
{
namespace mch
{

extern ClusterConfig clusterConfig;

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
        // maskCij[nPads * i + j] * Cij[nPads * i + j] * qPixels[i];
        Cij[nPads * i + j] * qPixels[i];
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
        // s_i += maskCij[nPads * i + j] * Cij[nPads * i + j] * qPad[j] /
        s_i += Cij[nPads * i + j] * qPad[j] /
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
  // printf("fastIterateEMPoisson Cij=%p, Ci=%p, qPixels=%p, qPad=%p, qPadPrediction=%p, nPixels=%d, nPads=%d, newQPixels=%p\n",
  //        Cij, Ci, qPixels, qPad, qPadPrediction, nPixels, nPads,newQPixels);
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

std::pair<double, double> computeChiSquare(const Pads& pads, const double* qPredictedPads,
                                           int N)
{
  // Compute Chi2 on unsaturated pads
  double chi20 = 0.0;
  double chi21 = 0.0;
  const double* q = pads.getCharges();
  const Mask_t* cath = pads.getCathodes();
  const Mask_t* sat = pads.getSaturates();
  for (int i = 0; i < N; i++) {
    double var = (1 - sat[i]) * (q[i] - qPredictedPads[i]);
    if (cath[i] == 0) {
      chi20 += var * var;
    } else {
      chi21 += var * var;
    }
  }
  return std::make_pair(chi20, chi21);
}

std::pair<double, double> PoissonEMLoop(const Pads& pads, Pads& pixels,
                                        const double* Cij, Mask_t* maskCij,
                                        int qCutMode, double minPadError,
                                        int nItMax)
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
  if (clusterConfig.EMLocalMaxLog > clusterConfig.info) {
    printf("Poisson EM\n");
    printf(
      "   it.  <Pixels_residu>   <Pad_residu>   max(Pad_residu)   "
      "sum(Pad_residu)/sum(qPad)\n");
  }
  double meanPixelsResidu = 0.0;
  double maxPixelsResidu = 0.0;
  double maxRelResidu;
  double pixelVariation;
  double padRelError;
  //

  while (!converge) {
    //
    // Filter pixels
    //
    if (qCutMode == -1) {
      // Percent of the min charge
      // qPixCut = 1.02 * vectorMin(qPixels, nPixels);
      qPixCut = 1.0e-14;
    } else {
      // qCutMode = 0
      // No filtering
      qPixCut = 0.0;
    }

    // Disable pixels
    if (qPixCut > 0.0) {
      for (int i = 0; i < (nPixels); i++) {
        if (qPixels[i] < qPixCut) {
          // old version with mask vectorSetShort(&maskCij[nPads * i], 0, nPads);
        }
      }
    }
    // Update Ci
    for (int i = 0; i < nPixels; i++) {
      Ci[i] = 0;
      int start = nPads * i;
      int end = start + nPads;
      // for (int j = 0; j < nPads; j++) {
      for (int l = start; l < end; l++) {
        // Ci[i] += Cij[nPads * i + j] * maskCij[nPads * i + j];
        // Ci[i] += Cij[nPads * i + j];
        Ci[i] += Cij[l];
      }
    }
    // Not needed
    /*
    double Cj[nPads];
    for (int j = 0; j < nPads; j++) {
      Cj[j] = 0;
      for (int i = 0; i < nPixels; i++) {
        // Cj[j] += Cij[nPads * i + j] * maskCij[nPads * i + j];
        Cj[j] += Cij[nPads * i + j];
      }
    }
    */
    // Store previous qPixels state
    vectorCopy(qPixels, nPixels, previousQPixels);
    //
    //
    // Poisson EM Iterations
    //

    // iterateEMPoisson( Cij, Ci, maskCij, qPixels, qPads, qPadPrediction,
    // nPixels, nPads, qPixels);

    fastIterateEMPoisson(Cij, Ci, previousQPixels, qPads, qPadPrediction, nPixels,
                         nPads, qPixels);

    // Measure of pixel variation to stop
    // the iteration if required
    double pixResidu[nPixels];
    vectorAddVector(previousQPixels, -1.0, qPixels, nPixels, pixResidu);
    vectorAbs(pixResidu, nPixels, pixResidu);
    // Pixel variation
    pixelVariation = vectorSum(pixResidu, nPixels) / vectorSum(qPixels, nPixels);
    int iMaxResidu = vectorArgMax(pixResidu, nPixels);
    maxRelResidu = pixResidu[iMaxResidu] / previousQPixels[iMaxResidu];

    // Relative error on pad prediction
    // Compute a stdError normalized with the pad Esperance
    double padResidu[nPads];
    vectorAddVector(qPads, -1.0, qPadPrediction, nPads, padResidu);
    double var = vectorDotProd(padResidu, padResidu, nPads) / nPads;
    double E = vectorSum(qPads, nPads) / nPads;
    padRelError = std::sqrt(var) / E;

    if (clusterConfig.EMLocalMaxLog > clusterConfig.info) {
      printf("    EM it=%d   <pixelResidu>=%10.6f, dQPixel/qPixel=%10.6f, max(dQPix)/qPix=%10.6f, relPadError=%10.6f\n",
             it, vectorSum(pixResidu, nPixels) / nPixels, pixelVariation, maxRelResidu, padRelError);
    }
    // maxPixelVariation = 1 / 20 * minPadError;
    converge = (pixelVariation < minPadError * 0.03) && (padRelError < minPadError) ||
               (it > nItMax);
    it += 1;
  }
  if (clusterConfig.EMLocalMaxLog > clusterConfig.info) {
    printf("  Exit criterom pixelVariation=%d padRelError=%d itend=%d \n", (pixelVariation < minPadError * 0.03), (padRelError < minPadError), (it > nItMax));
  }
  // Update pixels charge
  // Remove small charged pixels (<qPixCut)
  int oldValueNPads = pixels.getNbrOfPads();
  pixels.setCharges(qPixels, nPixels);
  int k = 0;
  if (qPixCut > 0.0) {
    k = pixels.removePads(qPixCut);
  }
  // Chi2 on cathodes 0/1
  double chi20, chi21;
  // Chi2 on cathode1
  std::pair<double, double> chi = computeChiSquare(pads, qPadPrediction, pads.getNbrOfPads());
  std::pair<double, double> chiObs = computeChiSquare(pads, qPadPrediction, pads.getNbrOfObsPads());
  if (clusterConfig.EMLocalMaxLog > clusterConfig.info) {
    printf(" ??? Chi2 over NbrPads  = (%f, %f); Chi2 over NbrObsPads = (%f, %f) \n",
           chi.first, chi.second, chiObs.first, chiObs.second);
  }
  // ??? Must chose a method. A the moment over pads is better
  if (1) {
    // Chi2 over Pads
    chi20 = chi.first;
    chi21 = chi.second;
  } else {
    // Chi2 over ObsPads
    chi20 = chiObs.first;
    chi21 = chiObs.second;
  }
  // Take care to the leadind dimension is getNbrOfPads()
  if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
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
