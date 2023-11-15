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

#ifndef O2_MCH_MATHIESONFIT_H
#define O2_MCH_MATHIESONFIT_H

#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_vector.h>

#include "MCHClustering/PadsPEM.h"
#include "mathieson.h"
#include "mathUtil.h"

namespace o2
{
namespace mch
{
typedef struct dataFit {
  int N;
  int K;
  const double* xInf_ptr;
  const double* xSup_ptr;
  const double* yInf_ptr;
  const double* ySup_ptr;
  const Mask_t* cath_ptr;
  const double* zObs_ptr;
  Mask_t* notSaturated_ptr;
  double* cathWeights_ptr;
  double* cathMax_ptr;
  int chamberId;
  double* zCathTotalCharge_ptr;
  int verbose;
  double* thetaInit;        // Only used by InspectModel
  double* cathCoefNorm_ptr; // Used to keep the normalization of the 2 cathodes
  int dimOfParameters;      // default is 3 dimensions (x, y, w), 2 is for (x/y, w) fits
  int axe;                  // -1 for both axes, 0 for x axis, 1 for y axis
  CompressedPads_t* compressedPads;
} funcDescription_t;

void fitMathieson(const Pads& iPads, double* thetaInit, int kInit,
                  int dimOfParameters, int axe, int mode,
                  double* thetaFinal, double* khi2, double* pError);

void printState(int iter, gsl_multifit_fdfsolver* s, int K);
// Notes :
//  - the intitialization of Mathieson module must be done before
//  (initMathieson)
} // namespace mch
} // namespace o2

extern "C" {
void fitMathieson0(double* muAndWi, double* xyAndDxy, double* z, o2::mch::Mask_t* cath,
                   o2::mch::Mask_t* notSaturated, double* zCathTotalCharge, int K, int N,
                   int chamberId, int jacobian, double* muAndWf, double* khi2,
                   double* pError);

void fitMathieson(const double* x, const double* y, const double* dx, const double* dy, const double* q,
                  const o2::mch::Mask_t* cath, const o2::mch::Mask_t* sat, int chId, int nPads,
                  double* thetaInit, int kInit,
                  double* thetaFinal, double* khi2, double* pError);

int f_ChargeIntegral(const gsl_vector* gslParams, void* data,
                     gsl_vector* residual);
}

#endif // O2_MCH_MATHIESONFIT_H
