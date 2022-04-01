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

/// \file  ChebyshevFit1D.cxx
/// \brief Implementation of ChebyshevFit1D class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation

#include "ChebyshevFit1D.h"
#include "GPUCommonLogger.h"
#include <cmath>

using namespace GPUCA_NAMESPACE::gpu;

void ChebyshevFit1D::reset(int order, double xMin, double xMax)
{
  if (order < 0) {
    order = 0;
  }
  mN = order + 1;
  mXmin = xMin;
  mXscale = (xMax - xMin > 1.e-8) ? 2. / (xMax - mXmin) : 2.;

  mA.resize(mN * mN);
  mB.resize(mN + 1);
  mC.resize(mN + 1);
  mT.resize(mN + 1);
  reset();
}

void ChebyshevFit1D::reset()
{
  for (int i = 0; i <= mN; i++) {
    mB[i] = 0.;
    mC[i] = 0.;
    mT[i] = 0.;
  }

  for (int i = 0; i < mN * mN; i++) {
    mA[i] = 0.;
  }
  mM = 0;
}

void ChebyshevFit1D::print()
{
  LOG(info) << "";
  double* Ai = mA.data();
  for (int i = 0; i < mN; i++, Ai += mN) {
    for (int j = 0; j < mN; j++) {
      LOG(info) << Ai[j] << " ";
    }
    LOG(info) << " | " << mB[i];
  }
}

void ChebyshevFit1D::fit()
{
  for (int i = 0; i < mN; i++) {
    for (int j = 0; j < i; j++) {
      mA[i * mN + j] = mA[j * mN + i];
    }
  }
  //print();
  {
    double* Ai = mA.data();
    for (int i = 0; i < mN; i++, Ai += mN) {
      double a = Ai[i];
      if (fabs(a) < 1.e-6) {
        Ai[i] = 0;
        continue;
      }
      double* Aj = Ai + mN;
      for (int j = i + 1; j < mN; j++, Aj += mN) {
        double c = Aj[i] / a;
        for (int k = i + 1; k < mN; k++) {
          Aj[k] -= c * Ai[k];
        }
        mB[j] -= c * mB[i];
      }
      //print();
    }
  }
  {
    double* Ai = mA.data() + (mN - 1) * mN;
    for (int i = mN - 1; i >= 0; i--, Ai -= mN) {
      double s = mB[i];
      for (int k = i + 1; k < mN; k++) {
        s -= mC[k] * Ai[k];
      }
      mC[i] = (fabs(Ai[i]) > 1.e-6) ? (s / Ai[i]) : 0.;
    }
  }
  /*
  for (int i = 0; i < mN; i++) {
    LOG(info) << mC[i] << " ";
  }
  LOG(info) ;
  */
}

#endif
