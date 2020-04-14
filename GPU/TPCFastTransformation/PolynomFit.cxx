// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  PolynomFit.cxx
/// \brief Implementation of PolynomFit class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "PolynomFit.h"
#include <cmath>

// #include "TMatrixDSym.h"
// #include "TMatrixD.h"
// #include "TVectorD.h"

using namespace GPUCA_NAMESPACE::gpu;

void PolynomFit::reset(int nCoefficients)
{
  if (nCoefficients >= 0) {
    mN = nCoefficients;
    delete[] mA;
    delete[] mAi;
    delete[] mB;
    mA = new long double[mN * (mN + 1) / 2];
    mAi = new long double[mN * (mN + 1) / 2];
    mB = new long double[mN];
  }

  for (int i = 0; i < mN; i++) {
    for (int j = 0; j <= i; j++) {
      mA[i * (i + 1) / 2 + j] = 0;
      mAi[i * (i + 1) / 2 + j] = 0;
    }
    mB[i] = 0;
  }
  mNmeasurements = 0;
}

int PolynomFit::invS(long double A[], int N)
{
  int ret = 0;

  const long double ZERO = 1.E-20;

  // input: simmetric > 0 NxN matrix A = {a11,a21,a22,a31..a33,..}
  // output: inverse A, in case of problems fill zero and return 1

  // A->low triangular Anew : A = Anew x Anew^T
  // method:
  // for(j=1,N) for(i=j,N) Aij=(Aii-sum_{k=1}^{j-1}Aik*Ajk )/Ajj
  //

  {
    long double *j1 = A, *jj = A;
    for (int j = 1; j <= N; j1 += j++, jj += j) {
      long double *ik = j1, x = 0;
      while (ik != jj) {
        x -= (*ik) * (*ik);
        ik++;
      }
      x += *ik;
      if (x > ZERO) {
        x = sqrt(x);
        *ik = x;
        ik++;
        x = 1 / x;
        for (int step = 1; step <= N - j; ik += ++step) { // ik==Ai1
          long double sum = 0;
          for (long double* jk = j1; jk != jj; sum += (*(jk++)) * (*(ik++)))
            ;
          *ik = (*ik - sum) * x; // ik == Aij
        }
      } else {
        long double* ji = jj;
        for (int i = j; i < N; i++)
          *(ji += i) = 0.;
        ret = -1;
      }
    }
  }

  // A -> Ainv
  // method :
  // for(i=1,N){
  //   Aii = 1/Aii;
  //   for(j=1,i-1) Aij=-(sum_{k=j}^{i-1} Aik * Akj) / Aii ;
  // }

  {
    long double *ii = A, *ij = A;
    for (int i = 1; i <= N; ij = ii + 1, ii += ++i) {
      if (*ii > ZERO) {
        long double x = -(*ii = 1. / *ii);
        {
          long double* jj = A;
          for (int j = 1; j < i; jj += ++j, ij++) {
            long double *ik = ij, *kj = jj, sum = 0.;
            for (int k = j; ik != ii; kj += k++, ik++) {
              sum += *ik * *kj;
            }
            *kj = sum * x;
          }
        }
      } else {
        for (long double* ik = ij; ik != ii + 1; ik++) {
          *ik = 0.;
        }
        ret = -1;
      }
    }
  }

  // A -> A^T x A
  // method:
  // Aij = sum_{k=i}^N Aki * Akj

  {
    long double *ii = A, *ij = A;
    for (int i = 1; i <= N; ii += ++i) {
      do {
        long double *ki = ii, *kj = ij, sum = 0.;
        for (int k = i; k <= N; ki += k, kj += k++)
          sum += (*ki) * (*kj);
        *ij = sum;
      } while ((ij++) != ii);
    }
  }
  return ret;
}
