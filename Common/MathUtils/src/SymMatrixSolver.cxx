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

/// \file  SymMatrixSolver.cxx
/// \brief Implementation of SymMatrixSolver class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>
/// RS Cloned from the GPU/TPCFastTransformation, consider simply moving here

#include "MathUtils/SymMatrixSolver.h"
#include "Framework/Logger.h"

#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>

using namespace o2::math_utils;

void SymMatrixSolver::solve()
{
  // Upper Triangulization
  for (int i = 0; i < mN; i++) {
    double* rowI = &mA[i * mShift];
    double* rowIb = &mA[i * mShift + mN];
    double c = (std::fabs(rowI[i]) > 1.e-10) ? 1. / rowI[i] : 0.;
    double* rowJ = rowI + mShift;
    for (int j = i + 1; j < mN; j++, rowJ += mShift) { // row j
      if (rowI[j] != 0.) {
        double aij = c * rowI[j]; // A[i][j] / A[i][i]
        for (int k = j; k < mShift; k++) {
          rowJ[k] -= aij * rowI[k]; // A[j][k] -= A[i][k]/A[i][i]*A[j][i]
        }
        rowI[j] = aij; // A[i][j] /= A[i][i]
      }
    }
    for (int k = 0; k < mM; k++) {
      rowIb[k] *= c;
    }
  }
  // Diagonalization
  for (int i = mN - 1; i >= 0; i--) {
    double* rowIb = &mA[i * mShift + mN];
    double* rowJb = rowIb - mShift;
    for (int j = i - 1; j >= 0; j--, rowJb -= mShift) { // row j
      double aji = mA[j * mShift + i];
      if (aji != 0.) {
        for (int k = 0; k < mM; k++) {
          rowJb[k] -= aji * rowIb[k];
        }
      }
    }
  }
}

void SymMatrixSolver::print()
{
  for (int i = 0; i < mN; i++) {
    LOG(info) << "";
    for (int j = 0; j < mN; j++) {
      LOG(info) << std::fixed << std::setw(5) << std::setprecision(2) << A(i, j) << " ";
    }
    LOG(info) << " | ";
    for (int j = 0; j < mM; j++) {
      LOG(info) << std::fixed << std::setw(5) << std::setprecision(2) << B(i, j) << " ";
    }
  }
  LOG(info) << std::setprecision(-1);
}

int SymMatrixSolver::test(bool prn)
{
  constexpr int n = 30;
  constexpr int d = 3;

  // std::random_device rd;  // Will be used to obtain a seed for the random
  std::mt19937 gen(1); // Standard mersenne_twister_engine seeded with 1
  std::uniform_real_distribution<> uniform(-.999, .999);

  double maxDiff = 0.;
  int nTries = 10000;

  auto tmpTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(tmpTime - tmpTime);
  auto durationMult = duration;

  for (int iter = 0; iter < nTries; iter++) {

    double x[n][d];
    double A[n][n];
    {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
          x[i][j] = 1. * uniform(gen);
        }
      }
      for (int i = 0; i < n; i++) {
        A[i][i] = fabs(2. + uniform(gen));
      }
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          A[i][j] = A[i][i] * A[j][j] * uniform(gen);
          A[j][i] = A[i][j];
        }
      }
      for (int i = 0; i < n; i++) {
        A[i][i] = A[i][i] * A[i][i];
      }
      if (prn && iter == nTries - 1) {
        for (int i = 0; i < n; i++) {
          LOG(info) << "";
          for (int j = 0; j < n; j++) {
            LOG(info) << std::fixed << std::setw(5) << std::setprecision(2) << A[i][j] << " ";
          }
        }
        LOG(info) << "";
      }
    }
    double b[n][d];
    auto startMult = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < d; k++) {
        b[i][k] = 0.;
      }
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < d; k++) {
          b[i][k] += x[j][k] * A[i][j];
        }
      }
    }
    auto stopMult = std::chrono::high_resolution_clock::now();
    durationMult += std::chrono::duration_cast<std::chrono::nanoseconds>(stopMult - startMult);

    SymMatrixSolver sym(n, d);

    for (int i = 0; i < n; i++) {
      for (int k = 0; k < d; k++) {
        sym.B(i, k) = b[i][k];
      }
      for (int j = i; j < n; j++) {
        sym.A(i, j) = A[i][j];
      }
    }

    auto start = std::chrono::high_resolution_clock::now();
    sym.solve();
    auto stop = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    double diff = 0.;
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < d; k++) {
        double t = std::fabs(x[i][k] - sym.B(i, k));
        if (diff < t) {
          diff = t;
        }
      }
    }
    if (maxDiff < diff) {
      maxDiff = diff;
    }
    // LOG(info) << std::defaultfloat ;
    // LOG(info) << "\n\n max diff " <<diff << "\n";
  }

  int ok = (maxDiff < 1.e-7);

  if (prn || !ok) {
    LOG(info) << std::defaultfloat;
    LOG(info) << "\n\n Overall max diff " << maxDiff << "\n";
    LOG(info) << " time " << duration.count() / nTries;
    LOG(info) << " time multiplication " << durationMult.count() / nTries;
  }
  return ok;
}
