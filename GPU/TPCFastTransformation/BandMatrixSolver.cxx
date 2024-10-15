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

/// \file  BandMatrixSolver.cxx
/// \brief Implementation of BandMatrixSolver class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "BandMatrixSolver.h"
#include "GPUCommonLogger.h"
#include <random>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

templateClassImp(GPUCA_NAMESPACE::gpu::BandMatrixSolver);

template <>
int32_t BandMatrixSolver<0>::test(bool prn)
{
  constexpr int32_t n = 30;
  constexpr int32_t m = 6;
  constexpr int32_t d = 3;

  // std::random_device rd;  // Will be used to obtain a seed for the random
  std::mt19937 gen(1); // Standard mersenne_twister_engine seeded with 1
  std::uniform_real_distribution<> uniform(-.999, .999);

  double maxDiff = 0.;
  double maxDiffType1 = 0.;
  int32_t nTries = 10000;

  auto tmpTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(tmpTime - tmpTime);
  auto durationMult = duration;

  for (int32_t iter = 0; iter < nTries; iter++) {

    double x[n][d];
    double A[n][n];
    double Atype1[n][n];

    {
      for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < d; j++) {
          x[i][j] = 1. * uniform(gen);
        }
      }
      for (int32_t i = 0; i < n; i++) {
        A[i][i] = fabs(2. + uniform(gen));
      }
      for (int32_t i = 0; i < n; i++) {
        for (int32_t j = i + 1; j < n; j++) {
          if (j < i + m) {
            A[i][j] = A[i][i] * A[j][j] * uniform(gen);
          } else {
            A[i][j] = 0;
          }
          A[j][i] = A[i][j];
        }
      }
      for (int32_t i = 0; i < n; i++) {
        A[i][i] = A[i][i] * A[i][i];
      }
    }

    for (int32_t i = 0; i < n; i++) {
      int32_t oddRow = ((i % 2) != 0);
      for (int32_t j = i; j < n; j++) {
        if (j < i + m - oddRow) {
          Atype1[i][j] = A[i][j];
        } else {
          Atype1[i][j] = 0;
        }
        Atype1[j][i] = Atype1[i][j];
      }
    }

    if (prn && iter == nTries - 1) {
      LOG(info) << "Matrix A:";
      for (int32_t i = 0; i < n; i++) {
        LOG(info) << "";
        for (int32_t j = 0; j < n; j++) {
          LOG(info) << std::fixed << std::setw(5) << std::setprecision(2) << A[i][j] << " ";
        }
      }
      LOG(info) << "";
      LOG(info) << "\nMatrix A type 1:";
      for (int32_t i = 0; i < n; i++) {
        LOG(info) << "";
        for (int32_t j = 0; j < n; j++) {
          LOG(info) << std::fixed << std::setw(5) << std::setprecision(2) << Atype1[i][j] << " ";
        }
      }
      LOG(info) << "";
    }

    double B[n][d];
    for (int32_t i = 0; i < n; i++) {
      for (int32_t k = 0; k < d; k++) {
        B[i][k] = 0.;
      }
      for (int32_t j = 0; j < n; j++) {
        for (int32_t k = 0; k < d; k++) {
          B[i][k] += x[j][k] * A[i][j];
        }
      }
    }

    double Btype1[n][d];
    auto startMult = std::chrono::high_resolution_clock::now();
    for (int32_t i = 0; i < n; i++) {
      for (int32_t k = 0; k < d; k++) {
        Btype1[i][k] = 0.;
      }
      for (int32_t j = 0; j < n; j++) {
        for (int32_t k = 0; k < d; k++) {
          Btype1[i][k] += x[j][k] * Atype1[i][j];
        }
      }
    }
    auto stopMult = std::chrono::high_resolution_clock::now();
    durationMult += std::chrono::duration_cast<std::chrono::nanoseconds>(stopMult - startMult);

    BandMatrixSolver<m> band(n, d);
    BandMatrixSolver<m> bandType1(n, d);

    band.initWithNaN();
    bandType1.initWithNaN();

    for (int32_t i = 0; i < n; i++) {
      for (int32_t k = 0; k < d; k++) {
        band.B(i, k) = B[i][k];
        bandType1.B(i, k) = Btype1[i][k];
      }
      int32_t oddRow = ((i % 2) != 0);
      for (int32_t j = 0; j < m; j++) {
        if (i + j < n && j < m) {
          band.A(i, i + j) = A[i][i + j];
        }
        if (i + j < n && j < m - oddRow) {
          bandType1.A(i, i + j) = Atype1[i][i + j];
        }
      }
    }

    band.solve();
    auto start = std::chrono::high_resolution_clock::now();
    bandType1.solveType1();
    auto stop = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    for (int32_t i = 0; i < n; i++) {
      for (int32_t k = 0; k < d; k++) {
        double t = fabs(x[i][k] - band.B(i, k));
        double t1 = fabs(x[i][k] - bandType1.B(i, k));
        if (!std::isfinite(t) || maxDiff < t) {
          maxDiff = t;
        }
        if (!std::isfinite(t1) || maxDiffType1 < t1) {
          maxDiffType1 = t1;
        }
      }
    }
  }

  int32_t ok = (maxDiff < 1.e-6);

  if (prn || !ok) {
    LOG(info) << std::defaultfloat;
    LOG(info) << "\n\n Band matrix. Overall max diff: " << maxDiff << "\n";
  }

  int32_t ok1 = (maxDiffType1 < 1.e-6);

  if (prn || !ok1) {
    LOG(info) << std::defaultfloat;
    LOG(info) << "\n\n Band matrix of Type 1. Overall max diff: " << maxDiffType1 << "\n";
    LOG(info) << " time " << duration.count() / nTries;
    LOG(info) << " time multiplication " << durationMult.count() / nTries << " ns";
  }

  return ok && ok1;
}

template class GPUCA_NAMESPACE::gpu::BandMatrixSolver<0>;
