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

/// \file  SymMatrixSolver.h
/// \brief Definition of SymMatrixSolver class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>
/// RS Cloned from the GPU/TPCFastTransformation, consider simply moving here

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SYMMATRIXSOLVER_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SYMMATRIXSOLVER_H

#include "GPUCommonRtypes.h"
#include <vector>
#include <cassert>
#include <algorithm>

namespace o2::math_utils
{

/// Linear Equation Solver for a symmetric positive-definite matrix A[n x n].
///
/// A[n x n] * X [n x m] = B[n x m]
///
/// A elements are stored in the upper triangle of A.
/// Thus A(i,j) and A(j,i) access the same element.
///
class SymMatrixSolver
{
 public:
  SymMatrixSolver(int N, int M) : mN(N), mM(M), mShift(mN + mM)
  {
    assert(N > 0 && M > 0);
    mA.resize(mN * mShift, 0.);
  }

  /// access to A elements
  double& A(int i, int j)
  {
    auto ij = std::minmax(i, j);
    assert(ij.first >= 0 && ij.second < mN);
    return mA[ij.first * mShift + ij.second];
  }

  /// access to B elements
  double& B(int i, int j)
  {
    assert(i >= 0 && i < mN && j >= 0 && j < mM);
    return mA[i * mShift + mN + j];
  }

  ///
  void solve();

  ///
  void print();

  /// Test the class functionality. Returns 1 when ok, 0 when not ok
  static int test(bool prn = 0);

 private:
 private:
  int mN = 0;
  int mM = 0;
  int mShift = 0;
  std::vector<double> mA;

  ClassDefNV(SymMatrixSolver, 0);
};

} // namespace o2::math_utils

#endif
