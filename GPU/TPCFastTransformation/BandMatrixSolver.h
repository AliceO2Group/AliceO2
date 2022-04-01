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

/// \file  BandMatrixSolver.h
/// \brief Definition of BandMatrixSolver class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_BANDMATRIXSOLVER_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_BANDMATRIXSOLVER_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include <vector>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <limits>

namespace GPUCA_NAMESPACE
{
namespace gpu
{

/// Linear Equation Solver for a symmetric positive-definite band matrix A[n x n].
///
/// The matrix has a pattern of BandWidthT adjacent non-zero entries right to the diagonal in each row
/// Here is an example with n==10, BandWidthT==4.  (*) means non-zero element, (+) means symmetric element):
///     (****      )
///     (+****     )
///     (++****    )
///     (+++****   )
/// A = ( +++****  )
///     (  +++**** )
///     (   +++****)
///     (    +++***)
///     (     +++**)
///     (      +++*)
///
/// The non-zero matrix elements are stored in [n x BandWidthT] array mA
///
/// The equation to sove is A[n][n] x X[n][Bdim] = B[n][Bdim].
/// During calculations, the initial values of mA and mB get lost, so one can call solve() only once.
/// The solution X is stored in mB.
///
template <int BandWidthT>
class BandMatrixSolver
{
 public:
  /// Consructor
  BandMatrixSolver(int N, int Bdim) : mN(N), mBdim(Bdim)
  {
    assert(N > 0 && Bdim > 0);
    mA.resize(mN * BandWidthT, 0.);
    mB.resize(mN * mBdim, 0.);
  }

  /// debug tool: init arrays with NaN's
  void initWithNaN()
  {
    // Assign NaN's to ensure that uninitialized elements (for the matrix type 1) are not used in calculations.
    mA.assign(mA.size(), std::numeric_limits<double>::signaling_NaN());
    mB.assign(mB.size(), std::numeric_limits<double>::signaling_NaN());
  }

  /// access to A elements
  double& A(int i, int j)
  {
    auto ij = std::minmax(i, j);
    assert(ij.first >= 0 && ij.second < mN);
    int k = ij.second - ij.first;
    assert(k < BandWidthT);
    return mA[ij.first * BandWidthT + k];
  }

  /// access to B elements
  double& B(int i, int j)
  {
    assert(i >= 0 && i < mN && j >= 0 && j < mBdim);
    return mB[i * mBdim + j];
  }

  /// solve the equation
  void solve();

  /// solve an equation of a special type
  void solveType1();

  /// Test the class functionality. Returns 1 when ok, 0 when not ok
  static int test(bool prn = 0)
  {
    return BandMatrixSolver<0>::test(prn);
  }

 private:
  template <int nRows>
  void triangulateBlock(double AA[], double bb[]);

  template <int nCols>
  void dioganalizeBlock(double A[], double b[]);

 private:
  int mN = 0;
  int mBdim = 0;
  std::vector<double> mA;
  std::vector<double> mB;

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(BandMatrixSolver, 0);
#endif
};

template <>
int BandMatrixSolver<0>::test(bool prn);

template <int BandWidthT>
template <int nRows>
inline void BandMatrixSolver<BandWidthT>::triangulateBlock(double AA[], double bb[])
{
  {
    int m = BandWidthT;
    double* A = AA;
    for (int rows = 0; rows < nRows; rows++) {
      double c = 1. / A[0];
      A[0] = c; // store 1/a[0][0]
      double* rowi = A + BandWidthT - 1;
      for (int i = 1; i < m; i++) { // row 0+i
        double ai = c * A[i];       // A[0][i]
        for (int j = i; j < m; j++) {
          rowi[j] -= ai * A[j]; // A[i][j] -= A[0][j]/A[0][0]*A[i][0]
        }
        A[i] = ai; // A[0][i] /= A[0][0]
        rowi += BandWidthT - 1;
      }
      m--;
      A += BandWidthT;
    }
  }

  for (int k = 0; k < mBdim; k++) {
    int m = BandWidthT;
    double* A = AA;
    double* b = bb;
    for (int rows = 0; rows < nRows; rows++) {
      double bk = b[k];
      for (int i = 1; i < m; i++) {
        b[mBdim * i + k] -= A[i] * bk;
      }
      b[k] *= A[0];
      m--;
      A += BandWidthT;
      b += mBdim;
    }
  }
}

template <int BandWidthT>
template <int nCols>
inline void BandMatrixSolver<BandWidthT>::dioganalizeBlock(double AA[], double bb[])
{
  for (int k = 0; k < mBdim; k++) {
    int rows = BandWidthT;
    double* A = AA;
    double* b = bb;
    for (int col = 0; col < nCols; col++) {
      double bk = b[k];
      for (int i = 1; i < rows; i++) {
        b[-i * mBdim + k] -= A[BandWidthT * (-i) + i] * bk;
      }
      A -= BandWidthT;
      b -= mBdim;
      rows--;
    }
  }
}

template <int BandWidthT>
inline void BandMatrixSolver<BandWidthT>::solve()
{
  /// Solution slover

  const int stepA = BandWidthT;
  const int stepB = mBdim;
  // Upper Triangulization
  {
    int k = 0;
    double* Ak = &mA[0];
    double* bk = &mB[0];
    for (; k < mN - BandWidthT; k += 1, Ak += stepA, bk += stepB) { // for each row k
      triangulateBlock<1>(Ak, bk);
    }
    // last m rows
    triangulateBlock<BandWidthT>(Ak, bk);
  }

  // Diagonalization
  {
    int k = mN - 1;
    double* Ak = &mA[BandWidthT * k];
    double* bk = &mB[mBdim * k];
    for (; k > BandWidthT - 1; k -= 1, Ak -= stepA, bk -= stepB) { // for each row k
      dioganalizeBlock<1>(Ak, bk);
    }
    // first m rows
    dioganalizeBlock<BandWidthT>(Ak, bk);
  }
}

template <int BandWidthT>
inline void BandMatrixSolver<BandWidthT>::solveType1()
{
  /// A special solver for a band matrix were every second row has 0 at the end of the band.
  /// An example with n==10, BandWidthT==4:
  ///
  ///     (****      )
  ///     (+***0     )
  ///     (++****    )
  ///     (+++***0   )
  /// A = ( 0++****  )
  ///     (  +++***0 )
  ///     (   0++****)
  ///     (    +++***)
  ///     (     0++**)
  ///     (      +++*)
  ///

  const int stepA = 2 * BandWidthT;
  const int stepB = 2 * mBdim;
  // Upper Triangulization
  {
    int k = 0;
    double* Ak = &mA[0];
    double* bk = &mB[0];
    for (; k < mN - BandWidthT; k += 2, Ak += stepA, bk += stepB) { // for each row k
      triangulateBlock<2>(Ak, bk);
    }
    // last m rows
    triangulateBlock<BandWidthT>(Ak, bk);
  }

  // Diagonalization
  {
    int k = mN - 1;
    double* Ak = &mA[BandWidthT * k];
    double* bk = &mB[mBdim * k];
    for (; k > BandWidthT - 1; k -= 2, Ak -= stepA, bk -= stepB) { // for each row k
      dioganalizeBlock<2>(Ak, bk);
    }
    // first m rows
    dioganalizeBlock<BandWidthT>(Ak, bk);
  }
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
