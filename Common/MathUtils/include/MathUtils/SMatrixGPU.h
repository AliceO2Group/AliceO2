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

/// \file SMatrixGPU.h
/// \author Matteo Concas mconcas@cern.ch
/// \brief This is a close porting of the SMatrix and SVectorGPU ROOT interfaces.
/// Only parts strictly requiring STD library have been changed.
/// Also some utilities to have basic checks and printouts working on GPUs have been rewritten.
///
/// Notably only templated implementation of
/// row_offsets_utils::make and row_offsets_utils::do_make
/// has been reworked to support gpustd::array as backend.
///
/// Other than that, the author is not taking any credit on the methodologies implemented
/// which have been taken straight from root source code
///
/// Original sources are on the official website:
/// - https://root.cern.ch

#ifndef ALICEO2_SMATRIX_GPU_H
#define ALICEO2_SMATRIX_GPU_H

#include "GPUCommonDef.h"
#include "GPUCommonArray.h"
#include "GPUCommonMath.h"
#include "GPUCommonAlgorithm.h"

namespace o2::math_utils
{
template <bool>
struct Check {
  GPUd() Check(void*) {}
};
template <>
struct Check<false> {
};

#define GPU_STATIC_CHECK(expr, msg) \
  {                                 \
    class ERROR_##msg               \
    {                               \
    };                              \
    ERROR_##msg e;                  \
    (void)(Check<(expr) != 0>(&e)); \
  }

template <typename T, unsigned int N>
class SVectorGPU
{
 public:
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  GPUd() iterator begin();
  GPUd() iterator end();
  GPUd() const_iterator begin() const;
  GPUd() const_iterator end() const;
  GPUd() SVectorGPU();
  GPUd() SVectorGPU(const SVectorGPU<T, N>& rhs);

  GPUd() const T& operator[](unsigned int i) const;
  GPUd() const T& operator()(unsigned int i) const;
  GPUd() T& operator[](unsigned int i);
  GPUd() T& operator()(unsigned int i);
  GPUd() const T* Array() const;
  GPUd() T* Array();
  GPUd() T apply(unsigned int i) const;

  // Operators
  GPUd() SVectorGPU<T, N>& operator-=(const SVectorGPU<T, N>& rhs);
  GPUd() SVectorGPU<T, N>& operator+=(const SVectorGPU<T, N>& rhs);

  enum {
    kSize = N
  };

  GPUdi() static unsigned int Dim() { return N; }

 private:
  T mArray[N];
};

template <class T, unsigned int D>
GPUdi() T* SVectorGPU<T, D>::begin()
{
  return mArray;
}

template <class T, unsigned int D>
GPUdi() const T* SVectorGPU<T, D>::begin() const
{
  return mArray;
}

template <class T, unsigned int D>
GPUdi() T* SVectorGPU<T, D>::end()
{
  return mArray + Dim();
}

template <class T, unsigned int D>
GPUdi() const T* SVectorGPU<T, D>::end() const
{
  return mArray + Dim();
}
template <class T, unsigned int N>

GPUdi() const T& SVectorGPU<T, N>::operator[](unsigned int i) const
{
  return mArray[i];
}

template <class T, unsigned int N>
GPUdi() const T& SVectorGPU<T, N>::operator()(unsigned int i) const
{
  return mArray[i];
}

template <class T, unsigned int N>
GPUdi() T& SVectorGPU<T, N>::operator[](unsigned int i)
{
  return mArray[i];
}

template <class T, unsigned int N>
GPUdi() T& SVectorGPU<T, N>::operator()(unsigned int i)
{
  return mArray[i];
}

template <class T, unsigned int N>
GPUd() SVectorGPU<T, N>::SVectorGPU()
{
  for (unsigned int i = 0; i < N; ++i) {
    mArray[i] = 0;
  }
}

template <class T, unsigned int N>
GPUd() SVectorGPU<T, N>::SVectorGPU(const SVectorGPU<T, N>& rhs)
{
  for (unsigned int i = 0; i < N; ++i) {
    mArray[i] = rhs.mArray[i];
  }
}

template <class T, unsigned int D>
GPUdi() T SVectorGPU<T, D>::apply(unsigned int i) const
{
  return mArray[i];
}

template <class T, unsigned int D>
GPUdi() const T* SVectorGPU<T, D>::Array() const
{
  return mArray;
}

template <class T, unsigned int D>
GPUdi() T* SVectorGPU<T, D>::Array()
{
  return mArray;
}

template <unsigned int I>
struct meta_dot {
  template <class A, class B, class T>
  static GPUdi() T f(const A& lhs, const B& rhs, const T& x)
  {
    return lhs.apply(I) * rhs.apply(I) + meta_dot<I - 1>::f(lhs, rhs, x);
  }
};

template <>
struct meta_dot<0> {
  template <class A, class B, class T>
  static GPUdi() T f(const A& lhs, const B& rhs, const T& /*x */)
  {
    return lhs.apply(0) * rhs.apply(0);
  }
};

template <class T, unsigned int D>
GPUdi() T Dot(const SVectorGPU<T, D>& lhs, const SVectorGPU<T, D>& rhs)
{
  return meta_dot<D - 1>::f(lhs, rhs, T());
}

template <class T, unsigned int D>
GPUdi() SVectorGPU<T, D>& SVectorGPU<T, D>::operator-=(const SVectorGPU<T, D>& rhs)
{
  for (unsigned int i = 0; i < D; ++i) {
    mArray[i] -= rhs.apply(i);
  }
  return *this;
}

template <class T, unsigned int D>
GPUd() SVectorGPU<T, D>& SVectorGPU<T, D>::operator+=(const SVectorGPU<T, D>& rhs)
{
  for (unsigned int i = 0; i < D; ++i) {
    mArray[i] += rhs.apply(i);
  }
  return *this;
}

template <unsigned int I>
struct meta_matrix_dot {

  template <class MatrixA, class MatrixB>
  static GPUdi() typename MatrixA::value_type f(const MatrixA& lhs,
                                                const MatrixB& rhs,
                                                const unsigned int offset)
  {
    return lhs.apply(offset / MatrixB::kCols * MatrixA::kCols + I) *
             rhs.apply(MatrixB::kCols * I + offset % MatrixB::kCols) +
           meta_matrix_dot<I - 1>::f(lhs, rhs, offset);
  }

  template <class MatrixA, class MatrixB>
  static GPUdi() typename MatrixA::value_type g(const MatrixA& lhs,
                                                const MatrixB& rhs,
                                                unsigned int i,
                                                unsigned int j)
  {
    return lhs(i, I) * rhs(I, j) +
           meta_matrix_dot<I - 1>::g(lhs, rhs, i, j);
  }
};

template <>
struct meta_matrix_dot<0> {

  template <class MatrixA, class MatrixB>
  static GPUdi() typename MatrixA::value_type f(const MatrixA& lhs,
                                                const MatrixB& rhs,
                                                const unsigned int offset)
  {
    return lhs.apply(offset / MatrixB::kCols * MatrixA::kCols) *
           rhs.apply(offset % MatrixB::kCols);
  }

  // multiplication using i and j
  template <class MatrixA, class MatrixB>
  static GPUdi() typename MatrixA::value_type g(const MatrixA& lhs,
                                                const MatrixB& rhs,
                                                unsigned int i, unsigned int j)
  {
    return lhs(i, 0) * rhs(0, j);
  }
};

namespace row_offsets_utils
{
template <int...>
struct indices {
};

template <int I, class IndexTuple, int N>
struct make_indices_impl;

template <int I, int... Indices, int N>
struct make_indices_impl<I, indices<Indices...>, N> {
  typedef typename make_indices_impl<I + 1, indices<Indices..., I>,
                                     N>::type type;
};

template <int N, int... Indices>
struct make_indices_impl<N, indices<Indices...>, N> {
  typedef indices<Indices...> type;
};

template <int N>
struct make_indices : make_indices_impl<0, indices<>, N> {
};

template <int I0, class F, int... I>
constexpr auto do_make(F f, indices<I...>) -> gpu::gpustd::array<int, sizeof...(I)>
{
  gpu::gpustd::array<int, sizeof...(I)> retarr = {f(I0 + I)...};
  return retarr;
}

template <int N, int I0 = 0, class F>
constexpr auto make(F f) -> gpu::gpustd::array<int, N>
{
  return do_make<I0>(f, typename make_indices<N>::type());
}
} // namespace row_offsets_utils

// Symm representation
template <class T, unsigned int D>
class MatRepSymGPU
{
 public:
  typedef T value_type;
  GPUdDefault() MatRepSymGPU() = default;
  GPUdi() T& operator()(unsigned int i, unsigned int j)
  {
    return mArray[offset(i, j)];
  }

  GPUdi() T const& operator()(unsigned int i, unsigned int j) const
  {
    return mArray[offset(i, j)];
  }

  GPUdi() T& operator[](unsigned int i)
  {
    return mArray[off(i)];
  }

  GPUdi() T const& operator[](unsigned int i) const
  {
    return mArray[off(i)];
  }

  GPUdi() T apply(unsigned int i) const
  {
    return mArray[off(i)];
  }

  GPUdi() T* Array() { return mArray; }

  GPUdi() const T* Array() const { return mArray; }

  // assignment: only symmetric to symmetric allowed
  template <class R>
  GPUdi() MatRepSymGPU<T, D>& operator=(const R&)
  {
    GPU_STATIC_CHECK(0 == 1, Check_symmetric_matrices_equivalence);
    return *this;
  }

  GPUdi() MatRepSymGPU<T, D>& operator=(const MatRepSymGPU& rhs)
  {
    for (unsigned int i = 0; i < kSize; ++i) {
      mArray[i] = rhs.Array()[i];
    }
    return *this;
  }

  enum {
    kRows = D,              // rows
    kCols = D,              // columns
    kSize = D * (D + 1) / 2 // rows*columns
  };

  static constexpr int off0(int i) { return i == 0 ? 0 : off0(i - 1) + i; }
  static constexpr int off2(int i, int j) { return j < i ? off0(i) + j : off0(j) + i; }
  static constexpr int off1(int i) { return off2(i / D, i % D); }

  static GPUdi() int off(int i)
  {
    static constexpr auto v = row_offsets_utils::make<D * D>(off1);
    return v[i];
  }

  static GPUdi() constexpr unsigned int offset(unsigned int i, unsigned int j)
  {
    return off(i * D + j);
  }

 private:
  T mArray[kSize];
};

/// SMatReprStd starting port here
template <class T, unsigned int D1, unsigned int D2 = D1>
class MatRepStdGPU
{
 public:
  typedef T value_type;
  GPUdDefault() MatRepStdGPU() = default;
  GPUdi() const T& operator()(unsigned int i, unsigned int j) const
  {
    return mArray[i * D2 + j];
  }
  GPUdi() T& operator()(unsigned int i, unsigned int j)
  {
    return mArray[i * D2 + j];
  }
  GPUdi() T& operator[](unsigned int i) { return mArray[i]; }
  GPUdi() const T& operator[](unsigned int i) const { return mArray[i]; }
  GPUdi() T apply(unsigned int i) const { return mArray[i]; }
  GPUdi() T* Array() { return mArray; }
  GPUdi() const T* Array() const { return mArray; }

  template <class R>
  GPUdi() MatRepStdGPU<T, D1, D2>& operator=(const R& rhs)
  {
    for (unsigned int i = 0; i < kSize; ++i) {
      mArray[i] = rhs[i];
    }
    return *this;
  }
  template <class R>
  GPUdi() bool operator==(const R& rhs) const
  {
    bool rc = true;
    for (unsigned int i = 0; i < kSize; ++i) {
      rc = rc && (mArray[i] == rhs[i]);
    }
    return rc;
  }
  enum {
    kRows = D1,     // rows
    kCols = D2,     // columns
    kSize = D1 * D2 // rows*columns
  };

 private:
  T mArray[kSize];
};

/// SMatrixGPU starting port here
struct SMatrixIdentity {
};
struct SMatrixNoInit {
};

// Expression utility to describe operations
template <class ExprType, class T, unsigned int D, unsigned int D2 = 1, class R1 = MatRepStdGPU<T, D, D2>>
class Expr
{
 public:
  typedef T value_type;
  GPUd() Expr(const ExprType& rhs) : mRhs(rhs) {} // NOLINT: False positive
  GPUdDefault() ~Expr() = default;
  GPUdi() T apply(unsigned int i) const
  {
    return mRhs.apply(i);
  }
  GPUdi() T operator()(unsigned int i, unsigned j) const
  {
    return mRhs(i, j);
  }
  GPUdi() bool IsInUse(const T* p) const
  {
    return mRhs.IsInUse(p);
  }

  enum {
    kRows = D,
    kCols = D2
  };

 private:
  ExprType mRhs;
};

template <class T, unsigned int D1, unsigned int D2 = D1, class R = MatRepStdGPU<T, D1, D2>>
class SMatrixGPU
{
 public:
  typedef T value_type;
  typedef R rep_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  GPUdDefault() SMatrixGPU() = default;
  GPUdi() SMatrixGPU(SMatrixNoInit) {}
  GPUd() SMatrixGPU(SMatrixIdentity);
  GPUd() SMatrixGPU(const SMatrixGPU<T, D1, D2, R>& rhs);
  template <class A, class R2>
  GPUd() SMatrixGPU(const Expr<A, T, D1, D2, R2>& rhs);
  template <class M>
  GPUd() SMatrixGPU<T, D1, D2, R>& operator=(const M& rhs);
  template <class A, class R2>
  GPUd() SMatrixGPU<T, D1, D2, R>& operator=(const Expr<A, T, D1, D2, R2>& rhs);
  enum {
    kRows = D1,     // rows
    kCols = D2,     // columns
    kSize = D1 * D2 // rows*columns
  };
  GPUd() T apply(unsigned int i) const;
  GPUd() const T* Array() const;
  GPUd() T* Array();
  GPUd() iterator begin();
  GPUd() iterator end();
  GPUd() const T& operator()(unsigned int i, unsigned int j) const;
  GPUd() T& operator()(unsigned int i, unsigned int j);

  class SMatrixRowGPU
  {
   public:
    GPUd() SMatrixRowGPU(SMatrixGPU<T, D1, D2, R>& rhs, unsigned int i) : mMat(&rhs), mRow(i)
    {
    }
    GPUd() T& operator[](int j) { return (*mMat)(mRow, j); }
    //
   private:
    SMatrixGPU<T, D1, D2, R>* mMat;
    unsigned int mRow;
  };

  class SMatrixRowGPUconst
  {
   public:
    GPUd() SMatrixRowGPUconst(const SMatrixGPU<T, D1, D2, R>& rhs, unsigned int i) : mMat(&rhs), mRow(i)
    {
    }
    //
    GPUd() const T& operator[](int j) const { return (*mMat)(mRow, j); }
    //
   private:
    const SMatrixGPU<T, D1, D2, R>* mMat;
    unsigned int mRow;
  };

  GPUd() SMatrixRowGPUconst operator[](unsigned int i) const { return SMatrixRowGPUconst(*this, i); }
  GPUd() SMatrixRowGPU operator[](unsigned int i) { return SMatrixRowGPU(*this, i); }
  GPUd() bool Invert();
  GPUd() bool IsInUse(const T* p) const;

 public:
  R mRep;
};

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() SMatrixGPU<T, D1, D2, R>::SMatrixGPU(SMatrixIdentity)
{
  for (unsigned int i = 0; i < R::kSize; ++i) {
    mRep.Array()[i] = 0;
  }
  if (D1 <= D2) {
    for (unsigned int i = 0; i < D1; ++i) {
      mRep[i * D2 + i] = 1;
    }
  } else {
    for (unsigned int i = 0; i < D2; ++i) {
      mRep[i * D2 + i] = 1;
    }
  }
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() SMatrixGPU<T, D1, D2, R>::SMatrixGPU(const SMatrixGPU<T, D1, D2, R>& rhs)
{
  mRep = rhs.mRep;
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() T* SMatrixGPU<T, D1, D2, R>::begin()
{
  return mRep.Array();
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() T* SMatrixGPU<T, D1, D2, R>::end()
{
  return mRep.Array() + R::kSize;
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() bool SMatrixGPU<T, D1, D2, R>::IsInUse(const T* p) const
{
  return p == mRep.Array();
}

template <class T, unsigned int D1, unsigned int D2, class A, class R1, class R2>
struct Assign {
  // Evaluate the expression from general to general matrices.
  GPUd() static void Evaluate(SMatrixGPU<T, D1, D2, R1>& lhs, const Expr<A, T, D1, D2, R2>& rhs)
  {
    if (!rhs.IsInUse(lhs.begin())) {
      unsigned int l = 0;
      for (unsigned int i = 0; i < D1; ++i) {
        for (unsigned int j = 0; j < D2; ++j) {
          lhs.mRep[l] = rhs(i, j);
          l++;
        }
      }
    } else {
      T tmp[D1 * D2];
      unsigned int l = 0;
      for (unsigned int i = 0; i < D1; ++i) {
        for (unsigned int j = 0; j < D2; ++j) {
          tmp[l] = rhs(i, j);
          l++;
        }
      }

      for (unsigned int i = 0; i < D1 * D2; ++i) {
        lhs.mRep[i] = tmp[i];
      }
    }
  }
};

template <class T, unsigned int D1, unsigned int D2, class A>
struct Assign<T, D1, D2, A, MatRepSymGPU<T, D1>, MatRepSymGPU<T, D1>> {
  // Evaluate the expression from  symmetric to symmetric matrices.
  GPUd() static void Evaluate(SMatrixGPU<T, D1, D2, MatRepSymGPU<T, D1>>& lhs,
                              const Expr<A, T, D1, D2, MatRepSymGPU<T, D1>>& rhs)
  {
    if (!rhs.IsInUse(lhs.begin())) {
      unsigned int l = 0;
      for (unsigned int i = 0; i < D1; ++i) {
        // storage of symmetric matrix is in lower block
        for (unsigned int j = 0; j <= i; ++j) {
          lhs.mRep.Array()[l] = rhs(i, j);
          l++;
        }
      }
    } else {
      T tmp[MatRepSymGPU<T, D1>::kSize];
      unsigned int l = 0;
      for (unsigned int i = 0; i < D1; ++i) {
        for (unsigned int j = 0; j <= i; ++j) {
          tmp[l] = rhs(i, j);
          l++;
        }
      }
      for (unsigned int i = 0; i < MatRepSymGPU<T, D1>::kSize; ++i) {
        lhs.mRep.Array()[i] = tmp[i];
      }
    }
  }
};

// avoid assigment from expression based on a general matrix to a symmetric matrix
template <class T, unsigned int D1, unsigned int D2, class A>
struct Assign<T, D1, D2, A, MatRepSymGPU<T, D1>, MatRepStdGPU<T, D1, D2>> {
  GPUd() static void Evaluate(SMatrixGPU<T, D1, D2, MatRepSymGPU<T, D1>>&,
                              const Expr<A, T, D1, D2, MatRepStdGPU<T, D1, D2>>&)
  {
    GPU_STATIC_CHECK(0 == 1, Check_general_to_symmetric_matrix_assignment);
  }
};

// Force Expression evaluation from general to symmetric
struct AssignSym {
  // assign a symmetric matrix from an expression
  template <class T, unsigned int D, class A, class R>
  GPUd() static void Evaluate(SMatrixGPU<T, D, D, MatRepSymGPU<T, D>>& lhs, const Expr<A, T, D, D, R>& rhs)
  {
    unsigned int l = 0;
    for (unsigned int i = 0; i < D; ++i) {
      for (unsigned int j = 0; j <= i; ++j) {
        lhs.mRep.Array()[l] = rhs(i, j);
        l++;
      }
    }
  }

  // assign the symmetric matrix from a general matrix
  template <class T, unsigned int D, class R>
  GPUd() static void Evaluate(SMatrixGPU<T, D, D, MatRepSymGPU<T, D>>& lhs, const SMatrixGPU<T, D, D, R>& rhs)
  {
    unsigned int l = 0;
    for (unsigned int i = 0; i < D; ++i) {
      for (unsigned int j = 0; j <= i; ++j) {
        lhs.mRep.Array()[l] = rhs(i, j);
        l++;
      }
    }
  }
};

template <class T, unsigned int D1, unsigned int D2, class R>
template <class A, class R2>
GPUdi() SMatrixGPU<T, D1, D2, R>& SMatrixGPU<T, D1, D2, R>::operator=(const Expr<A, T, D1, D2, R2>& rhs)
{
  Assign<T, D1, D2, A, R, R2>::Evaluate(*this, rhs);
  return *this;
}

template <class T, unsigned int D1, unsigned int D2, class R>
template <class M>
GPUdi() SMatrixGPU<T, D1, D2, R>& SMatrixGPU<T, D1, D2, R>::operator=(const M& rhs)
{
  mRep = rhs.mRep;
  return *this;
}

template <class T, unsigned int D1, unsigned int D2, class R>
template <class A, class R2>
GPUdi() SMatrixGPU<T, D1, D2, R>::SMatrixGPU(const Expr<A, T, D1, D2, R2>& rhs)
{
  operator=(rhs);
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() const T& SMatrixGPU<T, D1, D2, R>::operator()(unsigned int i, unsigned int j) const
{
  return mRep(i, j);
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() T& SMatrixGPU<T, D1, D2, R>::operator()(unsigned int i, unsigned int j)
{
  return mRep(i, j);
}

template <class T, class R1, class R2>
struct MultPolicyGPU {
  enum {
    N1 = R1::kRows,
    N2 = R2::kCols
  };
  typedef MatRepStdGPU<T, N1, N2> RepType;
};

template <class MatrixA, class MatrixB, class T, unsigned int D>
class MatrixMulOpGPU
{
 public:
  GPUd() MatrixMulOpGPU(const MatrixA& lhs, const MatrixB& rhs) : lhs_(lhs), rhs_(rhs) {}
  GPUdDefault() ~MatrixMulOpGPU() = default;
  GPUdi() T apply(unsigned int i) const
  {
    return meta_matrix_dot<D - 1>::f(lhs_, rhs_, i);
  }

  GPUdi() T operator()(unsigned int i, unsigned int j) const
  {
    return meta_matrix_dot<D - 1>::g(lhs_, rhs_, i, j);
  }

  GPUdi() bool IsInUse(const T* p) const
  {
    return lhs_.IsInUse(p) || rhs_.IsInUse(p);
  }

 protected:
  const MatrixA& lhs_;
  const MatrixB& rhs_;
};

template <class T, unsigned int D1, unsigned int D, unsigned int D2, class R1, class R2>
GPUdi() Expr<MatrixMulOpGPU<SMatrixGPU<T, D1, D, R1>, SMatrixGPU<T, D, D2, R2>, T, D>, T, D1, D2, typename MultPolicyGPU<T, R1, R2>::RepType>
  operator*(const SMatrixGPU<T, D1, D, R1>& lhs, const SMatrixGPU<T, D, D2, R2>& rhs)
{
  typedef MatrixMulOpGPU<SMatrixGPU<T, D1, D, R1>, SMatrixGPU<T, D, D2, R2>, T, D> MatMulOp;
  return Expr<MatMulOp, T, D1, D2,
              typename MultPolicyGPU<T, R1, R2>::RepType>(MatMulOp(lhs, rhs));
}

/// Inversion
template <unsigned int D, unsigned int N = D>
class Inverter
{
 public:
  // generic square matrix using LU factorization
  template <class MatrixRep>
  GPUd() static bool Dinv(MatrixRep& rhs)
  {
    unsigned int work[N + 1] = {0};
    typename MatrixRep::value_type det(0.0);

    if (DfactMatrix(rhs, det, work) != 0) {
      return false;
    }

    int ifail = DfinvMatrix(rhs, work);
    if (ifail == 0) {
      return true;
    }
    return false;
  }

  //  symmetric matrix inversion (Bunch-kaufman pivoting)
  template <class T>
  GPUd() static bool Dinv(MatRepSymGPU<T, D>& rhs)
  {
    int ifail{0};
    InvertBunchKaufman(rhs, ifail);
    if (!ifail) {
      return true;
    }
    return false;
  }

  // LU Factorization method for inversion of general square matrices
  template <class T>
  GPUd() static int DfactMatrix(MatRepStdGPU<T, D, N>& rhs, T& det, unsigned int* work);

  // LU inversion of general square matrices. To be called after DFactMatrix
  template <class T>
  GPUd() static int DfinvMatrix(MatRepStdGPU<T, D, N>& rhs, unsigned int* work);

  // Bunch-Kaufman method for inversion of symmetric matrices
  template <class T>
  GPUd() static void InvertBunchKaufman(MatRepSymGPU<T, D>& rhs, int& ifail);
};

template <unsigned int D, unsigned int N>
template <class T>
GPUdi() void Inverter<D, N>::InvertBunchKaufman(MatRepSymGPU<T, D>& rhs, int& ifail)
{
  typedef T value_type;
  int i, j, k, s;
  int pivrow;
  const int nrow = MatRepSymGPU<T, D>::kRows;

  SVectorGPU<T, MatRepSymGPU<T, D>::kRows> xvec;
  SVectorGPU<int, MatRepSymGPU<T, D>::kRows> pivv;

  typedef int* pivIter;
  typedef T* mIter;

  mIter x = xvec.begin();
  // x[i] is used as helper storage, needs to have at least size nrow.
  pivIter piv = pivv.begin();
  // piv[i] is used to store details of exchanges

  value_type temp1, temp2;
  mIter ip, mjj, iq;
  value_type lambda, sigma;
  const value_type alpha = .6404; // = (1+sqrt(17))/8
  // LM (04/2009) remove this useless check (it is not in LAPACK) which fails inversion of
  // a matrix with  values < epsilon in the diagonal
  //
  // const double epsilon = 32*std::numeric_limits<T>::epsilon();
  // whenever a sum of two doubles is below or equal to epsilon
  // it is set to zero.
  // this constant could be set to zero but then the algorithm
  // doesn't neccessarily detect that a matrix is singular

  for (i = 0; i < nrow; i++) {
    piv[i] = i + 1;
  }

  ifail = 0;

  // compute the factorization P*A*P^T = L * D * L^T
  // L is unit lower triangular, D is direct sum of 1x1 and 2x2 matrices
  // L and D^-1 are stored in A = *this, P is stored in piv[]

  for (j = 1; j < nrow; j += s) // main loop over columns
  {
    mjj = rhs.Array() + j * (j - 1) / 2 + j - 1;
    lambda = 0; // compute lambda = max of A(j+1:n,j)
    pivrow = j + 1;
    ip = rhs.Array() + (j + 1) * j / 2 + j - 1;
    for (i = j + 1; i <= nrow; ip += i++) {
      if (o2::gpu::GPUCommonMath::Abs(*ip) > lambda) {
        lambda = o2::gpu::GPUCommonMath::Abs(*ip);
        pivrow = i;
      }
    }

    if (lambda == 0) {
      if (*mjj == 0) {
        ifail = 1;
        return;
      }
      s = 1;
      *mjj = 1.0f / *mjj;
    } else {
      if (o2::gpu::GPUCommonMath::Abs(*mjj) >= lambda * alpha) {
        s = 1;
        pivrow = j;
      } else {
        sigma = 0; // compute sigma = max A(pivrow, j:pivrow-1)
        ip = rhs.Array() + pivrow * (pivrow - 1) / 2 + j - 1;
        for (k = j; k < pivrow; k++) {
          if (o2::gpu::GPUCommonMath::Abs(*ip) > sigma) {
            sigma = o2::gpu::GPUCommonMath::Abs(*ip);
          }
          ip++;
        }
        // sigma cannot be zero because it is at least lambda which is not zero
        if (o2::gpu::GPUCommonMath::Abs(*mjj) >= alpha * lambda * (lambda / sigma)) {
          s = 1;
          pivrow = j;
        } else if (o2::gpu::GPUCommonMath::Abs(*(rhs.Array() + pivrow * (pivrow - 1) / 2 + pivrow - 1)) >= alpha * sigma) {
          s = 1;
        } else {
          s = 2;
        }
      }
      if (pivrow == j) // no permutation neccessary
      {
        piv[j - 1] = pivrow;
        if (*mjj == 0) {
          ifail = 1;
          return;
        }
        temp2 = *mjj = 1.0f / *mjj; // invert D(j,j)

        // update A(j+1:n, j+1,n)
        for (i = j + 1; i <= nrow; i++) {
          temp1 = *(rhs.Array() + i * (i - 1) / 2 + j - 1) * temp2;
          ip = rhs.Array() + i * (i - 1) / 2 + j;
          for (k = j + 1; k <= i; k++) {
            *ip -= static_cast<T>(temp1 * *(rhs.Array() + k * (k - 1) / 2 + j - 1));
            //                   if (o2::gpu::GPUCommonMath::Abs(*ip) <= epsilon)
            //                      *ip=0;
            ip++;
          }
        }
        // update L
        ip = rhs.Array() + (j + 1) * j / 2 + j - 1;
        for (i = j + 1; i <= nrow; ip += i++) {
          *ip *= static_cast<T>(temp2);
        }
      } else if (s == 1) // 1x1 pivot
      {
        piv[j - 1] = pivrow;

        // interchange rows and columns j and pivrow in
        // submatrix (j:n,j:n)
        ip = rhs.Array() + pivrow * (pivrow - 1) / 2 + j;
        for (i = j + 1; i < pivrow; i++, ip++) {
          temp1 = *(rhs.Array() + i * (i - 1) / 2 + j - 1);
          *(rhs.Array() + i * (i - 1) / 2 + j - 1) = *ip;
          *ip = static_cast<T>(temp1);
        }
        temp1 = *mjj;
        *mjj = *(rhs.Array() + pivrow * (pivrow - 1) / 2 + pivrow - 1);
        *(rhs.Array() + pivrow * (pivrow - 1) / 2 + pivrow - 1) = static_cast<T>(temp1);
        ip = rhs.Array() + (pivrow + 1) * pivrow / 2 + j - 1;
        iq = ip + pivrow - j;
        for (i = pivrow + 1; i <= nrow; ip += i, iq += i++) {
          temp1 = *iq;
          *iq = *ip;
          *ip = static_cast<T>(temp1);
        }

        if (*mjj == 0) {
          ifail = 1;
          return;
        }
        temp2 = *mjj = 1.0f / *mjj; // invert D(j,j)

        // update A(j+1:n, j+1:n)
        for (i = j + 1; i <= nrow; i++) {
          temp1 = *(rhs.Array() + i * (i - 1) / 2 + j - 1) * temp2;
          ip = rhs.Array() + i * (i - 1) / 2 + j;
          for (k = j + 1; k <= i; k++) {
            *ip -= static_cast<T>(temp1 * *(rhs.Array() + k * (k - 1) / 2 + j - 1));
            //                   if (o2::gpu::GPUCommonMath::Abs(*ip) <= epsilon)
            //                      *ip=0;
            ip++;
          }
        }
        // update L
        ip = rhs.Array() + (j + 1) * j / 2 + j - 1;
        for (i = j + 1; i <= nrow; ip += i++) {
          *ip *= static_cast<T>(temp2);
        }
      } else // s=2, ie use a 2x2 pivot
      {
        piv[j - 1] = -pivrow;
        piv[j] = 0; // that means this is the second row of a 2x2 pivot

        if (j + 1 != pivrow) {
          // interchange rows and columns j+1 and pivrow in
          // submatrix (j:n,j:n)
          ip = rhs.Array() + pivrow * (pivrow - 1) / 2 + j + 1;
          for (i = j + 2; i < pivrow; i++, ip++) {
            temp1 = *(rhs.Array() + i * (i - 1) / 2 + j);
            *(rhs.Array() + i * (i - 1) / 2 + j) = *ip;
            *ip = static_cast<T>(temp1);
          }
          temp1 = *(mjj + j + 1);
          *(mjj + j + 1) =
            *(rhs.Array() + pivrow * (pivrow - 1) / 2 + pivrow - 1);
          *(rhs.Array() + pivrow * (pivrow - 1) / 2 + pivrow - 1) = static_cast<T>(temp1);
          temp1 = *(mjj + j);
          *(mjj + j) = *(rhs.Array() + pivrow * (pivrow - 1) / 2 + j - 1);
          *(rhs.Array() + pivrow * (pivrow - 1) / 2 + j - 1) = static_cast<T>(temp1);
          ip = rhs.Array() + (pivrow + 1) * pivrow / 2 + j;
          iq = ip + pivrow - (j + 1);
          for (i = pivrow + 1; i <= nrow; ip += i, iq += i++) {
            temp1 = *iq;
            *iq = *ip;
            *ip = static_cast<T>(temp1);
          }
        }
        // invert D(j:j+1,j:j+1)
        temp2 = *mjj * *(mjj + j + 1) - *(mjj + j) * *(mjj + j);
        if (temp2 == 0) {
          printf("SymMatrix::bunch_invert: error in pivot choice");
        }
        temp2 = 1. / temp2;
        // this quotient is guaranteed to exist by the choice
        // of the pivot
        temp1 = *mjj;
        *mjj = static_cast<T>(*(mjj + j + 1) * temp2);
        *(mjj + j + 1) = static_cast<T>(temp1 * temp2);
        *(mjj + j) = static_cast<T>(-*(mjj + j) * temp2);

        if (j < nrow - 1) // otherwise do nothing
        {
          // update A(j+2:n, j+2:n)
          for (i = j + 2; i <= nrow; i++) {
            ip = rhs.Array() + i * (i - 1) / 2 + j - 1;
            temp1 = *ip * *mjj + *(ip + 1) * *(mjj + j);
            //                   if (o2::gpu::GPUCommonMath::Abs(temp1 ) <= epsilon)
            //                      temp1 = 0;
            temp2 = *ip * *(mjj + j) + *(ip + 1) * *(mjj + j + 1);
            //                   if (o2::gpu::GPUCommonMath::Abs(temp2 ) <= epsilon)
            //                      temp2 = 0;
            for (k = j + 2; k <= i; k++) {
              ip = rhs.Array() + i * (i - 1) / 2 + k - 1;
              iq = rhs.Array() + k * (k - 1) / 2 + j - 1;
              *ip -= static_cast<T>(temp1 * *iq + temp2 * *(iq + 1));
              //                      if (o2::gpu::GPUCommonMath::Abs(*ip) <= epsilon)
              //                         *ip = 0;
            }
          }
          // update L
          for (i = j + 2; i <= nrow; i++) {
            ip = rhs.Array() + i * (i - 1) / 2 + j - 1;
            temp1 = *ip * *mjj + *(ip + 1) * *(mjj + j);
            //                   if (o2::gpu::GPUCommonMath::Abs(temp1) <= epsilon)
            //                      temp1 = 0;
            *(ip + 1) = *ip * *(mjj + j) + *(ip + 1) * *(mjj + j + 1);
            //                   if (o2::gpu::GPUCommonMath::Abs(*(ip+1)) <= epsilon)
            //                      *(ip+1) = 0;
            *ip = static_cast<T>(temp1);
          }
        }
      }
    }
  } // end of main loop over columns

  if (j == nrow) // the the last pivot is 1x1
  {
    mjj = rhs.Array() + j * (j - 1) / 2 + j - 1;
    if (*mjj == 0) {
      ifail = 1;
      return;
    } else {
      *mjj = 1.0f / *mjj;
    }
  } // end of last pivot code

  // computing the inverse from the factorization

  for (j = nrow; j >= 1; j -= s) // loop over columns
  {
    mjj = rhs.Array() + j * (j - 1) / 2 + j - 1;
    if (piv[j - 1] > 0) // 1x1 pivot, compute column j of inverse
    {
      s = 1;
      if (j < nrow) {
        ip = rhs.Array() + (j + 1) * j / 2 + j - 1;
        for (i = 0; i < nrow - j; ip += 1 + j + i++) {
          x[i] = *ip;
        }
        for (i = j + 1; i <= nrow; i++) {
          temp2 = 0;
          ip = rhs.Array() + i * (i - 1) / 2 + j;
          for (k = 0; k <= i - j - 1; k++) {
            temp2 += *ip++ * x[k];
          }
          for (ip += i - 1; k < nrow - j; ip += 1 + j + k++) {
            temp2 += *ip * x[k];
          }
          *(rhs.Array() + i * (i - 1) / 2 + j - 1) = static_cast<T>(-temp2);
        }
        temp2 = 0;
        ip = rhs.Array() + (j + 1) * j / 2 + j - 1;
        for (k = 0; k < nrow - j; ip += 1 + j + k++) {
          temp2 += x[k] * *ip;
        }
        *mjj -= static_cast<T>(temp2);
      }
    } else // 2x2 pivot, compute columns j and j-1 of the inverse
    {
      if (piv[j - 1] != 0) {
        printf("error in piv %lf \n", static_cast<T>(piv[j - 1]));
      }
      s = 2;
      if (j < nrow) {
        ip = rhs.Array() + (j + 1) * j / 2 + j - 1;
        for (i = 0; i < nrow - j; ip += 1 + j + i++) {
          x[i] = *ip;
        }
        for (i = j + 1; i <= nrow; i++) {
          temp2 = 0;
          ip = rhs.Array() + i * (i - 1) / 2 + j;
          for (k = 0; k <= i - j - 1; k++) {
            temp2 += *ip++ * x[k];
          }
          for (ip += i - 1; k < nrow - j; ip += 1 + j + k++) {
            temp2 += *ip * x[k];
          }
          *(rhs.Array() + i * (i - 1) / 2 + j - 1) = static_cast<T>(-temp2);
        }
        temp2 = 0;
        ip = rhs.Array() + (j + 1) * j / 2 + j - 1;
        for (k = 0; k < nrow - j; ip += 1 + j + k++) {
          temp2 += x[k] * *ip;
        }
        *mjj -= static_cast<T>(temp2);
        temp2 = 0;
        ip = rhs.Array() + (j + 1) * j / 2 + j - 2;
        for (i = j + 1; i <= nrow; ip += i++) {
          temp2 += *ip * *(ip + 1);
        }
        *(mjj - 1) -= static_cast<T>(temp2);
        ip = rhs.Array() + (j + 1) * j / 2 + j - 2;
        for (i = 0; i < nrow - j; ip += 1 + j + i++) {
          x[i] = *ip;
        }
        for (i = j + 1; i <= nrow; i++) {
          temp2 = 0;
          ip = rhs.Array() + i * (i - 1) / 2 + j;
          for (k = 0; k <= i - j - 1; k++) {
            temp2 += *ip++ * x[k];
          }
          for (ip += i - 1; k < nrow - j; ip += 1 + j + k++) {
            temp2 += *ip * x[k];
          }
          *(rhs.Array() + i * (i - 1) / 2 + j - 2) = static_cast<T>(-temp2);
        }
        temp2 = 0;
        ip = rhs.Array() + (j + 1) * j / 2 + j - 2;
        for (k = 0; k < nrow - j; ip += 1 + j + k++) {
          temp2 += x[k] * *ip;
        }
        *(mjj - j) -= static_cast<T>(temp2);
      }
    }

    // interchange rows and columns j and piv[j-1]
    // or rows and columns j and -piv[j-2]

    pivrow = (piv[j - 1] == 0) ? -piv[j - 2] : piv[j - 1];
    ip = rhs.Array() + pivrow * (pivrow - 1) / 2 + j;
    for (i = j + 1; i < pivrow; i++, ip++) {
      temp1 = *(rhs.Array() + i * (i - 1) / 2 + j - 1);
      *(rhs.Array() + i * (i - 1) / 2 + j - 1) = *ip;
      *ip = static_cast<T>(temp1);
    }
    temp1 = *mjj;
    *mjj = *(rhs.Array() + pivrow * (pivrow - 1) / 2 + pivrow - 1);
    *(rhs.Array() + pivrow * (pivrow - 1) / 2 + pivrow - 1) = static_cast<T>(temp1);
    if (s == 2) {
      temp1 = *(mjj - 1);
      *(mjj - 1) = *(rhs.Array() + pivrow * (pivrow - 1) / 2 + j - 2);
      *(rhs.Array() + pivrow * (pivrow - 1) / 2 + j - 2) = static_cast<T>(temp1);
    }

    ip = rhs.Array() + (pivrow + 1) * pivrow / 2 + j - 1; // &A(i,j)
    iq = ip + pivrow - j;
    for (i = pivrow + 1; i <= nrow; ip += i, iq += i++) {
      temp1 = *iq;
      *iq = *ip;
      *ip = static_cast<T>(temp1);
    }
  } // end of loop over columns (in computing inverse from factorization)

  return; // inversion successful
}

// LU factorization
template <unsigned int D, unsigned int n>
template <class T>
GPUdi() int Inverter<D, n>::DfactMatrix(MatRepStdGPU<T, D, n>& rhs, T& det, unsigned int* ir)
{
  if (D != n) {
    return -1;
  }

  int ifail, jfail;
  typedef T* mIter;

  typedef T value_type;

  value_type tf;
  value_type g1 = 1.0e-19, g2 = 1.0e19;

  value_type p, q, t;
  value_type s11, s12;

  // LM (04.09) : remove useless check on epsilon and set it to zero
  const value_type epsilon = 0.0;
  // double epsilon = 8*std::numeric_limits<T>::epsilon();
  // could be set to zero (like it was before)
  // but then the algorithm often doesn't detect
  // that a matrix is singular

  int normal = 0, imposs = -1;
  int jrange = 0, jover = 1, junder = -1;
  ifail = normal;
  jfail = jrange;
  int nxch = 0;
  det = 1.0;
  mIter mj = rhs.Array();
  mIter mjj = mj;
  for (unsigned int j = 1; j <= n; j++) {
    unsigned int k = j;
    p = (o2::gpu::GPUCommonMath::Abs(*mjj));
    if (j != n) {
      mIter mij = mj + n + j - 1;
      for (unsigned int i = j + 1; i <= n; i++) {
        q = (o2::gpu::GPUCommonMath::Abs(*(mij)));
        if (q > p) {
          k = i;
          p = q;
        }
        mij += n;
      }
      if (k == j) {
        if (p <= epsilon) {
          det = 0;
          ifail = imposs;
          jfail = jrange;
          return ifail;
        }
        det = -det; // in this case the sign of the determinant
                    // must not change. So I change it twice.
      }
      mIter mjl = mj;
      mIter mkl = rhs.Array() + (k - 1) * n;
      for (unsigned int l = 1; l <= n; l++) {
        tf = *mjl;
        *(mjl++) = *mkl;
        *(mkl++) = static_cast<T>(tf);
      }
      nxch = nxch + 1; // this makes the determinant change its sign
      ir[nxch] = (((j) << 12) + (k));
    } else {
      if (p <= epsilon) {
        det = 0.0;
        ifail = imposs;
        jfail = jrange;
        return ifail;
      }
    }
    det *= *mjj;
    *mjj = 1.0f / *mjj;
    t = (o2::gpu::GPUCommonMath::Abs(det));
    if (t < g1) {
      det = 0.0;
      if (jfail == jrange) {
        jfail = junder;
      }
    } else if (t > g2) {
      det = 1.0;
      if (jfail == jrange) {
        jfail = jover;
      }
    }
    if (j != n) {
      mIter mk = mj + n;
      mIter mkjp = mk + j;
      mIter mjk = mj + j;
      for (k = j + 1; k <= n; k++) {
        s11 = -(*mjk);
        s12 = -(*mkjp);
        if (j != 1) {
          mIter mik = rhs.Array() + k - 1;
          mIter mijp = rhs.Array() + j;
          mIter mki = mk;
          mIter mji = mj;
          for (unsigned int i = 1; i < j; i++) {
            s11 += (*mik) * (*(mji++));
            s12 += (*mijp) * (*(mki++));
            mik += n;
            mijp += n;
          }
        }
        // cast to avoid warnings from double to float conversions
        *(mjk++) = static_cast<T>(-s11 * (*mjj));
        *(mkjp) = static_cast<T>(-(((*(mjj + 1))) * ((*(mkjp - 1))) + (s12)));
        mk += n;
        mkjp += n;
      }
    }
    mj += n;
    mjj += (n + 1);
  }
  if (nxch % 2 == 1) {
    det = -det;
  }
  if (jfail != jrange) {
    det = 0.0;
  }
  ir[n] = nxch;
  return 0;
}

template <unsigned int D, unsigned int n>
template <class T>
GPUdi() int Inverter<D, n>::DfinvMatrix(MatRepStdGPU<T, D, n>& rhs, unsigned int* ir)
{
  typedef T* mIter;
  typedef T value_type;

  if (D != n) {
    return -1;
  }

  value_type s31, s32;
  value_type s33, s34;

  mIter m11 = rhs.Array();
  mIter m12 = m11 + 1;
  mIter m21 = m11 + n;
  mIter m22 = m12 + n;
  *m21 = -(*m22) * (*m11) * (*m21);
  *m12 = -(*m12);
  if (n > 2) {
    mIter mi = rhs.Array() + 2 * n;
    mIter mii = rhs.Array() + 2 * n + 2;
    mIter mimim = rhs.Array() + n + 1;
    for (unsigned int i = 3; i <= n; i++) {
      unsigned int im2 = i - 2;
      mIter mj = rhs.Array();
      mIter mji = mj + i - 1;
      mIter mij = mi;
      for (unsigned int j = 1; j <= im2; j++) {
        s31 = 0.0;
        s32 = *mji;
        mIter mkj = mj + j - 1;
        mIter mik = mi + j - 1;
        mIter mjkp = mj + j;
        mIter mkpi = mj + n + i - 1;
        for (unsigned int k = j; k <= im2; k++) {
          s31 += (*mkj) * (*(mik++));
          s32 += (*(mjkp++)) * (*mkpi);
          mkj += n;
          mkpi += n;
        }
        *mij = static_cast<T>(-(*mii) * (((*(mij - n))) * ((*(mii - 1))) + (s31)));
        *mji = static_cast<T>(-s32);
        mj += n;
        mji += n;
        mij++;
      }
      *(mii - 1) = -(*mii) * (*mimim) * (*(mii - 1));
      *(mimim + 1) = -(*(mimim + 1));
      mi += n;
      mimim += (n + 1);
      mii += (n + 1);
    }
  }
  mIter mi = rhs.Array();
  mIter mii = rhs.Array();
  for (unsigned int i = 1; i < n; i++) {
    unsigned int ni = n - i;
    mIter mij = mi;
    // int j;
    for (unsigned j = 1; j <= i; j++) {
      s33 = *mij;
      mIter mikj = mi + n + j - 1;
      mIter miik = mii + 1;
      mIter min_end = mi + n;
      for (; miik < min_end;) {
        s33 += (*mikj) * (*(miik++));
        mikj += n;
      }
      *(mij++) = static_cast<T>(s33);
    }
    for (unsigned j = 1; j <= ni; j++) {
      s34 = 0.0;
      mIter miik = mii + j;
      mIter mikij = mii + j * n + j;
      for (unsigned int k = j; k <= ni; k++) {
        s34 += *mikij * (*(miik++));
        mikij += n;
      }
      *(mii + j) = s34;
    }
    mi += n;
    mii += (n + 1);
  }
  unsigned int nxch = ir[n];
  if (nxch == 0) {
    return 0;
  }
  for (unsigned int mm = 1; mm <= nxch; mm++) {
    unsigned int k = nxch - mm + 1;
    int ij = ir[k];
    int i = ij >> 12;
    int j = ij % 4096;
    mIter mki = rhs.Array() + i - 1;
    mIter mkj = rhs.Array() + j - 1;
    for (k = 1; k <= n; k++) {
      T ti = *mki;
      *mki = *mkj;
      *mkj = ti;
      mki += n;
      mkj += n;
    }
  }
  return 0;
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() bool SMatrixGPU<T, D1, D2, R>::Invert()
{
  GPU_STATIC_CHECK(D1 == D2, SMatrixGPU_not_square);
  return Inverter<D1, D2>::Dinv((*this).mRep);
}

template <class T, unsigned int D1, unsigned int D2, class R>
struct TranspPolicyGPU {
  enum {
    N1 = R::kRows,
    N2 = R::kCols
  };
  typedef MatRepStdGPU<T, N2, N1> RepType;
};

template <class T, unsigned int D1, unsigned int D2>
struct TranspPolicyGPU<T, D1, D2, MatRepSymGPU<T, D1>> {
  typedef MatRepSymGPU<T, D1> RepType;
};

template <class Matrix, class T, unsigned int D1, unsigned int D2 = D1>
class TransposeOpGPU
{
 public:
  GPUd() TransposeOpGPU(const Matrix& rhs) : mRhs(rhs) {}

  GPUdDefault() ~TransposeOpGPU() = default;

  GPUdi() T apply(unsigned int i) const
  {
    return mRhs.apply((i % D1) * D2 + i / D1);
  }
  GPUdi() T operator()(unsigned int i, unsigned j) const
  {
    return mRhs(j, i);
  }

  GPUdi() bool IsInUse(const T* p) const
  {
    return mRhs.IsInUse(p);
  }

 protected:
  const Matrix& mRhs;
};

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() SVectorGPU<T, D1> operator*(const SMatrixGPU<T, D1, D2, R>& rhs, const SVectorGPU<T, D2>& lhs)
{
  SVectorGPU<T, D1> tmp;
  for (unsigned int i = 0; i < D1; ++i) {
    const unsigned int rpos = i * D2;
    for (unsigned int j = 0; j < D2; ++j) {
      tmp[i] += rhs.apply(rpos + j) * lhs.apply(j);
    }
  }
  return tmp;
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() Expr<TransposeOpGPU<SMatrixGPU<T, D1, D2, R>, T, D1, D2>, T, D2, D1, typename TranspPolicyGPU<T, D1, D2, R>::RepType> Transpose(const SMatrixGPU<T, D1, D2, R>& rhs)
{
  typedef TransposeOpGPU<SMatrixGPU<T, D1, D2, R>, T, D1, D2> MatTrOp;
  return Expr<MatTrOp, T, D2, D1, typename TranspPolicyGPU<T, D1, D2, R>::RepType>(MatTrOp(rhs));
}

template <class T, unsigned int D1, unsigned int D2, class R>
GPUdi() SMatrixGPU<T, D1, D1, MatRepSymGPU<T, D1>> Similarity(const SMatrixGPU<T, D1, D2, R>& lhs, const SMatrixGPU<T, D2, D2, MatRepSymGPU<T, D2>>& rhs)
{
  SMatrixGPU<T, D1, D2, MatRepStdGPU<T, D1, D2>> tmp = lhs * rhs;
  typedef SMatrixGPU<T, D1, D1, MatRepSymGPU<T, D1>> SMatrixSym;
  SMatrixSym mret;
  AssignSym::Evaluate(mret, tmp * Transpose(lhs));
  return mret;
}
}; // namespace o2::math_utils
#endif
