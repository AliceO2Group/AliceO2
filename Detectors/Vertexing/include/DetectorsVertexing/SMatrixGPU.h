// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SMatrixGPU.h
/// This is a close porting of the SMatrix and SVector ROOT interfaces.
/// Original sources are on the official website:
/// - https://root.cern.ch

#ifndef ALICEO2_ROOTSMATRIX_H
#define ALICEO2_ROOTSMATRIX_H

#ifndef GPUCA_GPUCODE_DEVICE
#include <TMath.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <Math/Functions.h>
#endif

#include "GPUCommonDef.h"
#include "GPUCommonArray.h"

namespace o2::math_utils
{
#if defined(GPUCA_GPUCODE)

template <bool>
struct Check {
  Check(void*) {}
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

template <typename T, size_t N>
class SVectorGPU
{
 public:
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  GPUd() SVectorGPU();
  GPUd() SVectorGPU(const SVectorGPU<T, N>& rhs);

  GPUd() const T& operator[](size_t i) const;
  GPUd() const T& operator()(size_t i) const;
  GPUd() T& operator[](size_t i);
  GPUd() T& operator()(size_t i);
  GPUd() const T* Array() const;
  GPUd() T* Array();
  GPUd() T apply(size_t i) const;

 private:
  T mArray[N];
};

template <class T, size_t N>
GPUdi() const T& SVectorGPU<T, N>::operator[](size_t i) const
{
  return mArray[i];
}

template <class T, size_t N>
GPUdi() const T& SVectorGPU<T, N>::operator()(size_t i) const
{
  return mArray[i];
}

template <class T, size_t N>
GPUdi() T& SVectorGPU<T, N>::operator[](size_t i)
{
  return mArray[i];
}

template <class T, size_t N>
GPUdi() T& SVectorGPU<T, N>::operator()(size_t i)
{
  return mArray[i];
}

template <class T, size_t N>
GPUd() SVectorGPU<T, N>::SVectorGPU()
{
  for (size_t i = 0; i < N; ++i) {
    mArray[i] = 7;
  }
}

template <class T, size_t N>
GPUd() SVectorGPU<T, N>::SVectorGPU(const SVectorGPU<T, N>& rhs)
{
  for (size_t i = 0; i < N; ++i) {
    mArray[i] = rhs.mArray[i];
  }
}

template <class T, size_t D>
GPUdi() T SVectorGPU<T, D>::apply(size_t i) const
{
  return mArray[i];
}

template <class T, size_t D>
GPUdi() const T* SVectorGPU<T, D>::Array() const
{
  return mArray;
}

template <class T, size_t D>
GPUdi() T* SVectorGPU<T, D>::Array()
{
  return mArray;
}

// Dot operator support
template <size_t I>
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

template <class T, size_t D>
GPUdi() T Dot(const SVectorGPU<T, D>& lhs, const SVectorGPU<T, D>& rhs)
{
  return meta_dot<D - 1>::f(lhs, rhs, T());
}

template <size_t I>
struct meta_matrix_dot {

  template <class MatrixA, class MatrixB>
  static GPUdi() typename MatrixA::value_type f(const MatrixA& lhs,
                                                const MatrixB& rhs,
                                                const size_t offset)
  {
    return lhs.apply(offset / MatrixB::kCols * MatrixA::kCols + I) *
             rhs.apply(MatrixB::kCols * I + offset % MatrixB::kCols) +
           meta_matrix_dot<I - 1>::f(lhs, rhs, offset);
  }

  template <class MatrixA, class MatrixB>
  static GPUdi() typename MatrixA::value_type g(const MatrixA& lhs,
                                                const MatrixB& rhs,
                                                size_t i,
                                                size_t j)
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
                                                const size_t offset)
  {
    return lhs.apply(offset / MatrixB::kCols * MatrixA::kCols) *
           rhs.apply(offset % MatrixB::kCols);
  }

  // multiplication using i and j
  template <class MatrixA, class MatrixB>
  static GPUdi() typename MatrixA::value_type g(const MatrixA& lhs,
                                                const MatrixB& rhs,
                                                size_t i, size_t j)
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
template <class T, size_t D>
class MatRepSymGPU
{
 public:
  typedef T value_type;
  GPUdi() MatRepSymGPU(){};
  GPUdi() T& operator()(size_t i, size_t j)
  {
    return mArray[offset(i, j)];
  }

  GPUdi() T const& operator()(size_t i, size_t j) const
  {
    return mArray[offset(i, j)];
  }

  GPUdi() T& operator[](size_t i)
  {
    return mArray[off(i)];
  }

  GPUdi() T const& operator[](size_t i) const
  {
    return mArray[off(i)];
  }

  GPUdi() T apply(size_t i) const
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
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] = rhs.Array()[i];
    return *this;
  }

  // // self addition : only symmetric to symmetric allowed
  // template <class R>
  // GPUdi() MatRepSymGPU<T, D>& operator+=(const R&)
  // {
  //   GPU_STATIC_CHECK(0 == 1, Check_symmetric_matrices_sum);
  //   return *this;
  // }
  // GPUdi() MatRepSymGPU<T, D>& operator+=(const MatRepSymGPU& rhs)
  // {
  //   for (size_t i = 0; i < kSize; ++i)
  //     mArray[i] += rhs.Array()[i];
  //   return *this;
  // }

  // // self subtraction : only symmetric to symmetric allowed
  // template <class R>
  // GPUdi() MatRepSymGPU<T, D>& operator-=(const R&)
  // {
  //   GPU_STATIC_CHECK(0 == 1, Check_symmetric_matrices_subtraction);
  //   return *this;
  // }
  // GPUdi() MatRepSymGPU<T, D>& operator-=(const MatRepSymGPU& rhs)
  // {
  //   for (size_t i = 0; i < kSize; ++i)
  //     mArray[i] -= rhs.Array()[i];
  //   return *this;
  // }

  // template <class R>
  // GPUdi() bool operator==(const R& rhs) const
  // {
  //   bool rc = true;
  //   for (size_t i = 0; i < D * D; ++i) {
  //     rc = rc && (operator[](i) == rhs[i]);
  //   }
  //   return rc;
  // }

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

  static GPUdi() constexpr size_t
    offset(size_t i, size_t j)
  {
    return off(i * D + j);
  }

 private:
  T mArray[kSize];
};

/// SMatReprStd starting port here
template <class T, size_t D1, size_t D2 = D1>
class MatRepStdGPU
{
 public:
  typedef T value_type;
  GPUdi() MatRepStdGPU(){};
  GPUdi() const T& operator()(size_t i, size_t j) const
  {
    return mArray[i * D2 + j];
  }
  GPUdi() T& operator()(size_t i, size_t j)
  {
    return mArray[i * D2 + j];
  }
  GPUdi() T& operator[](size_t i) { return mArray[i]; }
  GPUdi() const T& operator[](size_t i) const { return mArray[i]; }
  GPUdi() T apply(size_t i) const { return mArray[i]; }
  GPUdi() T* Array() { return mArray; }
  GPUdi() const T* Array() const { return mArray; }
  //
  //   template <class R>
  //   GPUdi() MatRepStdGPU<T, D1, D2>& operator+=(const R& rhs)
  //   {
  //     for (size_t i = 0; i < kSize; ++i)
  //       mArray[i] += rhs[i];
  //     return *this;
  //   }
  // //
  //   template <class R>
  //   GPUdi() MatRepStdGPU<T, D1, D2>& operator-=(const R& rhs)
  //   {
  //     for (size_t i = 0; i < kSize; ++i)
  //       mArray[i] -= rhs[i];
  //     return *this;
  //   }
  //
  template <class R>
  GPUdi() MatRepStdGPU<T, D1, D2>& operator=(const R& rhs)
  {
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] = rhs[i];
    return *this;
  }
  template <class R>
  GPUdi() bool operator==(const R& rhs) const
  {
    bool rc = true;
    for (size_t i = 0; i < kSize; ++i) {
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

// template <class ExprType, class T, size_t D, size_t D2 = 1, class R1 = MatRepStdGPU<T, D, D2>>
// class Expr
// {
//  public:
//   typedef T value_type;
//   GPUd() Expr(const ExprType& rhs) : mRhs(rhs) {}
//   GPUd() ~Expr() {}
//   GPUdi() T apply(size_t i) const
//   {
//     return mRhs.apply(i);
//   }
//   GPUdi() T operator()(size_t i, unsigned j) const
//   {
//     return mRhs(i, j);
//   }
//   GPUdi() bool IsInUse(const T* p) const
//   {
//     return mRhs.IsInUse(p);
//   }

//   enum {
//     kRows = D,
//     kCols = D2
//   };

//  private:
//   ExprType mRhs;
// };

template <class T, size_t D1, size_t D2 = D1, class R = MatRepStdGPU<T, D1, D2>>
class SMatrixGPU
{
 public:
  typedef T value_type;
  typedef R rep_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  GPUd() SMatrixGPU(){};
  GPUdi() SMatrixGPU(SMatrixNoInit) {}
  SMatrixGPU(SMatrixIdentity);
  GPUd() SMatrixGPU(const SMatrixGPU<T, D1, D2, R>& rhs);
  // template <class R2>
  // SMatrixGPU(const SMatrixGPU<T, D1, D2, R2>& rhs);
  // template <class A, class R2>
  // GPUd() SMatrixGPU(const Expr<A, T, D1, D2, R2>& rhs);
  // template <class InputIterator>
  // SMatrixGPU(InputIterator begin, InputIterator end, bool triang = false, bool lower = true);
  // template <class InputIterator>
  // SMatrixGPU(InputIterator begin, size_t size, bool triang = false, bool lower = true);
  // template <size_t N>
  // SMatrixGPU(const SVector<T, N>& v, bool lower = true);
  // explicit SMatrixGPU(const T& rhs);
  // template <class M>
  // SMatrixGPU<T, D1, D2, R>& operator=(const M& rhs);
  // template <class A, class R2>
  // GPUd() SMatrixGPU<T, D1, D2, R>& operator=(const Expr<A, T, D1, D2, R2>& rhs);
  // SMatrixGPU<T, D1, D2, R>& operator=(SMatrixIdentity);
  // SMatrixGPU<T, D1, D2, R>& operator=(const T& rhs);
  enum {
    kRows = D1,     // rows
    kCols = D2,     // columns
    kSize = D1 * D2 // rows*columns
  };
  T apply(size_t i) const;
  GPUd() const T* Array() const;
  T* Array();
  iterator begin();
  iterator end();
  // const_iterator begin() const;
  // const_iterator end() const;
  // template <class InputIterator>
  // void SetElements(InputIterator begin, InputIterator end, bool triang = false, bool lower = true);
  // template <class InputIterator>
  // void SetElements(InputIterator begin, size_t size, bool triang = false, bool lower = true);
  // /// element wise comparisons
  // bool operator==(const T& rhs) const;
  // bool operator!=(const T& rhs) const;
  // template <class R2>
  // bool operator==(const SMatrixGPU<T, D1, D2, R2>& rhs) const;
  // bool operator!=(const SMatrixGPU<T, D1, D2, R>& rhs) const;
  // template <class A, class R2>
  // bool operator==(const Expr<A, T, D1, D2, R2>& rhs) const;
  // template <class A, class R2>
  // bool operator!=(const Expr<A, T, D1, D2, R2>& rhs) const;
  //
  // bool operator>(const T& rhs) const;
  // bool operator<(const T& rhs) const;
  // template <class R2>
  // bool operator>(const SMatrixGPU<T, D1, D2, R2>& rhs) const;
  // template <class R2>
  // bool operator<(const SMatrixGPU<T, D1, D2, R2>& rhs) const;
  // template <class A, class R2>
  // bool operator>(const Expr<A, T, D1, D2, R2>& rhs) const;
  // template <class A, class R2>
  // bool operator<(const Expr<A, T, D1, D2, R2>& rhs) const;
  GPUd() const T& operator()(size_t i, size_t j) const;
  GPUd() T& operator()(size_t i, size_t j);
  // const T& At(size_t i, size_t j) const;
  // T& At(size_t i, size_t j);
  //
  class SMatrixRowGPU
  {
   public:
    GPUd() SMatrixRowGPU(SMatrixGPU<T, D1, D2, R>& rhs, size_t i) : mMat(&rhs), mRow(i)
    {
    }
    GPUd() T& operator[](int j) { return (*mMat)(mRow, j); }
    //
   private:
    SMatrixGPU<T, D1, D2, R>* mMat;
    size_t mRow;
  };
  //
  class SMatrixRowGPUconst
  {
   public:
    GPUd() SMatrixRowGPUconst(const SMatrixGPU<T, D1, D2, R>& rhs, size_t i) : mMat(&rhs), mRow(i)
    {
    }
    //
    GPUd() const T& operator[](int j) const { return (*mMat)(mRow, j); }
    //
   private:
    const SMatrixGPU<T, D1, D2, R>* mMat;
    size_t mRow;
  };
  //
  GPUd() SMatrixRowGPUconst operator[](size_t i) const { return SMatrixRowGPUconst(*this, i); }
  GPUd() SMatrixRowGPU operator[](size_t i) { return SMatrixRowGPU(*this, i); }
  // SMatrixGPU<T, D1, D2, R>& operator+=(const T& rhs);
  // template <class R2>
  // SMatrixGPU<T, D1, D2, R>& operator+=(const SMatrixGPU<T, D1, D2, R2>& rhs);
  // // template <class A, class R2>
  // // SMatrixGPU<T, D1, D2, R>& operator+=(const Expr<A, T, D1, D2, R2>& rhs);
  // SMatrixGPU<T, D1, D2, R>& operator-=(const T& rhs);
  // template <class R2>
  // SMatrixGPU<T, D1, D2, R>& operator-=(const SMatrixGPU<T, D1, D2, R2>& rhs);
  // // template <class A, class R2>
  // // SMatrixGPU<T, D1, D2, R>& operator-=(const Expr<A, T, D1, D2, R2>& rhs);
  // SMatrixGPU<T, D1, D2, R>& operator*=(const T& rhs);
  // template <class R2>
  // SMatrixGPU<T, D1, D2, R>& operator*=(const SMatrixGPU<T, D1, D2, R2>& rhs);
  // template <class A, class R2>
  // GPUd() SMatrixGPU<T, D1, D2, R>& operator*=(const Expr<A, T, D1, D2, R2>& rhs);
  // SMatrixGPU<T, D1, D2, R>& operator/=(const T& rhs);
  // GPUd() bool Invert();
  // SMatrixGPU<T, D1, D2, R> Inverse(int& ifail) const;
  // bool InvertFast();
  // SMatrixGPU<T, D1, D2, R> InverseFast(int& ifail) const;
  // bool InvertChol();
  // SMatrixGPU<T, D1, D2, R> InverseChol(int& ifail) const;
  // bool Det(T& det);
  // bool Det2(T& det) const;
  // template <size_t D>
  // SMatrixGPU<T, D1, D2, R>& Place_in_row(const SVector<T, D>& rhs,
  //                                     size_t row,
  //                                     size_t col);
  // // place a vector expression in a Matrix row
  // template <class A, size_t D>
  // SMatrixGPU<T, D1, D2, R>& Place_in_row(const VecExpr<A, T, D>& rhs,
  //                                     size_t row,
  //                                     size_t col);
  // // place a vector in a Matrix column
  // template <size_t D>
  // SMatrixGPU<T, D1, D2, R>& Place_in_col(const SVector<T, D>& rhs,
  //                                     size_t row,
  //                                     size_t col);
  // // place a vector expression in a Matrix column
  // template <class A, size_t D>
  // SMatrixGPU<T, D1, D2, R>& Place_in_col(const VecExpr<A, T, D>& rhs,
  //                                     size_t row,
  //                                     size_t col);
  // // place a matrix in this matrix
  // template <size_t D3, size_t D4, class R2>
  // SMatrixGPU<T, D1, D2, R>& Place_at(const SMatrixGPU<T, D3, D4, R2>& rhs,
  //                                 size_t row,
  //                                 size_t col);
  // // place a matrix expression in this matrix
  // template <class A, size_t D3, size_t D4, class R2>
  // SMatrixGPU<T, D1, D2, R>& Place_at(const Expr<A, T, D3, D4, R2>& rhs,
  //                                 size_t row,
  //                                 size_t col);
  //
  // SVector<T, D2> Row(size_t therow) const;
  // SVector<T, D1> Col(size_t thecol) const;
  // template <class SubVector>
  // SubVector SubRow(size_t therow, size_t col0 = 0) const;
  // template <class SubVector>
  // SubVector SubCol(size_t thecol, size_t row0 = 0) const;
  // template <class SubMatrix>
  // SubMatrix Sub(size_t row0, size_t col0) const;
  // SVector<T, D1> Diagonal() const;
  // template <class Vector>
  // void SetDiagonal(const Vector& v);
  // T Trace() const;
  // template <class SubVector>
  // SubVector UpperBlock() const;
  // template <class SubVector>
  // SubVector LowerBlock() const;
  // bool IsInUse(const T* p) const;
  //
  // void Print() const;
  //
 public:
  R mRep;
};

// template <class T, size_t D1, size_t D2, class A, class R1, class R2>
// struct Assign {
//   // Evaluate the expression from general to general matrices.
//   GPUd() static void Evaluate(SMatrixGPU<T, D1, D2, R1>& lhs, const Expr<A, T, D1, D2, R2>& rhs)
//   {
//     if (!rhs.IsInUse(lhs.begin())) {
//       size_t l = 0;
//       for (size_t i = 0; i < D1; ++i)
//         for (size_t j = 0; j < D2; ++j) {
//           lhs.fRep[l] = rhs(i, j);
//           l++;
//         }
//     } else {
//       T tmp[D1 * D2];
//       size_t l = 0;
//       for (size_t i = 0; i < D1; ++i)
//         for (size_t j = 0; j < D2; ++j) {
//           tmp[l] = rhs(i, j);
//           l++;
//         }

//       for (size_t i = 0; i < D1 * D2; ++i)
//         lhs.fRep[i] = tmp[i];
//     }
//   }
// };

// template <class T, size_t D1, size_t D2, class A>
// struct Assign<T, D1, D2, A, MatRepSymGPU<T, D1>, MatRepSymGPU<T, D1>> {
//   // Evaluate the expression from  symmetric to symmetric matrices.
//   GPUd() static void Evaluate(SMatrixGPU<T, D1, D2, MatRepSymGPU<T, D1>>& lhs,
//                               const Expr<A, T, D1, D2, MatRepSymGPU<T, D1>>& rhs)
//   {
//     if (!rhs.IsInUse(lhs.begin())) {
//       size_t l = 0;
//       for (size_t i = 0; i < D1; ++i)
//         // storage of symmetric matrix is in lower block
//         for (size_t j = 0; j <= i; ++j) {
//           lhs.fRep.Array()[l] = rhs(i, j);
//           l++;
//         }
//     } else {
//       T tmp[MatRepSymGPU<T, D1>::kSize];
//       size_t l = 0;
//       for (size_t i = 0; i < D1; ++i)
//         for (size_t j = 0; j <= i; ++j) {
//           tmp[l] = rhs(i, j);
//           l++;
//         }
//       for (size_t i = 0; i < MatRepSymGPU<T, D1>::kSize; ++i)
//         lhs.fRep.Array()[i] = tmp[i];
//     }
//   }
// };

// // avoid assigment from expression based on a general matrix to a symmetric matrix
// template <class T, size_t D1, size_t D2, class A>
// struct Assign<T, D1, D2, A, MatRepSymGPU<T, D1>, MatRepStdGPU<T, D1, D2>> {
//   GPUd() static void Evaluate(SMatrixGPU<T, D1, D2, MatRepSymGPU<T, D1>>&,
//                               const Expr<A, T, D1, D2, MatRepStdGPU<T, D1, D2>>&)
//   {
//     GPU_STATIC_CHECK(0 == 1, Check_general_to_symmetric_matrix_assignment);
//   }
// };

// // Force Expression evaluation from general to symmetric
// struct AssignSym {
//   // assign a symmetric matrix from an expression
//   template <class T,
//             size_t D,
//             class A,
//             class R>
//   GPUd() static void Evaluate(SMatrixGPU<T, D, D, MatRepSymGPU<T, D>>& lhs, const Expr<A, T, D, D, R>& rhs)
//   {
//     size_t l = 0;
//     for (size_t i = 0; i < D; ++i)
//       for (size_t j = 0; j <= i; ++j) {
//         lhs.fRep.Array()[l] = rhs(i, j);
//         l++;
//       }
//   }

//   // assign the symmetric matric  from a general matrix
//   template <class T,
//             size_t D,
//             class R>
//   GPUd() static void Evaluate(SMatrixGPU<T, D, D, MatRepSymGPU<T, D>>& lhs, const SMatrixGPU<T, D, D, R>& rhs)
//   {
//     size_t l = 0;
//     for (size_t i = 0; i < D; ++i)
//       for (size_t j = 0; j <= i; ++j) {
//         lhs.fRep.Array()[l] = rhs(i, j);
//         l++;
//       }
//   }
// };

// template <class T, size_t D1, size_t D2, class R>
// template <class A, class R2>
// GPUdi() SMatrixGPU<T, D1, D2, R>::SMatrixGPU(const Expr<A, T, D1, D2, R2>& rhs)
// {
//   operator=(rhs);
// }

template <class T, size_t D1, size_t D2, class R>
GPUdi() const T& SMatrixGPU<T, D1, D2, R>::operator()(size_t i, size_t j) const
{
  return mRep(i, j);
}

template <class T, size_t D1, size_t D2, class R>
GPUdi() T& SMatrixGPU<T, D1, D2, R>::operator()(size_t i, size_t j)
{
  return mRep(i, j);
}

// template <class T, size_t D1, size_t D2, class R>
// template <class A, class R2>
// GPUdi() SMatrixGPU<T, D1, D2, R>& SMatrixGPU<T, D1, D2, R>::operator=(const Expr<A, T, D1, D2, R2>& rhs)
// {
//   Assign<T, D1, D2, A, R, R2>::Evaluate(*this, rhs);
//   return *this;
// }

// template <class T, class R1, class R2>
// struct MultPolicyGPU {
//   enum {
//     N1 = R1::kRows,
//     N2 = R2::kCols
//   };
//   typedef MatRepStdGPU<T, N1, N2> RepType;
// };

// template <class T, size_t D1, size_t D, size_t D2, class R1, class R2>
// GPUdi() Expr<MatrixMulOpGPU<SMatrixGPU<T, D1, D, R1>, SMatrixGPU<T, D, D2, R2>, T, D>, T, D1, D2, typename MultPolicyGPU<T, R1, R2>::RepType>
//   operator*(const SMatrixGPU<T, D1, D, R1>& lhs, const SMatrixGPU<T, D, D2, R2>& rhs)
// {
//   typedef MatrixMulOpGPU<SMatrixGPU<T, D1, D, R1>, SMatrixGPU<T, D, D2, R2>, T, D> MatMulOp;
//   return Expr<MatMulOp, T, D1, D2,
//               typename MultPolicyGPU<T, R1, R2>::RepType>(MatMulOp(lhs, rhs));
// }

// template <class T, size_t D1, size_t D2, class R>
// struct TranspPolicyGPU {
//   enum {
//     N1 = R::kRows,
//     N2 = R::kCols
//   };
//   typedef MatRepStdGPU<T, N2, N1> RepType;
// };

// template <class T, size_t D1, size_t D2>
// struct TranspPolicyGPU<T, D1, D2, MatRepSymGPU<T, D1>> {
//   typedef MatRepSymGPU<T, D1> RepType;
// };

// template <class Matrix, class T, size_t D1, size_t D2 = D1>
// class TransposeOpGPU
// {
//  public:
//   GPUd() TransposeOpGPU(const Matrix& rhs) : mRhs(rhs) {}

//   ~TransposeOpGPU() {}

//   GPUdi() T apply(size_t i) const
//   {
//     return mRhs.apply((i % D1) * D2 + i / D1);
//   }
//   GPUdi() T operator()(size_t i, unsigned j) const
//   {
//     return mRhs(j, i);
//   }

//   GPUdi() bool IsInUse(const T* p) const
//   {
//     return mRhs.IsInUse(p);
//   }

//  protected:
//   const Matrix& mRhs;
// };

// template <class T, size_t D1, size_t D2, class R>
// GPUdi() Expr<TransposeOpGPU<SMatrixGPU<T, D1, D2, R>, T, D1, D2>, T, D2, D1, typename TranspPolicyGPU<T, D1, D2, R>::RepType> Transpose(const SMatrixGPU<T, D1, D2, R>& rhs)
// {
//   typedef TransposeOpGPU<SMatrixGPU<T, D1, D2, R>, T, D1, D2> MatTrOp;
//   return Expr<MatTrOp, T, D2, D1, typename TranspPolicyGPU<T, D1, D2, R>::RepType>(MatTrOp(rhs));
// }

// template <class T, size_t D1, size_t D2, class R>
// GPUdi() SMatrixGPU<T, D1, D1, MatRepSymGPU<T, D1>> Similarity(const SMatrixGPU<T, D1, D2, R>& lhs, const SMatrixGPU<T, D2, D2, MatRepSymGPU<T, D2>>& rhs)
// {
//   SMatrixGPU<T, D1, D2, MatRepStdGPU<T, D1, D2>> tmp = lhs * rhs;
//   typedef SMatrixGPU<T, D1, D1, MatRepSymGPU<T, D1>> SMatrixSym;
//   SMatrixSym mret;
//   AssignSym::Evaluate(mret, tmp * Transpose(lhs));
//   return mret;
// }

// /// SMatrixGPU end port here

// Define aliases
template <typename T, size_t N>
using SVector = SVectorGPU<T, N>;

template <class T, size_t D>
using MatRepSym = MatRepSymGPU<T, D>;

template <class T, size_t D1, size_t D2 = D1>
using MatRepStd = MatRepStdGPU<T, D1, D2>;

template <class T, size_t D1, size_t D2 = D1, class R = MatRepStdGPU<T, D1, D2>>
using SMatrix = SMatrixGPU<T, D1, D2, R>;

#else
template <typename T, size_t N>
using SVector = ROOT::Math::SVector<T, N>;

template <class T, size_t D1, size_t D2, class R>
using SMatrix = ROOT::Math::SMatrix<T, D1, D2, R>;

template <class T, size_t D>
using MatRepSym = ROOT::Math::MatRepSym<T, D>;

template <class T, size_t D>
using MatRepStd = ROOT::Math::MatRepStd<T, D>;

#endif
}; // namespace o2::math_utils

#endif