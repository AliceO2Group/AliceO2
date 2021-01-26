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
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#endif

#include "GPUCommonDef.h"
#include "GPUCommonArray.h"

namespace o2::math_utils
{
#ifdef GPUCA_GPUCODE_DEVICE

template <bool>
struct Check {
  Check(void*) {}
};
template <>
struct Check<false> {
};

#define STATIC_CHECK(expr, msg)     \
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
  GPUdDefault() SVectorGPU();
  GPUd() SVectorGPU(const SVectorGPU<T, N>& rhs);

  GPUd() const T& operator[](size_t i) const;
  GPUd() const T& operator()(size_t i) const;
  GPUd() T& operator[](size_t i);
  GPUd() T& operator()(size_t i);

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
GPUdDefault() SVectorGPU<T, N>::SVectorGPU()
{
  for (size_t i = 0; i < N; ++i) {
    mArray[i] = 0;
  }
}

template <class T, size_t N>
GPUd() SVectorGPU<T, N>::SVectorGPU(const SVectorGPU<T, N>& rhs)
{
  for (size_t i = 0; i < N; ++i) {
    mArray[i] = rhs.mArray[i];
  }
}

// utils for SMatrixGPU
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
constexpr gpu::gpustd::array<F, sizeof...(I)>
  do_make(F f, indices<I...>)
{
  gpu::gpustd::array<F, sizeof...(I)> retarr;
  retarr.m_internal_V__ = {f(I0 + I)...};
  return retarr;
}

template <int N, int I0 = 0, class F>
constexpr gpu::gpustd::array<F, N>
  make(F f)
{
  return do_make<I0>(f, typename make_indices<N>::type());
}
} // namespace row_offsets_utils

// Symm representation
template <class T, size_t D>
class MatRepSymGPU
{
 public:
  inline MatRepSymGPU() {}
  typedef T value_type;

  inline T& operator()(size_t i, size_t j)
  {
    return mArray[offset(i, j)];
  }

  inline T const& operator()(size_t i, size_t j) const
  {
    return mArray[offset(i, j)];
  }

  inline T& operator[](size_t i)
  {
    return mArray[off(i)];
  }

  inline T const& operator[](size_t i) const
  {
    return mArray[off(i)];
  }

  inline T apply(size_t i) const
  {
    return mArray[off(i)];
  }

  inline T* Array() { return mArray; }

  inline const T* Array() const { return mArray; }

  // assignment: only symmetric to symmetric allowed
  template <class R>
  inline MatRepSymGPU<T, D>& operator=(const R&)
  {
    STATIC_CHECK(0 == 1, Check_symmetric_matrices_equivalence);
    return *this;
  }
  inline MatRepSymGPU<T, D>& operator=(const MatRepSymGPU& rhs)
  {
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] = rhs.Array()[i];
    return *this;
  }

  // self addition : only symmetric to symmetric allowed
  template <class R>
  inline MatRepSymGPU<T, D>& operator+=(const R&)
  {
    STATIC_CHECK(0 == 1, Check_symmetric_matrices_sum);
    return *this;
  }
  inline MatRepSymGPU<T, D>& operator+=(const MatRepSymGPU& rhs)
  {
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] += rhs.Array()[i];
    return *this;
  }

  // self subtraction : only symmetric to symmetric allowed
  template <class R>
  inline MatRepSymGPU<T, D>& operator-=(const R&)
  {
    STATIC_CHECK(0 == 1, Check_symmetric_matrices_subtraction);
    return *this;
  }
  inline MatRepSymGPU<T, D>& operator-=(const MatRepSymGPU& rhs)
  {
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] -= rhs.Array()[i];
    return *this;
  }

  template <class R>
  inline bool operator==(const R& rhs) const
  {
    bool rc = true;
    for (size_t i = 0; i < D * D; ++i) {
      rc = rc && (operator[](i) == rhs[i]);
    }
    return rc;
  }

  enum {
    kRows = D,              // rows
    kCols = D,              // columns
    kSize = D * (D + 1) / 2 // rows*columns
  };

  static constexpr int off0(int i) { return i == 0 ? 0 : off0(i - 1) + i; }
  static constexpr int off2(int i, int j) { return j < i ? off0(i) + j : off0(j) + i; }
  static constexpr int off1(int i) { return off2(i / D, i % D); }

  static int off(int i)
  {
    static constexpr auto v = row_offsets_utils::make<D * D>(off1);
    return v[i];
  }

  static inline constexpr size_t
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

  inline const T& operator()(size_t i, size_t j) const
  {
    return mArray[i * D2 + j];
  }
  inline T& operator()(size_t i, size_t j)
  {
    return mArray[i * D2 + j];
  }
  inline T& operator[](size_t i) { return mArray[i]; }

  inline const T& operator[](size_t i) const { return mArray[i]; }

  inline T apply(size_t i) const { return mArray[i]; }

  inline T* Array() { return mArray; }

  inline const T* Array() const { return mArray; }

  template <class R>
  inline MatRepStdGPU<T, D1, D2>& operator+=(const R& rhs)
  {
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] += rhs[i];
    return *this;
  }

  template <class R>
  inline MatRepStdGPU<T, D1, D2>& operator-=(const R& rhs)
  {
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] -= rhs[i];
    return *this;
  }

  template <class R>
  inline MatRepStdGPU<T, D1, D2>& operator=(const R& rhs)
  {
    for (size_t i = 0; i < kSize; ++i)
      mArray[i] = rhs[i];
    return *this;
  }

  template <class R>
  inline bool operator==(const R& rhs) const
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

template <class T,
          size_t D1,
          size_t D2 = D1,
          class R = MatRepStdGPU<T, D1, D2>>
class SMatrixGPU
{
 public:
  typedef T value_type;
  typedef R rep_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  SMatrixGPU();
  // inline SMatrixGPU(SMatrixNoInit) {}
  // SMatrixGPU(SMatrixIdentity);
  // SMatrixGPU(const SMatrixGPU<T, D1, D2, R>& rhs);
  // template <class R2>
  // SMatrixGPU(const SMatrixGPU<T, D1, D2, R2>& rhs);
  // template <class A, class R2>
  // SMatrixGPU(const Expr<A, T, D1, D2, R2>& rhs);
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
  // SMatrixGPU<T, D1, D2, R>& operator=(const Expr<A, T, D1, D2, R2>& rhs);
  // SMatrixGPU<T, D1, D2, R>& operator=(SMatrixIdentity);
  // SMatrixGPU<T, D1, D2, R>& operator=(const T& rhs);
  // enum class SMatrixKeys{
  //   kRows = D1, // rows
  //   kCols = D2, // columns
  //   kSize = D1 * D2 // rows*columns
  // };
  // T apply(size_t i) const;
  // const T* Array() const;
  // T* Array();
  // iterator begin();
  // iterator end();
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
  const T& operator()(size_t i, size_t j) const;
  T& operator()(size_t i, size_t j);
  // const T& At(size_t i, size_t j) const;
  // T& At(size_t i, size_t j);

  // class SMatrixRow
  // {
  //  public:
  //   SMatrixRow(SMatrixGPU<T, D1, D2, R>& rhs, size_t i) : mMat(&rhs), mRow(i)
  //   {
  //   }
  //   T& operator[](int j) { return (*mMat)(mRow, j); }

  //  private:
  //   SMatrixGPU<T, D1, D2, R>* mMat;
  //   size_t mRow;
  // };

  // class SMatrixRow_const
  // {
  //  public:
  //   SMatrixRow_const(const SMatrixGPU<T, D1, D2, R>& rhs, size_t i) : mMat(&rhs), mRow(i)
  //   {
  //   }

  //   const T& operator[](int j) const { return (*mMat)(mRow, j); }

  //  private:
  //   const SMatrixGPU<T, D1, D2, R>* mMat;
  //   size_t mRow;
  // };

  // SMatrixRow_const operator[](size_t i) const { return SMatrixRow_const(*this, i); }
  // SMatrixRow operator[](size_t i) { return SMatrixRow(*this, i); }
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
  // // template <class A, class R2>
  // // SMatrixGPU<T, D1, D2, R>& operator*=(const Expr<A, T, D1, D2, R2>& rhs);
  // SMatrixGPU<T, D1, D2, R>& operator/=(const T& rhs);
  // bool Invert();
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

  // void Print() const;

 public:
  R mRep;

}; // end of class SMatrixGPU

template <class T, size_t D1, size_t D2, class R>
inline const T& SMatrixGPU<T, D1, D2, R>::operator()(size_t i, size_t j) const
{
  return mRep(i, j);
}

template <class T, size_t D1, size_t D2, class R>
inline T& SMatrixGPU<T, D1, D2, R>::operator()(size_t i, size_t j)
{
  return mRep(i, j);
}
/// SMatrixGPU end port here

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