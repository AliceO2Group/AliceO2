// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CartesianGPU.h
/// @author David Rohr

#ifndef ALICEO2_CARTESIANGPU_H
#define ALICEO2_CARTESIANGPU_H

#ifndef GPUCA_GPUCODE_DEVICE
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#endif

#include <cstddef>
#include "GPUCommonDef.h"

namespace o2::math_utils
{

namespace detail
{
template <typename T, int I>
struct GPUPoint2D {
  GPUdDefault() GPUPoint2D() = default;
  GPUd() GPUPoint2D(T a, T b) : xx(a), yy(b) {}
  GPUd() float X() const { return xx; }
  GPUd() float Y() const { return yy; }
  GPUd() float R() const { return o2::gpu::CAMath::Sqrt(xx * xx + yy * yy); }
  GPUd() void SetX(float v) { xx = v; }
  GPUd() void SetY(float v) { yy = v; }
  T xx;
  T yy;
};

template <typename T, int I>
struct GPUPoint3D : public GPUPoint2D<T, I> {
  GPUdDefault() GPUPoint3D() = default;
  GPUd() GPUPoint3D(T a, T b, T c) : GPUPoint2D<T, I>(a, b), zz(c) {}
  GPUd() float Z() const { return zz; }
  GPUd() float R() const { return o2::gpu::CAMath::Sqrt(GPUPoint2D<T, I>::xx * GPUPoint2D<T, I>::xx + GPUPoint2D<T, I>::yy * GPUPoint2D<T, I>::yy + zz * zz); }
  GPUd() void SetZ(float v) { zz = v; }
  T zz;
};
} // namespace detail

#ifdef GPUCA_GPUCODE_DEVICE
template <typename T, size_t N>
class GPUSVector
{
 public:
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  GPUSVector();
  //   template <class A>
  //   SVector(const VecExpr<A, T, D>& rhs);
  GPUSVector(const GPUSVector<T, N>& rhs);

 private:
  T mArray[N];
};

template <class T, size_t D>
GPUSVector<T, D>::GPUSVector()
{
  for (size_t i = 0; i < D; ++i) {
    mArray[i] = 0;
  }
}

template <class T, size_t D>
GPUSVector<T, D>::GPUSVector(const GPUSVector<T, D>& rhs)
{
  for (size_t i = 0; i < D; ++i) {
    mArray[i] = rhs.mArray[i];
  }
}

// namespace rowOffsetsUtils
// {

// template <int...>
// struct indices {
// };

// template <int I, class IndexTuple, int N>
// struct make_indices_impl;

// template <int I, int... Indices, int N>
// struct make_indices_impl<I, indices<Indices...>, N> {
//   typedef typename make_indices_impl<I + 1, indices<Indices..., I>,
//                                      N>::type type;
// };

// template <int N, int... Indices>
// struct make_indices_impl<N, indices<Indices...>, N> {
//   typedef indices<Indices...> type;
// };

// template <int N>
// struct make_indices : make_indices_impl<0, indices<>, N> {
// };
// // end of stuff

// template <int I0, class F, int... I>
// constexpr gpustd::array<decltype(std::declval<F>()(std::declval<int>())), sizeof...(I)>
//   do_make(F f, indices<I...>)
// {
//   return gpustd::array<decltype(std::declval<F>()(std::declval<int>())),
//                     sizeof...(I)>{{f(I0 + I)...}};
// }

// template <int N, int I0 = 0, class F>
// constexpr gpustd::array<decltype(std::declval<F>()(std::declval<int>())), N>
//   make(F f)
// {
//   return do_make<I0>(f, typename make_indices<N>::type());
// }

// } // namespace rowOffsetsUtils

// template <class T, size_t D>
// class MatRepSym
// {
//  public:
//   inline MatRepSym() {}
//   typedef T value_type;

//   inline T& operator()(size_t i, size_t j)
//   {
//     return mArray[offset(i, j)];
//   }

//   inline T const& operator()(size_t i, size_t j) const
//   {
//     return mArray[offset(i, j)];
//   }

//   inline T& operator[](size_t i)
//   {
//     return mArray[off(i)];
//   }

//   inline T const& operator[](size_t i) const
//   {
//     return mArray[off(i)];
//   }

//   inline T apply(size_t i) const
//   {
//     return mArray[off(i)];
//   }

//   inline T* Array() { return mArray; }

//   inline const T* Array() const { return mArray; }

//   /**
//           assignment : only symmetric to symmetric allowed
//         */
//   // template <class R>
//   // inline MatRepSym<T, D>& operator=(const R&)
//   // {
//   //   STATIC_CHECK(0 == 1,
//   //                Cannot_assign_general_to_symmetric_matrix_representation);
//   //   return *this;
//   // }
//   inline MatRepSym<T, D>& operator=(const MatRepSym& rhs)
//   {
//     for (size_t i = 0; i < mSize; ++i)
//       mArray[i] = rhs.Array()[i];
//     return *this;
//   }

//   /**
//           self addition : only symmetric to symmetric allowed
//         */
//   // template <class R>
//   // inline MatRepSym<T, D>& operator+=(const R&)
//   // {
//   //   STATIC_CHECK(0 == 1,
//   //                Cannot_add_general_to_symmetric_matrix_representation);
//   //   return *this;
//   // }
//   inline MatRepSym<T, D>& operator+=(const MatRepSym& rhs)
//   {
//     for (size_t i = 0; i < mSize; ++i)
//       mArray[i] += rhs.Array()[i];
//     return *this;
//   }

//   // /**
//   //         self subtraction : only symmetric to symmetric allowed
//   //       */
//   // template <class R>
//   // inline MatRepSym<T, D>& operator-=(const R&)
//   // {
//   //   STATIC_CHECK(0 == 1,
//   //                Cannot_substract_general_to_symmetric_matrix_representation);
//   //   return *this;
//   // }
//   inline MatRepSym<T, D>& operator-=(const MatRepSym& rhs)
//   {
//     for (size_t i = 0; i < mSize; ++i)
//       mArray[i] -= rhs.Array()[i];
//     return *this;
//   }

//   template <class R>
//   inline bool operator==(const R& rhs) const
//   {
//     bool rc = true;
//     for (size_t i = 0; i < D * D; ++i) {
//       rc = rc && (operator[](i) == rhs[i]);
//     }
//     return rc;
//   }

//   enum {
//     /// return no. of matrix rows
//     kRows = D,
//     /// return no. of matrix columns
//     kCols = D,
//     /// return no of elements: rows*columns
//     mSize = D * (D + 1) / 2
//   };

//   static constexpr int off0(int i) { return i == 0 ? 0 : off0(i - 1) + i; }
//   static constexpr int off2(int i, int j) { return j < i ? off0(i) + j : off0(j) + i; }
//   static constexpr int off1(int i) { return off2(i / D, i % D); }

//   static int off(int i)
//   {
//     static constexpr auto v = rowOffsetsUtils::make<D * D>(off1);
//     return v[i];
//   }

//   static inline constexpr size_t
//     offset(size_t i, size_t j)
//   {
//     //if (j > i) std::swap(i, j);
//     return off(i * D + j);
//     // return (i>j) ? (i * (i+1) / 2) + j :  (j * (j+1) / 2) + i;
//   }

//  private:
//   //T __attribute__ ((aligned (16))) mArray[mSize];
//   T mArray[mSize];
// };

template <typename T, size_t N>
using SVector = GPUSVector<T, N>;
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

} // end namespace o2::math_utils

#endif
