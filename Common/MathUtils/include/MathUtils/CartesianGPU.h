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
  SVector(const SVector<T, D>& rhs);

  //   // new constructs using STL iterator interface
  //   // skip - need to solve the ambiguities
  // #ifdef LATER
  //   /**
  //         Constructor with STL iterator interface. The data will be copied into the vector
  //         The iterator size must be equal to the vector size
  //      */
  //   template <class InputIterator>
  //   explicit SVector(InputIterator begin, InputIterator end);

  //   /**
  //        Constructor with STL iterator interface. The data will be copied into the vector
  //        The size must be <= vector size
  //      */
  //   template <class InputIterator>
  //   explicit SVector(InputIterator begin, size_t size);

  // #else
  //   // if you use iterator this is not necessary

  //   /// fill from array with len must be equal to D!
  //   SVector(const T* a, size_t len);

  //   /** fill from a SVector iterator of type T*
  //        (for ambiguities iterator cannot be generic )
  //     */
  //   SVector(const_iterator begin, const_iterator end);

  // #endif
  //   /// construct a vector of size 1 from a single scalar value
  //   explicit SVector(const T& a1);
  //   /// construct a vector of size 2 from 2 scalar values
  //   SVector(const T& a1, const T& a2);
  //   /// construct a vector of size 3 from 3 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3);
  //   /// construct a vector of size 4 from 4 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3, const T& a4);
  //   /// construct a vector of size 5 from 5 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3, const T& a4,
  //           const T& a5);
  //   /// construct a vector of size 6 from 6 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3, const T& a4,
  //           const T& a5, const T& a6);
  //   /// construct a vector of size 7 from 7 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3, const T& a4,
  //           const T& a5, const T& a6, const T& a7);
  //   /// construct a vector of size 8 from 8 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3, const T& a4,
  //           const T& a5, const T& a6, const T& a7, const T& a8);
  //   /// construct a vector of size 9 from 9 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3, const T& a4,
  //           const T& a5, const T& a6, const T& a7, const T& a8,
  //           const T& a9);
  //   /// construct a vector of size 10 from 10 scalar values
  //   SVector(const T& a1, const T& a2, const T& a3, const T& a4,
  //           const T& a5, const T& a6, const T& a7, const T& a8,
  //           const T& a9, const T& a10);

  //   /// assignment from a scalar (only for size 1 vector)
  //   SVector<T, D>& operator=(const T& a1);
  //   /// assignment from another vector
  //   SVector<T, D>& operator=(const SVector<T, D>& rhs);
  //   /// assignment  from Vector Expression
  //   template <class A>
  //   SVector<T, D>& operator=(const VecExpr<A, T, D>& rhs);

  //   /** @name --- Access functions --- */

  //   /**
  //        Enumeration defining the Vector size
  //      */
  //   enum {
  //     /// return vector size
  //     mSize = D
  //   };

  //   /// return dimension \f$D\f$
  //   inline static size_t Dim() { return D; }
  //   /// access the parse tree. Index starts from zero
  //   T apply(size_t i) const;
  //   /// return read-only pointer to internal array
  //   const T* Array() const;
  //   /// return non-const pointer to internal array
  //   T* Array();

  //   /** @name --- STL-like interface --- */

  //   /** STL iterator interface. */
  //   iterator begin();

  //   /** STL iterator interface. */
  //   iterator end();

  //   /** STL const_iterator interface. */
  //   const_iterator begin() const;

  //   /** STL const_iterator interface. */
  //   const_iterator end() const;

  //   /// set vector elements copying the values
  //   /// iterator size must match vector size
  //   template <class InputIterator>
  //   void SetElements(InputIterator begin, InputIterator end);

  //   /// set vector elements copying the values
  //   /// size must be <= vector size
  //   template <class InputIterator>
  //   void SetElements(InputIterator begin, size_t size);

  //   /** @name --- Operators --- */

  //   /// element wise comparison
  //   bool operator==(const T& rhs) const;
  //   /// element wise comparison
  //   bool operator!=(const T& rhs) const;
  //   /// element wise comparison
  //   bool operator==(const SVector<T, D>& rhs) const;
  //   /// element wise comparison
  //   bool operator!=(const SVector<T, D>& rhs) const;
  //   /// element wise comparison
  //   template <class A>
  //   bool operator==(const VecExpr<A, T, D>& rhs) const;
  //   /// element wise comparison
  //   template <class A>
  //   bool operator!=(const VecExpr<A, T, D>& rhs) const;

  //   /// element wise comparison
  //   bool operator>(const T& rhs) const;
  //   /// element wise comparison
  //   bool operator<(const T& rhs) const;
  //   /// element wise comparison
  //   bool operator>(const SVector<T, D>& rhs) const;
  //   /// element wise comparison
  //   bool operator<(const SVector<T, D>& rhs) const;
  //   /// element wise comparison
  //   template <class A>
  //   bool operator>(const VecExpr<A, T, D>& rhs) const;
  //   /// element wise comparison
  //   template <class A>
  //   bool operator<(const VecExpr<A, T, D>& rhs) const;

  //   /// read-only access of vector elements. Index starts from 0.
  //   const T& operator[](size_t i) const;
  //   /// read-only access of vector elements. Index starts from 0.
  //   const T& operator()(size_t i) const;
  //   /// read-only access of vector elements with check on index. Index starts from 0.
  //   const T& At(size_t i) const;
  //   /// read/write access of vector elements. Index starts from 0.
  //   T& operator[](size_t i);
  //   /// read/write access of vector elements. Index starts from 0.
  //   T& operator()(size_t i);
  //   /// read/write access of vector elements with check on index. Index starts from 0.
  //   T& At(size_t i);

  //   /// self addition with a scalar
  //   SVector<T, D>& operator+=(const T& rhs);
  //   /// self subtraction with a scalar
  //   SVector<T, D>& operator-=(const T& rhs);
  //   /// self multiplication with a scalar
  //   SVector<T, D>& operator*=(const T& rhs);
  //   /// self division with a scalar
  //   SVector<T, D>& operator/=(const T& rhs);

  //   /// self addition with another vector
  //   SVector<T, D>& operator+=(const SVector<T, D>& rhs);
  //   /// self subtraction with another vector
  //   SVector<T, D>& operator-=(const SVector<T, D>& rhs);
  //   /// self addition with a vector expression
  //   template <class A>
  //   SVector<T, D>& operator+=(const VecExpr<A, T, D>& rhs);
  //   /// self subtraction with a vector expression
  //   template <class A>
  //   SVector<T, D>& operator-=(const VecExpr<A, T, D>& rhs);

  // #ifdef OLD_IMPL
  // #ifndef __CINT__
  //   /// self element-wise multiplication  with another vector
  //   SVector<T, D>& operator*=(const SVector<T, D>& rhs);
  //   /// self element-wise division with another vector
  //   SVector<T, D>& operator/=(const SVector<T, D>& rhs);

  //   /// self element-wise multiplication  with a vector expression
  //   template <class A>
  //   SVector<T, D>& operator*=(const VecExpr<A, T, D>& rhs);
  //   /// self element-wise division  with a vector expression
  //   template <class A>
  //   SVector<T, D>& operator/=(const VecExpr<A, T, D>& rhs);

  // #endif
  // #endif

  //   /** @name --- Expert functions --- */
  //   /// transform vector into a vector of length 1
  //   SVector<T, D>& Unit();
  //   /// place a sub-vector starting from the given position
  //   template <size_t D2>
  //   SVector<T, D>& Place_at(const SVector<T, D2>& rhs, size_t row);
  //   /// place a sub-vector expression starting from the given position
  //   template <class A, size_t D2>
  //   SVector<T, D>& Place_at(const VecExpr<A, T, D2>& rhs, size_t row);

  //   /**
  //        return a subvector of size N starting at the value row
  //      where N is the size of the returned vector (SubVector::mSize)
  //      Condition  row+N <= D
  //      */
  //   template <class SubVector>
  //   SubVector Sub(size_t row) const;

  //   /**
  //         Function to check if a vector is sharing same memory location of the passed pointer
  //         This function is used by the expression templates to avoid the alias problem during
  //         expression evaluation. When  the vector is in use, for example in operations
  //         like V = M * V, where M is a mtrix, a temporary object storing the intermediate result is automatically
  //         created when evaluating the expression.

  //     */
  //   bool IsInUse(const T* p) const;

  //   /// used by operator<<()
  //   std::ostream& Print(std::ostream& os) const;

 private:
  T mArray[N];
};

template <class T, size_t D>
SVector<T, D>::SVector()
{
  for (size_t i = 0; i < D; ++i) {
    mArray[i] = 0;
  }
}

template <class T, size_t D>
template <class A>
SVector<T, D>::SVector(const VecExpr<A, T, D>& rhs)
{
  operator=(rhs);
}

template <class T, size_t D>
SVector<T, D>::SVector(const SVector<T, D>& rhs)
{
  for (size_t i = 0; i < D; ++i)
    mArray[i] = rhs.mArray[i];
}

// template <class T, size_t D>
// class MatRepSym
// {

//  public:
//   /* constexpr */ inline MatRepSym() {}

//   typedef T value_type;

//   inline T& operator()(size_t i, size_t j)
//   {
//     return fArray[offset(i, j)];
//   }

//   inline /* constexpr */ T const& operator()(size_t i, size_t j) const
//   {
//     return fArray[offset(i, j)];
//   }

//   inline T& operator[](size_t i)
//   {
//     return fArray[off(i)];
//   }

//   inline /* constexpr */ T const& operator[](size_t i) const
//   {
//     return fArray[off(i)];
//   }

//   inline /* constexpr */ T apply(size_t i) const
//   {
//     return fArray[off(i)];
//   }

//   inline T* Array() { return fArray; }

//   inline const T* Array() const { return fArray; }

//   /**
//           assignment : only symmetric to symmetric allowed
//         */
//   template <class R>
//   inline MatRepSym<T, D>& operator=(const R&)
//   {
//     STATIC_CHECK(0 == 1,
//                  Cannot_assign_general_to_symmetric_matrix_representation);
//     return *this;
//   }
//   inline MatRepSym<T, D>& operator=(const MatRepSym& rhs)
//   {
//     for (size_t i = 0; i < mSize; ++i)
//       fArray[i] = rhs.Array()[i];
//     return *this;
//   }

//   /**
//           self addition : only symmetric to symmetric allowed
//         */
//   template <class R>
//   inline MatRepSym<T, D>& operator+=(const R&)
//   {
//     STATIC_CHECK(0 == 1,
//                  Cannot_add_general_to_symmetric_matrix_representation);
//     return *this;
//   }
//   inline MatRepSym<T, D>& operator+=(const MatRepSym& rhs)
//   {
//     for (size_t i = 0; i < mSize; ++i)
//       fArray[i] += rhs.Array()[i];
//     return *this;
//   }

//   /**
//           self subtraction : only symmetric to symmetric allowed
//         */
//   template <class R>
//   inline MatRepSym<T, D>& operator-=(const R&)
//   {
//     STATIC_CHECK(0 == 1,
//                  Cannot_substract_general_to_symmetric_matrix_representation);
//     return *this;
//   }
//   inline MatRepSym<T, D>& operator-=(const MatRepSym& rhs)
//   {
//     for (size_t i = 0; i < mSize; ++i)
//       fArray[i] -= rhs.Array()[i];
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
//   //T __attribute__ ((aligned (16))) fArray[mSize];
//   T fArray[mSize];
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
