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
class GPUSVector<T, N>
{
 public:
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  GPUSVector();
//   template <class A>
//   SVector(const VecExpr<A, T, D>& rhs);
//   /// copy contructor
//   SVector(const SVector<T, D>& rhs);

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
//   explicit SVector(InputIterator begin, unsigned int size);

// #else
//   // if you use iterator this is not necessary

//   /// fill from array with len must be equal to D!
//   SVector(const T* a, unsigned int len);

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
//     kSize = D
//   };

//   /// return dimension \f$D\f$
//   inline static unsigned int Dim() { return D; }
//   /// access the parse tree. Index starts from zero
//   T apply(unsigned int i) const;
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
//   void SetElements(InputIterator begin, unsigned int size);

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
//   const T& operator[](unsigned int i) const;
//   /// read-only access of vector elements. Index starts from 0.
//   const T& operator()(unsigned int i) const;
//   /// read-only access of vector elements with check on index. Index starts from 0.
//   const T& At(unsigned int i) const;
//   /// read/write access of vector elements. Index starts from 0.
//   T& operator[](unsigned int i);
//   /// read/write access of vector elements. Index starts from 0.
//   T& operator()(unsigned int i);
//   /// read/write access of vector elements with check on index. Index starts from 0.
//   T& At(unsigned int i);

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
//   template <unsigned int D2>
//   SVector<T, D>& Place_at(const SVector<T, D2>& rhs, unsigned int row);
//   /// place a sub-vector expression starting from the given position
//   template <class A, unsigned int D2>
//   SVector<T, D>& Place_at(const VecExpr<A, T, D2>& rhs, unsigned int row);

//   /**
//        return a subvector of size N starting at the value row
//      where N is the size of the returned vector (SubVector::kSize)
//      Condition  row+N <= D
//      */
//   template <class SubVector>
//   SubVector Sub(unsigned int row) const;

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
  T mArray[D];
};

template <typename T, size_t N>
using SVector = GPUSVector<T, N>;

#else
template <typename T, size_t N>
using SVector = ROOT::Math::SVector<T, N>;
#endif

} // end namespace o2::math_utils

#endif
