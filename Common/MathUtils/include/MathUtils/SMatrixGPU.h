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
/// @author Matteo Concas

#ifndef ALICEO2_ROOTSMATRIX_H
#define ALICEO2_ROOTSMATRIX_H

#ifndef GPUCA_GPUCODE_DEVICE
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#endif

#include "GPUCommonArray.h"

namespace o2::math_utils
{
#ifdef GPUCA_GPUCODE_DEVICE

template <bool>
struct CompileTimeChecker {
  CompileTimeChecker(void*) {}
};
template <>
struct CompileTimeChecker<false> {
};

#define STATIC_CHECK(expr, msg)                  \
  {                                              \
    class ERROR_##msg                            \
    {                                            \
    };                                           \
    ERROR_##msg e;                               \
    (void)(CompileTimeChecker<(expr) != 0>(&e)); \
  }

template <typename T, size_t N>
class SVectorGPU
{
 public:
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  SVectorGPU();
  SVectorGPU(const SVectorGPU<T, N>& rhs);

 private:
  T mArray[N];
};

template <class T, size_t D>
SVectorGPU<T, D>::SVectorGPU()
{
  for (size_t i = 0; i < D; ++i) {
    mArray[i] = 0;
  }
}

template <class T, size_t D>
SVectorGPU<T, D>::SVectorGPU(const SVectorGPU<T, D>& rhs)
{
  for (size_t i = 0; i < D; ++i) {
    mArray[i] = rhs.mArray[i];
  }
}

// utils for SMatrix METTI A POSTO ANCHE GLI ENUM
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

template <class T, size_t D>
class MatRepSym
{
 public:
  inline MatRepSym() {}
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

  // assignment : only symmetric to symmetric allowed
  template <class R>
  inline MatRepSym<T, D>& operator=(const R&)
  {
    STATIC_CHECK(0 == 1,
                 Cannot_assign_general_to_symmetric_matrix_representation);
    return *this;
  }
  inline MatRepSym<T, D>& operator=(const MatRepSym& rhs)
  {
    for (size_t i = 0; i < mSize; ++i)
      mArray[i] = rhs.Array()[i];
    return *this;
  }

  // self addition : only symmetric to symmetric allowed
  template <class R>
  inline MatRepSym<T, D>& operator+=(const R&)
  {
    STATIC_CHECK(0 == 1,
                 Cannot_add_general_to_symmetric_matrix_representation);
    return *this;
  }
  inline MatRepSym<T, D>& operator+=(const MatRepSym& rhs)
  {
    for (size_t i = 0; i < mSize; ++i)
      mArray[i] += rhs.Array()[i];
    return *this;
  }

  // self subtraction : only symmetric to symmetric allowed
  template <class R>
  inline MatRepSym<T, D>& operator-=(const R&)
  {
    STATIC_CHECK(0 == 1,
                 Cannot_substract_general_to_symmetric_matrix_representation);
    return *this;
  }
  inline MatRepSym<T, D>& operator-=(const MatRepSym& rhs)
  {
    for (size_t i = 0; i < mSize; ++i)
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
    /// return no. of matrix rows
    kRows = D,
    /// return no. of matrix columns
    kCols = D,
    /// return no of elements: rows*columns
    mSize = D * (D + 1) / 2
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
    //if (j > i) std::swap(i, j);
    return off(i * D + j);
    // return (i>j) ? (i * (i+1) / 2) + j :  (j * (j+1) / 2) + i;
  }

 private:
  //T __attribute__ ((aligned (16))) mArray[mSize];
  T mArray[mSize];
};

/// SMatReprStd starting port here
template <class T, unsigned int D1, unsigned int D2 = D1>
class MatRepStd
{

 public:
  typedef T value_type;

  inline const T& operator()(unsigned int i, unsigned int j) const
  {
    return fArray[i * D2 + j];
  }
  inline T& operator()(unsigned int i, unsigned int j)
  {
    return fArray[i * D2 + j];
  }
  inline T& operator[](unsigned int i) { return fArray[i]; }

  inline const T& operator[](unsigned int i) const { return fArray[i]; }

  inline T apply(unsigned int i) const { return fArray[i]; }

  inline T* Array() { return fArray; }

  inline const T* Array() const { return fArray; }

  template <class R>
  inline MatRepStd<T, D1, D2>& operator+=(const R& rhs)
  {
    for (unsigned int i = 0; i < kSize; ++i)
      fArray[i] += rhs[i];
    return *this;
  }

  template <class R>
  inline MatRepStd<T, D1, D2>& operator-=(const R& rhs)
  {
    for (unsigned int i = 0; i < kSize; ++i)
      fArray[i] -= rhs[i];
    return *this;
  }

  template <class R>
  inline MatRepStd<T, D1, D2>& operator=(const R& rhs)
  {
    for (unsigned int i = 0; i < kSize; ++i)
      fArray[i] = rhs[i];
    return *this;
  }

  template <class R>
  inline bool operator==(const R& rhs) const
  {
    bool rc = true;
    for (unsigned int i = 0; i < kSize; ++i) {
      rc = rc && (fArray[i] == rhs[i]);
    }
    return rc;
  }

  enum {
    /// return no. of matrix rows
    kRows = D1,
    /// return no. of matrix columns
    kCols = D2,
    /// return no of elements: rows*columns
    kSize = D1 * D2
  };

 private:
  //T __attribute__ ((aligned (16))) fArray[kSize];
  T fArray[kSize];
};

/**
         Static structure to keep the conversion from (i,j) to offsets in the storage data for a
         symmetric matrix
      */

template <unsigned int D>
struct RowOffsets {
  inline RowOffsets()
  {
    int v[D];
    v[0] = 0;
    for (unsigned int i = 1; i < D; ++i)
      v[i] = v[i - 1] + i;
    for (unsigned int i = 0; i < D; ++i) {
      for (unsigned int j = 0; j <= i; ++j)
        fOff[i * D + j] = v[i] + j;
      for (unsigned int j = i + 1; j < D; ++j)
        fOff[i * D + j] = v[j] + i;
    }
  }
  inline int operator()(unsigned int i, unsigned int j) const { return fOff[i * D + j]; }
  inline int apply(unsigned int i) const { return fOff[i]; }
  int fOff[D * D];
};
/// SMatReprStd ending port here

/// SMatrix starting port here
struct SMatrixIdentity {
};
struct SMatrixNoInit {
};

template <class T,
          unsigned int D1,
          unsigned int D2 = D1,
          class R = MatRepStd<T, D1, D2>>
class SMatrix
{
 public:
  /** @name --- Typedefs --- */

  /** contained scalar type */
  typedef T value_type;

  /** storage representation type */
  typedef R rep_type;

  /** STL iterator interface. */
  typedef T* iterator;

  /** STL const_iterator interface. */
  typedef const T* const_iterator;

  /** @name --- Constructors and Assignment --- */

  /**
       Default constructor:
    */
  SMatrix();
  ///
  /**
        construct from without initialization
    */
  inline SMatrix(SMatrixNoInit) {}

  /**
        construct from an identity matrix
    */
  SMatrix(SMatrixIdentity);
  /**
        copy constructor (from a matrix of the same representation
    */
  SMatrix(const SMatrix<T, D1, D2, R>& rhs);
  /**
       construct from a matrix with different representation.
       Works only from symmetric to general and not viceversa.
    */
  template <class R2>
  SMatrix(const SMatrix<T, D1, D2, R2>& rhs);

  /**
       Construct from an expression.
       In case of symmetric matrices does not work if expression is of type general
       matrices. In case one needs to force the assignment from general to symmetric, one can use the
       ROOT::Math::AssignSym::Evaluate function.
    */
  template <class A, class R2>
  SMatrix(const Expr<A, T, D1, D2, R2>& rhs);

  /**
       Constructor with STL iterator interface. The data will be copied into the matrix
       \param begin start iterator position
       \param end end iterator position
       \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
       \param lower if true the lower triangular part is filled
 
       Size of the matrix must match size of the iterators, if triang is false, otherwise the size of the
       triangular block. In the case of symmetric matrices triang is considered always to be true
       (what-ever the user specifies) and the size of the iterators must be equal to the size of the
       triangular block, which is the number of independent elements of a symmetric matrix:  N*(N+1)/2
 
    */
  template <class InputIterator>
  SMatrix(InputIterator begin, InputIterator end, bool triang = false, bool lower = true);

  /**
       Constructor with STL iterator interface. The data will be copied into the matrix
       \param begin  start iterator position
       \param size   iterator size
       \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
       \param lower if true the lower triangular part is filled
 
       Size of the iterators must not be larger than the size of the matrix representation.
       In the case of symmetric matrices the size is N*(N+1)/2.
 
    */
  template <class InputIterator>
  SMatrix(InputIterator begin, unsigned int size, bool triang = false, bool lower = true);

  /**
       constructor of a symmetrix a matrix from a SVector containing the lower (upper)
       triangular part.
    */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SMatrix(const SVector<T, D1*(D2 + 1) / 2>& v, bool lower = true);
#else
  template <unsigned int N>
  SMatrix(const SVector<T, N>& v, bool lower = true);
#endif

  /**
        Construct from a scalar value (only for size 1 matrices)
    */
  explicit SMatrix(const T& rhs);

  /**
        Assign from another compatible matrix.
        Possible Symmetirc to general but NOT vice-versa
    */
  template <class M>
  SMatrix<T, D1, D2, R>& operator=(const M& rhs);

  /**
        Assign from a matrix expression
    */
  template <class A, class R2>
  SMatrix<T, D1, D2, R>& operator=(const Expr<A, T, D1, D2, R2>& rhs);

  /**
       Assign from an identity matrix
    */
  SMatrix<T, D1, D2, R>& operator=(SMatrixIdentity);

  /**
        Assign from a scalar value (only for size 1 matrices)
    */
  SMatrix<T, D1, D2, R>& operator=(const T& rhs);

  /** @name --- Matrix dimension --- */

  /**
       Enumeration defining the matrix dimension,
       number of rows, columns and size = rows*columns)
    */
  enum {
    /// return no. of matrix rows
    kRows = D1,
    /// return no. of matrix columns
    kCols = D2,
    /// return no of elements: rows*columns
    kSize = D1 * D2
  };

  /** @name --- Access functions --- */

  /** access the parse tree with the index starting from zero and
        following the C convention for the order in accessing
        the matrix elements.
        Same convention for general and symmetric matrices.
    */
  T apply(unsigned int i) const;

  /// return read-only pointer to internal array
  const T* Array() const;
  /// return pointer to internal array
  T* Array();

  /** @name --- STL-like interface ---
        The iterators access the matrix element in the order how they are
        stored in memory. The C (row-major) convention is used, and in the
        case of symmetric matrices the iterator spans only the lower diagonal
        block. For example for a symmetric 3x3 matrices the order of the 6
        elements \f${a_0,...a_5}\f$ is:
        \f[
        M = \left( \begin{array}{ccc}
        a_0 & a_1 & a_3  \\
        a_1 & a_2  & a_4  \\
        a_3 & a_4 & a_5   \end{array} \right)
        \f]
    */

  /** STL iterator interface. */
  iterator begin();

  /** STL iterator interface. */
  iterator end();

  /** STL const_iterator interface. */
  const_iterator begin() const;

  /** STL const_iterator interface. */
  const_iterator end() const;

  /**
       Set matrix elements with STL iterator interface. The data will be copied into the matrix
       \param begin start iterator position
       \param end end iterator position
       \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
       \param lower if true the lower triangular part is filled
 
       Size of the matrix must match size of the iterators, if triang is false, otherwise the size of the
       triangular block. In the case of symmetric matrices triang is considered always to be true
       (what-ever the user specifies) and the size of the iterators must be equal to the size of the
       triangular block, which is the number of independent elements of a symmetric matrix:  N*(N+1)/2
 
    */
  template <class InputIterator>
  void SetElements(InputIterator begin, InputIterator end, bool triang = false, bool lower = true);

  /**
       Constructor with STL iterator interface. The data will be copied into the matrix
       \param begin  start iterator position
       \param size   iterator size
       \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
       \param lower if true the lower triangular part is filled
 
       Size of the iterators must not be larger than the size of the matrix representation.
       In the case of symmetric matrices the size is N*(N+1)/2.
 
    */
  template <class InputIterator>
  void SetElements(InputIterator begin, unsigned int size, bool triang = false, bool lower = true);

  /** @name --- Operators --- */
  /// element wise comparison
  bool operator==(const T& rhs) const;
  /// element wise comparison
  bool operator!=(const T& rhs) const;
  /// element wise comparison
  template <class R2>
  bool operator==(const SMatrix<T, D1, D2, R2>& rhs) const;
  /// element wise comparison
  bool operator!=(const SMatrix<T, D1, D2, R>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator==(const Expr<A, T, D1, D2, R2>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator!=(const Expr<A, T, D1, D2, R2>& rhs) const;

  /// element wise comparison
  bool operator>(const T& rhs) const;
  /// element wise comparison
  bool operator<(const T& rhs) const;
  /// element wise comparison
  template <class R2>
  bool operator>(const SMatrix<T, D1, D2, R2>& rhs) const;
  /// element wise comparison
  template <class R2>
  bool operator<(const SMatrix<T, D1, D2, R2>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator>(const Expr<A, T, D1, D2, R2>& rhs) const;
  /// element wise comparison
  template <class A, class R2>
  bool operator<(const Expr<A, T, D1, D2, R2>& rhs) const;

  /**
       read only access to matrix element, with indices starting from 0
    */
  const T& operator()(unsigned int i, unsigned int j) const;
  /**
       read/write access to matrix element with indices starting from 0
    */
  T& operator()(unsigned int i, unsigned int j);

  /**
       read only access to matrix element, with indices starting from 0.
       Function will check index values and it will assert if they are wrong
    */
  const T& At(unsigned int i, unsigned int j) const;
  /**
       read/write access to matrix element with indices starting from 0.
       Function will check index values and it will assert if they are wrong
    */
  T& At(unsigned int i, unsigned int j);

  // helper class for implementing the m[i][j] operator

  class SMatrixRow
  {
   public:
    SMatrixRow(SMatrix<T, D1, D2, R>& rhs, unsigned int i) : fMat(&rhs), fRow(i)
    {
    }
    T& operator[](int j) { return (*fMat)(fRow, j); }

   private:
    SMatrix<T, D1, D2, R>* fMat;
    unsigned int fRow;
  };

  class SMatrixRow_const
  {
   public:
    SMatrixRow_const(const SMatrix<T, D1, D2, R>& rhs, unsigned int i) : fMat(&rhs), fRow(i)
    {
    }

    const T& operator[](int j) const { return (*fMat)(fRow, j); }

   private:
    const SMatrix<T, D1, D2, R>* fMat;
    unsigned int fRow;
  };

  /**
       read only access to matrix element, with indices starting from 0 : m[i][j]
    */
  SMatrixRow_const operator[](unsigned int i) const { return SMatrixRow_const(*this, i); }
  /**
       read/write access to matrix element with indices starting from 0 : m[i][j]
    */
  SMatrixRow operator[](unsigned int i) { return SMatrixRow(*this, i); }

  /**
       addition with a scalar
    */
  SMatrix<T, D1, D2, R>& operator+=(const T& rhs);

  /**
       addition with another matrix of any compatible representation
    */
  template <class R2>
  SMatrix<T, D1, D2, R>& operator+=(const SMatrix<T, D1, D2, R2>& rhs);

  /**
       addition with a compatible matrix expression
    */
  template <class A, class R2>
  SMatrix<T, D1, D2, R>& operator+=(const Expr<A, T, D1, D2, R2>& rhs);

  /**
       subtraction with a scalar
    */
  SMatrix<T, D1, D2, R>& operator-=(const T& rhs);

  /**
       subtraction with another matrix of any compatible representation
    */
  template <class R2>
  SMatrix<T, D1, D2, R>& operator-=(const SMatrix<T, D1, D2, R2>& rhs);

  /**
       subtraction with a compatible matrix expression
    */
  template <class A, class R2>
  SMatrix<T, D1, D2, R>& operator-=(const Expr<A, T, D1, D2, R2>& rhs);

  /**
       multiplication with a scalar
     */
  SMatrix<T, D1, D2, R>& operator*=(const T& rhs);

#ifndef __CINT__

  /**
       multiplication with another compatible matrix (it is a real matrix multiplication)
       Note that this operation does not avid to create a temporary to store intermidiate result
    */
  template <class R2>
  SMatrix<T, D1, D2, R>& operator*=(const SMatrix<T, D1, D2, R2>& rhs);

  /**
       multiplication with a compatible matrix expression (it is a real matrix multiplication)
    */
  template <class A, class R2>
  SMatrix<T, D1, D2, R>& operator*=(const Expr<A, T, D1, D2, R2>& rhs);

#endif

  /**
       division with a scalar
    */
  SMatrix<T, D1, D2, R>& operator/=(const T& rhs);

  /** @name --- Linear Algebra Functions --- */

  /**
       Invert a square Matrix ( this method changes the current matrix).
       Return true if inversion is successfull.
       The method used for general square matrices is the LU factorization taken from Dinv routine
       from the CERNLIB (written in C++ from CLHEP authors)
       In case of symmetric matrices Bunch-Kaufman diagonal pivoting method is used
       (The implementation is the one written by the CLHEP authors)
    */
  bool Invert();

  /**
       Invert a square Matrix and  returns a new matrix. In case the inversion fails
       the current matrix is returned.
       \param ifail . ifail will be set to 0 when inversion is successfull.
       See ROOT::Math::SMatrix::Invert for the inversion algorithm
    */
  SMatrix<T, D1, D2, R> Inverse(int& ifail) const;

  /**
       Fast Invertion of a square Matrix ( this method changes the current matrix).
       Return true if inversion is successfull.
       The method used is based on direct inversion using the Cramer rule for
       matrices upto 5x5. Afterwards the same default algorithm of Invert() is used.
       Note that this method is faster but can suffer from much larger numerical accuracy
       when the condition of the matrix is large
    */
  bool InvertFast();

  /**
       Invert a square Matrix and  returns a new matrix. In case the inversion fails
       the current matrix is returned.
       \param ifail . ifail will be set to 0 when inversion is successfull.
       See ROOT::Math::SMatrix::InvertFast for the inversion algorithm
    */
  SMatrix<T, D1, D2, R> InverseFast(int& ifail) const;

  /**
       Invertion of a symmetric positive defined Matrix using Choleski decomposition.
       ( this method changes the current matrix).
       Return true if inversion is successfull.
       The method used is based on Choleski decomposition
       A compile error is given if the matrix is not of type symmetric and a run-time failure if the
       matrix is not positive defined.
       For solving  a linear system, it is possible to use also the function
       ROOT::Math::SolveChol(matrix, vector) which will be faster than performing the inversion
    */
  bool InvertChol();

  /**
       Invert of a symmetric positive defined Matrix using Choleski decomposition.
       A compile error is given if the matrix is not of type symmetric and a run-time failure if the
       matrix is not positive defined.
       In case the inversion fails the current matrix is returned.
       \param ifail . ifail will be set to 0 when inversion is successfull.
       See ROOT::Math::SMatrix::InvertChol for the inversion algorithm
    */
  SMatrix<T, D1, D2, R> InverseChol(int& ifail) const;

  /**
       determinant of square Matrix via Dfact.
       Return true when the calculation is successfull.
       \param det will contain the calculated determinant value
       \b Note: this will destroy the contents of the Matrix!
    */
  bool Det(T& det);

  /**
       determinant of square Matrix via Dfact.
       Return true when the calculation is successfull.
       \param det will contain the calculated determinant value
       \b Note: this will preserve the content of the Matrix!
    */
  bool Det2(T& det) const;

  /** @name --- Matrix Slice Functions --- */

  /// place a vector in a Matrix row
  template <unsigned int D>
  SMatrix<T, D1, D2, R>& Place_in_row(const SVector<T, D>& rhs,
                                      unsigned int row,
                                      unsigned int col);
  /// place a vector expression in a Matrix row
  template <class A, unsigned int D>
  SMatrix<T, D1, D2, R>& Place_in_row(const VecExpr<A, T, D>& rhs,
                                      unsigned int row,
                                      unsigned int col);
  /// place a vector in a Matrix column
  template <unsigned int D>
  SMatrix<T, D1, D2, R>& Place_in_col(const SVector<T, D>& rhs,
                                      unsigned int row,
                                      unsigned int col);
  /// place a vector expression in a Matrix column
  template <class A, unsigned int D>
  SMatrix<T, D1, D2, R>& Place_in_col(const VecExpr<A, T, D>& rhs,
                                      unsigned int row,
                                      unsigned int col);
  /// place a matrix in this matrix
  template <unsigned int D3, unsigned int D4, class R2>
  SMatrix<T, D1, D2, R>& Place_at(const SMatrix<T, D3, D4, R2>& rhs,
                                  unsigned int row,
                                  unsigned int col);
  /// place a matrix expression in this matrix
  template <class A, unsigned int D3, unsigned int D4, class R2>
  SMatrix<T, D1, D2, R>& Place_at(const Expr<A, T, D3, D4, R2>& rhs,
                                  unsigned int row,
                                  unsigned int col);

  /**
       return a full Matrix row as a vector (copy the content in a new vector)
    */
  SVector<T, D2> Row(unsigned int therow) const;

  /**
       return a full Matrix column as a vector (copy the content in a new vector)
    */
  SVector<T, D1> Col(unsigned int thecol) const;

  /**
       return a slice of therow as a vector starting at the colum value col0 until col0+N,
       where N is the size of the vector (SubVector::kSize )
       Condition  col0+N <= D2
    */
  template <class SubVector>
  SubVector SubRow(unsigned int therow, unsigned int col0 = 0) const;

  /**
       return a slice of the column as a vector starting at the row value row0 until row0+Dsub.
       where N is the size of the vector (SubVector::kSize )
       Condition  row0+N <= D1
    */
  template <class SubVector>
  SubVector SubCol(unsigned int thecol, unsigned int row0 = 0) const;

  /**
       return a submatrix with the upper left corner at the values (row0, col0) and with sizes N1, N2
       where N1 and N2 are the dimension of the sub-matrix (SubMatrix::kRows and SubMatrix::kCols )
       Condition  row0+N1 <= D1 && col0+N2 <=D2
    */
  template <class SubMatrix>
  SubMatrix Sub(unsigned int row0, unsigned int col0) const;

  /**
       return diagonal elements of a matrix as a Vector.
       It works only for squared matrices D1 == D2, otherwise it will produce a compile error
    */
  SVector<T, D1> Diagonal() const;

  /**
       Set the diagonal elements from a Vector
       Require that vector implements ::kSize since a check (statically) is done on
       diagonal size == vector size
    */
  template <class Vector>
  void SetDiagonal(const Vector& v);

  /**
       return the trace of a matrix
       Sum of the diagonal elements
    */
  T Trace() const;

  /**
       return the upper Triangular block of the matrices (including the diagonal) as
       a vector of sizes N = D1 * (D1 + 1)/2.
       It works only for square matrices with D1==D2, otherwise it will produce a compile error
    */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SVector<T, D1*(D2 + 1) / 2> UpperBlock() const;
#else
  template <class SubVector>
  SubVector UpperBlock() const;
#endif

  /**
       return the lower Triangular block of the matrices (including the diagonal) as
       a vector of sizes N = D1 * (D1 + 1)/2.
       It works only for square matrices with D1==D2, otherwise it will produce a compile error
    */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
  SVector<T, D1*(D2 + 1) / 2> LowerBlock() const;
#else
  template <class SubVector>
  SubVector LowerBlock() const;
#endif

  /** @name --- Other Functions --- */

  /**
        Function to check if a matrix is sharing same memory location of the passed pointer
        This function is used by the expression templates to avoid the alias problem during
        expression evaluation. When  the matrix is in use, for example in operations
        like A = B * A, a temporary object storing the intermediate result is automatically
        created when evaluating the expression.
 
    */
  bool IsInUse(const T* p) const;

  // submatrices

  /// Print: used by operator<<()
  std::ostream& Print(std::ostream& os) const;

 public:
  /** @name --- Data Member --- */

  /**
       Matrix Storage Object containing matrix data
    */
  R fRep;

}; // end of class SMatrix

//==============================================================================
// operator<<
//==============================================================================
template <class T, unsigned int D1, unsigned int D2, class R>
inline std::ostream& operator<<(std::ostream& os, const ROOT::Math::SMatrix<T, D1, D2, R>& rhs)
{
  return rhs.Print(os);
}
/// SMatrixGPU end port here

// Define aliases
template <typename T, size_t N>
using SVector = SVectorGPU<T, N>;
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