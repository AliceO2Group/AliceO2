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

/// \file SymMatrix.h
/// \brief Fast symmetric matrix (from AliROOT) with dynamically expandable size
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MFT_SYMMATRIX_H
#define ALICEO2_MFT_SYMMATRIX_H

#include <TVectorD.h>
#include <TString.h>

#include "MFTAlignment/MatrixSq.h"

namespace o2
{
namespace mft
{

/// \class SymMatrix
/// \brief Fast symmetric matrix with dynamically expandable size.
///
/// Only part can be used for matrix operations. It is defined as:
/// \param fNCols: rows built by constructor (GetSizeBooked)
/// \param fNRows: number of rows added dynamically (automatically added on assignment to row), GetNRowAdded
/// \param fNRowIndex: total size (fNCols+fNRows), GetSize
/// \param fRowLwb: actual size to used for given operation, by default = total size, GetSizeUsed
class SymMatrix : public MatrixSq
{

 public:
  /// \brief default constructor
  SymMatrix();

  /// \brief constructor for matrix with defined size
  SymMatrix(Int_t size);

  /// \brief copy constructor
  SymMatrix(const SymMatrix& mat);

  /// \brief destructor
  ~SymMatrix() override;

  /// \brief clear dynamic part
  void Clear(Option_t* option = "") override;
  void Reset() override;

  Int_t GetSize() const override { return fNrowIndex; }
  Int_t GetSizeUsed() const { return fRowLwb; }
  Int_t GetSizeBooked() const { return fNcols; }
  Int_t GetSizeAdded() const { return fNrows; }

  /// \brief get fraction of non-zero elements
  Float_t GetDensity() const override;

  /// \brief assignment operator
  SymMatrix& operator=(const SymMatrix& src);

  /// \brief add operator
  SymMatrix& operator+=(const SymMatrix& src);

  /// \brief minus operator
  SymMatrix& operator-=(const SymMatrix& src);

  Double_t operator()(Int_t rown, Int_t coln) const override;
  Double_t& operator()(Int_t rown, Int_t coln) override;

  Double_t DiagElem(Int_t r) const override { return (*(const SymMatrix*)this)(r, r); }
  Double_t& DiagElem(Int_t r) override { return (*this)(r, r); }

  /// \brief get pointer on the row
  Double_t* GetRow(Int_t r);

  /// \brief print itself
  void Print(const Option_t* option = "") const override;

  /// \brief add empty rows
  void AddRows(int nrows = 1);

  void SetSizeUsed(Int_t sz) { fRowLwb = sz; }

  void Scale(Double_t coeff);

  /// \brief multiply from the right
  Bool_t Multiply(const SymMatrix& right);

  /// \brief fill vecOut by matrix*vecIn
  ///
  /// vector should be of the same size as the matrix
  void MultiplyByVec(const Double_t* vecIn, Double_t* vecOut) const override;

  void MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const override;
  void AddToRow(Int_t r, Double_t* valc, Int_t* indc, Int_t n) override;

  // ---------------------------------- Dummy methods of MatrixBase
  const Double_t* GetMatrixArray() const override { return fElems; };
  Double_t* GetMatrixArray() override { return (Double_t*)fElems; }
  const Int_t* GetRowIndexArray() const override
  {
    Error("GetRowIndexArray", "Dummy");
    return nullptr;
  };
  Int_t* GetRowIndexArray() override
  {
    Error("GetRowIndexArray", "Dummy");
    return nullptr;
  };
  const Int_t* GetColIndexArray() const override
  {
    Error("GetColIndexArray", "Dummy");
    return nullptr;
  };
  Int_t* GetColIndexArray() override
  {
    Error("GetColIndexArray", "Dummy");
    return nullptr;
  };
  TMatrixDBase& SetRowIndexArray(Int_t*) override
  {
    Error("SetRowIndexArray", "Dummy");
    return *this;
  }
  TMatrixDBase& SetColIndexArray(Int_t*) override
  {
    Error("SetColIndexArray", "Dummy");
    return *this;
  }
  TMatrixDBase& GetSub(Int_t, Int_t, Int_t, Int_t, TMatrixDBase&, Option_t*) const override
  {
    Error("GetSub", "Dummy");
    return *((TMatrixDBase*)this);
  }
  TMatrixDBase& SetSub(Int_t, Int_t, const TMatrixDBase&) override
  {
    Error("GetSub", "Dummy");
    return *this;
  }
  TMatrixDBase& ResizeTo(Int_t, Int_t, Int_t) override
  {
    Error("ResizeTo", "Dummy");
    return *this;
  }
  TMatrixDBase& ResizeTo(Int_t, Int_t, Int_t, Int_t, Int_t) override
  {
    Error("ResizeTo", "Dummy");
    return *this;
  }

  // ----------------------------- Choleski methods ----------------------------------------
  /// \brief Return a matrix with Choleski decomposition
  ///
  /// Adopted from Numerical Recipes in C, ch.2-9, http://www.nr.com
  /// consturcts Cholesky decomposition of SYMMETRIC and
  /// POSITIVELY-DEFINED matrix a (a=L*Lt)
  /// Only upper triangle of the matrix has to be filled.
  /// In opposite to function from the book, the matrix is modified:
  /// lower triangle and diagonal are refilled.
  SymMatrix* DecomposeChol();

  /// \brief Invert using provided Choleski decomposition, provided the Cholseki's L matrix
  void InvertChol(SymMatrix* mchol);

  /// \brief Invert matrix using Choleski decomposition
  Bool_t InvertChol();

  /// \brief Solves the set of n linear equations A x = b
  ///
  /// Adopted from Numerical Recipes in C, ch.2-9, http://www.nr.com
  /// Solves the set of n linear equations A x = b,
  /// where a is a positive-definite symmetric matrix.
  /// a[1..n][1..n] is the output of the routine CholDecomposw.
  /// Only the lower triangle of a is accessed. b[1..n] is input as the
  /// right-hand side vector. The solution vector is returned in b[1..n].
  Bool_t SolveChol(Double_t* brhs, Bool_t invert = kFALSE);

  Bool_t SolveChol(Double_t* brhs, Double_t* bsol, Bool_t invert = kFALSE);
  Bool_t SolveChol(TVectorD& brhs, Bool_t invert = kFALSE);
  Bool_t SolveChol(const TVectorD& brhs, TVectorD& bsol, Bool_t invert = kFALSE);

  /// \brief Solves the set of n linear equations A x = b; this version solve multiple RHSs at once
  ///
  /// Adopted from Numerical Recipes in C, ch.2-9, http://www.nr.com
  /// Solves the set of n linear equations A x = b,
  /// where a is a positive-definite symmetric matrix.
  /// a[1..n][1..n] is the output of the routine CholDecomposw.
  /// Only the lower triangle of a is accessed. b[1..n] is input as the
  /// right-hand side vector. The solution vector is returned in b[1..n].
  /// This version solve multiple RHSs at once
  Bool_t SolveCholN(Double_t* bn, int nRHS, Bool_t invert = kFALSE);

  /// \brief Obtain solution of a system of linear equations with symmetric matrix
  ///        and the inverse (using 'singular-value friendly' GAUSS pivot)
  ///
  /// Solution a la MP1: gaussian eliminations
  int SolveSpmInv(double* vecB, Bool_t stabilize = kTRUE);

 protected:
  virtual Int_t GetIndex(Int_t row, Int_t col) const;
  Double_t GetEl(Int_t row, Int_t col) const { return operator()(row, col); }
  void SetEl(Int_t row, Int_t col, Double_t val) { operator()(row, col) = val; }

 protected:
  Double_t* fElems;     ///<   Elements booked by constructor
  Double_t** fElemsAdd; ///<   Elements (rows) added dynamicaly

  static SymMatrix* fgBuffer; ///< buffer for fast solution
  static Int_t fgCopyCnt;     ///< matrix copy counter

  ClassDefOverride(SymMatrix, 0);
};

//___________________________________________________________
inline Int_t SymMatrix::GetIndex(Int_t row, Int_t col) const
{
  // lower triangle is actually filled
  return ((row * (row + 1)) >> 1) + col;
}

//___________________________________________________________
inline Double_t SymMatrix::operator()(Int_t row, Int_t col) const
{
  //
  if (row < col) {
    Swap(row, col);
  }
  if (row >= fNrowIndex) {
    return 0;
  }
  return (const Double_t&)(row < fNcols ? fElems[GetIndex(row, col)] : (fElemsAdd[row - fNcols])[col]);
}

//___________________________________________________________
inline Double_t& SymMatrix::operator()(Int_t row, Int_t col)
{
  if (row < col) {
    Swap(row, col);
  }
  if (row >= fNrowIndex) {
    AddRows(row - fNrowIndex + 1);
  }
  return (row < fNcols ? fElems[GetIndex(row, col)] : (fElemsAdd[row - fNcols])[col]);
}

//___________________________________________________________
inline void SymMatrix::MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const
{
  MultiplyByVec(vecIn.GetMatrixArray(), vecOut.GetMatrixArray());
}

//___________________________________________________________
inline void SymMatrix::Scale(Double_t coeff)
{
  for (int i = fNrowIndex; i--;) {
    for (int j = i; j--;) {
      double& el = operator()(i, j);
      if (el) {
        el *= coeff;
      }
    }
  }
}

//___________________________________________________________
inline void SymMatrix::AddToRow(Int_t r, Double_t* valc, Int_t* indc, Int_t n)
{
  for (int i = n; i--;) {
    (*this)(indc[i], r) += valc[i];
  }
}

} // namespace mft
} // namespace o2

#endif
