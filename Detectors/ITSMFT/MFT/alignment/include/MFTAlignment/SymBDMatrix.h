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

/// \file SymBDMatrix.h
/// \brief Symmetric Band Diagonal matrix (from AliROOT) with half band width W (+1 for diagonal)
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MFT_SYMBDMATRIX_H
#define ALICEO2_MFT_SYMBDMATRIX_H

#include <TObject.h>
#include <TVectorD.h>
#include "MFTAlignment/MatrixSq.h"

namespace o2
{
namespace mft
{

/// \class SymBDMatrix
/// \brief Symmetric Matrix Class
///
/// Only lower triangle is stored in the "profile" format
class SymBDMatrix : public MatrixSq
{

 public:
  enum { kDecomposedBit = 0x1 };

  /// \brief def. c-tor
  SymBDMatrix();

  /// \brief c-tor for given size
  SymBDMatrix(Int_t size, Int_t w = 0);

  /// \brief copy c-tor
  SymBDMatrix(const SymBDMatrix& mat);

  /// \brief d-tor
  virtual ~SymBDMatrix();

  Int_t GetBandHWidth() const { return fNrows; }
  Int_t GetNElemsStored() const { return fNelems; }

  /// \brief clear dynamic part
  void Clear(Option_t* option = "");

  /// \brief set all elems to 0
  void Reset();

  /// \brief get fraction of non-zero elements
  Float_t GetDensity() const;

  /// \brief assignment operator
  SymBDMatrix& operator=(const SymBDMatrix& src);

  Double_t operator()(Int_t rown, Int_t coln) const;
  Double_t& operator()(Int_t rown, Int_t coln);
  Double_t operator()(Int_t rown) const;
  Double_t& operator()(Int_t rown);

  Double_t DiagElem(Int_t r) const { return (*(const SymBDMatrix*)this)(r, r); }
  Double_t& DiagElem(Int_t r) { return (*this)(r, r); }

  /// \brief decomposition to L Diag L^T
  void DecomposeLDLT();

  /// \brief solve matrix equation
  void Solve(Double_t* rhs);

  /// \brief solve matrix equation
  void Solve(const Double_t* rhs, Double_t* sol);

  void Solve(TVectorD& rhs) { Solve(rhs.GetMatrixArray()); }
  void Solve(const TVectorD& rhs, TVectorD& sol) { Solve(rhs.GetMatrixArray(), sol.GetMatrixArray()); }

  /// \brief print data
  void Print(Option_t* option = "") const;

  void SetDecomposed(Bool_t v = kTRUE) { SetBit(kDecomposedBit, v); }
  Bool_t IsDecomposed() const { return TestBit(kDecomposedBit); }

  /// \brief fill vecOut by matrix*vecIn
  ///
  /// vector should be of the same size as the matrix
  void MultiplyByVec(const Double_t* vecIn, Double_t* vecOut) const;

  void MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const;

  /// \brief add list of elements to row r
  void AddToRow(Int_t r, Double_t* valc, Int_t* indc, Int_t n);

  virtual Int_t GetIndex(Int_t row, Int_t col) const;
  virtual Int_t GetIndex(Int_t diagID) const;
  Double_t GetEl(Int_t row, Int_t col) const { return operator()(row, col); }
  void SetEl(Int_t row, Int_t col, Double_t val) { operator()(row, col) = val; }

 protected:
  Double_t* fElems; ///< Elements booked by constructor

  ClassDef(SymBDMatrix, 0);
};

//___________________________________________________________
inline Int_t SymBDMatrix::GetIndex(Int_t row, Int_t col) const
{
  // lower triangle band is actually filled
  if (row < col)
    Swap(row, col);
  col -= row;
  if (col < -GetBandHWidth())
    return -1;
  return GetIndex(row) + col;
}

//___________________________________________________________
/// \brief Get index of the diagonal element on row diagID
inline Int_t SymBDMatrix::GetIndex(Int_t diagID) const
{
  return (diagID + 1) * fRowLwb - 1;
}

//___________________________________________________________
inline Double_t SymBDMatrix::operator()(Int_t row, Int_t col) const
{
  // query element
  int idx = GetIndex(row, col);
  return (const Double_t&)idx < 0 ? 0.0 : fElems[idx];
}

//___________________________________________________________
inline Double_t& SymBDMatrix::operator()(Int_t row, Int_t col)
{
  // get element for assingment; assignment outside of the stored range has no effect
  int idx = GetIndex(row, col);
  if (idx >= 0)
    return fElems[idx];
  fTol = 0;
  return fTol;
}

//___________________________________________________________
inline Double_t SymBDMatrix::operator()(Int_t row) const
{
  // query diagonal
  return (const Double_t&)fElems[GetIndex(row)];
}

//___________________________________________________________
inline Double_t& SymBDMatrix::operator()(Int_t row)
{
  // get diagonal for assingment; assignment outside of the stored range has no effect
  return fElems[GetIndex(row)];
}

//___________________________________________________________
inline void SymBDMatrix::MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const
{
  MultiplyByVec(vecIn.GetMatrixArray(), vecOut.GetMatrixArray());
}

} // namespace mft
} // namespace o2

#endif
