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

/// \file MatrixSparse.h
/// \brief Sparse matrix class (from AliROOT), used as a global matrix for MillePede2
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MFT_MATRIXSPARSE_H
#define ALICEO2_MFT_MATRIXSPARSE_H

#include "MFTAlignment/MatrixSq.h"
#include "MFTAlignment/VectorSparse.h"

namespace o2
{
namespace mft
{

/// \class MatrixSparse
class MatrixSparse : public MatrixSq
{
 public:
  MatrixSparse() : fVecs(0) {}

  /// \brief constructor
  MatrixSparse(Int_t size);

  /// \brief copy c-tor
  MatrixSparse(const MatrixSparse& mat);

  virtual ~MatrixSparse() { Clear(); }

  VectorSparse* GetRow(Int_t ir) const { return (ir < fNcols) ? fVecs[ir] : 0; }
  VectorSparse* GetRowAdd(Int_t ir);

  virtual Int_t GetSize() const { return fNrows; }
  virtual Int_t GetNRows() const { return fNrows; }
  virtual Int_t GetNCols() const { return fNcols; }

  void Clear(Option_t* option = "");
  void Reset()
  {
    for (int i = fNcols; i--;)
      GetRow(i)->Reset();
  }
  void Print(Option_t* option = "") const;
  MatrixSparse& operator=(const MatrixSparse& src);
  Double_t& operator()(Int_t row, Int_t col);
  Double_t operator()(Int_t row, Int_t col) const;
  void SetToZero(Int_t row, Int_t col);

  /// \brief get fraction of non-zero elements
  Float_t GetDensity() const;

  Double_t DiagElem(Int_t r) const;
  Double_t& DiagElem(Int_t r);

  /// \brief sort columns in increasing order. Used to fix the matrix after ILUk decompostion
  void SortIndices(Bool_t valuesToo = kFALSE);

  /// \brief fill vecOut by matrix * vecIn (vector should be of the same size as the matrix)
  void MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const;

  void MultiplyByVec(const Double_t* vecIn, Double_t* vecOut) const;

  void AddToRow(Int_t r, Double_t* valc, Int_t* indc, Int_t n);

 protected:
  VectorSparse** fVecs;

  ClassDef(MatrixSparse, 0);
};

//___________________________________________________
/// \brief multiplication
inline void MatrixSparse::MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const
{
  MultiplyByVec((Double_t*)vecIn.GetMatrixArray(), (Double_t*)vecOut.GetMatrixArray());
}

//___________________________________________________
/// \brief set existing element to 0
inline void MatrixSparse::SetToZero(Int_t row, Int_t col)
{
  if (IsSymmetric() && col > row)
    Swap(row, col);
  VectorSparse* rowv = GetRow(row);
  if (rowv)
    rowv->SetToZero(col);
}

//___________________________________________________
inline Double_t MatrixSparse::operator()(Int_t row, Int_t col) const
{
  if (IsSymmetric() && col > row)
    Swap(row, col);
  VectorSparse* rowv = GetRow(row);
  if (!rowv)
    return 0;
  return rowv->FindIndex(col);
}

//___________________________________________________
inline Double_t& MatrixSparse::operator()(Int_t row, Int_t col)
{
  //  printf("M: findindexAdd\n");
  if (IsSymmetric() && col > row)
    Swap(row, col);
  VectorSparse* rowv = GetRowAdd(row);
  if (col >= fNcols)
    fNcols = col + 1;
  return rowv->FindIndexAdd(col);
}

//___________________________________________________
/// \brief get diag elem
inline Double_t MatrixSparse::DiagElem(Int_t row) const
{
  VectorSparse* rowv = GetRow(row);
  if (!rowv)
    return 0;
  if (IsSymmetric()) {
    return (rowv->GetNElems() > 0 && rowv->GetLastIndex() == row) ? rowv->GetLastElem() : 0.;
  } else
    return rowv->FindIndex(row);
}

//___________________________________________________
/// \brief get diag elem
inline Double_t& MatrixSparse::DiagElem(Int_t row)
{
  VectorSparse* rowv = GetRowAdd(row);
  if (row >= fNcols)
    fNcols = row + 1;
  if (IsSymmetric()) {
    return (rowv->GetNElems() > 0 && rowv->GetLastIndex() == row) ? rowv->GetLastElem() : rowv->FindIndexAdd(row);
  } else
    return rowv->FindIndexAdd(row);
}

} // namespace mft
} // namespace o2

#endif
