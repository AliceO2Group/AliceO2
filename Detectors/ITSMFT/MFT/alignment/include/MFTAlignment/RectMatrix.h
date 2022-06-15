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

/// \file RectMatrix.h
/// \brief Class for rectangular matrix used for millepede2 operation
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MFT_RECTMATRIX_H
#define ALICEO2_MFT_RECTMATRIX_H

#include "TObject.h"
class TString;

namespace o2
{
namespace mft
{

/// \brief Class for rectangular matrix used for millepede2 operation
///
/// Matrix may be sparse or dense
class RectMatrix : public TObject
{

 public:
  /// \brief default c-tor
  RectMatrix();

  /// \brief c-tor
  RectMatrix(Int_t nrow, Int_t ncol);

  /// \brief copy c-tor
  RectMatrix(const RectMatrix& src);

  /// \brief dest-tor
  virtual ~RectMatrix();

  Int_t GetNRows() const { return fNRows; }
  Int_t GetNCols() const { return fNCols; }

  Double_t Query(Int_t rown, Int_t coln) const { return operator()(rown, coln); }

  /// \brief assignment op-r
  RectMatrix& operator=(const RectMatrix& src);

  Double_t operator()(Int_t rown, Int_t coln) const;
  Double_t& operator()(Int_t rown, Int_t coln);
  Double_t* operator()(Int_t row) const { return GetRow(row); }
  Double_t* GetRow(Int_t row) const { return fRows[row]; }

  /// \brief reset all
  void Reset() const;

  /// \brief print itself
  virtual void Print(Option_t* option = "") const;

 protected:
  Int_t fNRows;     ///< Number of rows
  Int_t fNCols;     ///< Number of columns
  Double_t** fRows; ///< pointers on rows

  ClassDef(RectMatrix, 0);
};

//___________________________________________________________
inline Double_t RectMatrix::operator()(Int_t row, Int_t col) const
{
  return (const Double_t&)GetRow(row)[col];
}

//___________________________________________________________
inline Double_t& RectMatrix::operator()(Int_t row, Int_t col)
{
  return (Double_t&)fRows[row][col];
}

} // namespace mft
} // namespace o2

#endif
