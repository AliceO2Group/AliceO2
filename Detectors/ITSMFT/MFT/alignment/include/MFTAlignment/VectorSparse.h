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

/// \file VectorSparse.h
/// \brief Sparse vector class (from AliROOT), used as row of the MatrixSparse class
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MFT_VECTORSPARSE_H
#define ALICEO2_MFT_VECTORSPARSE_H

#include <TObject.h>
#include <TMath.h>

namespace o2
{
namespace mft
{

/// \class VectorSparse
class VectorSparse : public TObject
{
 public:
  VectorSparse();

  /// \brief copy c-tor
  VectorSparse(const VectorSparse& src);

  ~VectorSparse() override { Clear(); }

  /// \brief print itself
  void Print(Option_t* option = "") const override;

  Int_t GetNElems() const { return fNElems; }
  UShort_t* GetIndices() const { return fIndex; }
  Double_t* GetElems() const { return fElems; }
  UShort_t& GetIndex(Int_t i) { return fIndex[i]; }
  Double_t& GetElem(Int_t i) const { return fElems[i]; }

  /// \brief clear all
  void Clear(Option_t* option = "") override;

  void Reset() { memset(fElems, 0, fNElems * sizeof(Double_t)); }

  /// \brief change the size
  void ReSize(Int_t sz, Bool_t copy = kFALSE);

  /// \brief sort indices in increasing order. Used to fix the row after ILUk decomposition
  void SortIndices(Bool_t valuesToo = kFALSE);

  /// \brief add indiced array to row. Indices must be in increasing order
  void Add(Double_t* valc, Int_t* indc, Int_t n);

  /// \brief assignment op-tor
  VectorSparse& operator=(const VectorSparse& src);

  virtual Double_t operator()(Int_t ind) const;
  virtual Double_t& operator()(Int_t ind);

  /// \brief set element to 0 if it was already defined
  virtual void SetToZero(Int_t ind);

  /// \brief return an element with given index
  Double_t FindIndex(Int_t ind) const;

  /// \brief increment an element with given index
  Double_t& FindIndexAdd(Int_t ind);

  Int_t GetLastIndex() const { return fIndex[fNElems - 1]; }
  Double_t GetLastElem() const { return fElems[fNElems - 1]; }
  Double_t& GetLastElem() { return fElems[fNElems - 1]; }

 protected:
  Int_t fNElems;    ///< Number of elements
  UShort_t* fIndex; ///< Index of stored elems
  Double_t* fElems; ///< pointer on elements

  ClassDefOverride(VectorSparse, 0);
};

//___________________________________________________
inline Double_t VectorSparse::operator()(Int_t ind) const
{
  return FindIndex(ind);
}

//___________________________________________________
inline Double_t& VectorSparse::operator()(Int_t ind)
{
  return FindIndexAdd(ind);
}

} // namespace mft
} // namespace o2

#endif
