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

/// \file MatrixSq.h
/// \brief Abstract class (from AliROOT) for square matrix used for millepede2 operation
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MFT_MATRIXSQ_H
#define ALICEO2_MFT_MATRIXSQ_H

#include <TMatrixDBase.h>
#include <TVectorD.h>

namespace o2
{
namespace mft
{

/// \class MatrixSq
class MatrixSq : public TMatrixDBase
{

 public:
  MatrixSq() : fSymmetric(kFALSE) {}
  MatrixSq(const MatrixSq& src) : TMatrixDBase(src), fSymmetric(src.fSymmetric) {}
  virtual ~MatrixSq() {}

  /// \brief = operator
  MatrixSq& operator=(const MatrixSq& src);

  virtual Int_t GetSize() const { return fNcols; }
  virtual Float_t GetDensity() const = 0;

  virtual void Clear(Option_t* option = "") = 0;

  virtual Double_t Query(Int_t rown, Int_t coln) const { return operator()(rown, coln); }
  virtual Double_t operator()(Int_t rown, Int_t coln) const = 0;
  virtual Double_t& operator()(Int_t rown, Int_t coln) = 0;

  virtual Double_t QueryDiag(Int_t rc) const { return DiagElem(rc); }
  virtual Double_t DiagElem(Int_t r) const = 0;
  virtual Double_t& DiagElem(Int_t r) = 0;
  virtual void AddToRow(Int_t r, Double_t* valc, Int_t* indc, Int_t n) = 0;

  virtual void Print(Option_t* option = "") const = 0;
  virtual void Reset() = 0;

  /// \brief print matrix in COO sparse format
  virtual void PrintCOO() const;

  /// \brief fill vecOut by matrix * vecIn (vector should be of the same size as the matrix)
  virtual void MultiplyByVec(const Double_t* vecIn, Double_t* vecOut) const;

  virtual void MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const;

  Bool_t IsSymmetric() const { return fSymmetric; }
  void SetSymmetric(Bool_t v = kTRUE) { fSymmetric = v; }

  // ---------------------------------- Dummy methods of MatrixBase
  virtual const Double_t* GetMatrixArray() const
  {
    Error("GetMatrixArray", "Dummy");
    return 0;
  };
  virtual Double_t* GetMatrixArray()
  {
    Error("GetMatrixArray", "Dummy");
    return 0;
  };
  virtual const Int_t* GetRowIndexArray() const
  {
    Error("GetRowIndexArray", "Dummy");
    return 0;
  };
  virtual Int_t* GetRowIndexArray()
  {
    Error("GetRowIndexArray", "Dummy");
    return 0;
  };
  virtual const Int_t* GetColIndexArray() const
  {
    Error("GetColIndexArray", "Dummy");
    return 0;
  };
  virtual Int_t* GetColIndexArray()
  {
    Error("GetColIndexArray", "Dummy");
    return 0;
  };
  virtual TMatrixDBase& SetRowIndexArray(Int_t*)
  {
    Error("SetRowIndexArray", "Dummy");
    return *this;
  }
  virtual TMatrixDBase& SetColIndexArray(Int_t*)
  {
    Error("SetColIndexArray", "Dummy");
    return *this;
  }
  virtual TMatrixDBase& GetSub(Int_t, Int_t, Int_t, Int_t, TMatrixDBase&, Option_t*) const
  {
    Error("GetSub", "Dummy");
    return *((TMatrixDBase*)this);
  }
  virtual TMatrixDBase& SetSub(Int_t, Int_t, const TMatrixDBase&)
  {
    Error("GetSub", "Dummy");
    return *this;
  }
  virtual TMatrixDBase& ResizeTo(Int_t, Int_t, Int_t)
  {
    Error("ResizeTo", "Dummy");
    return *this;
  }
  virtual TMatrixDBase& ResizeTo(Int_t, Int_t, Int_t, Int_t, Int_t)
  {
    Error("ResizeTo", "Dummy");
    return *this;
  }
  virtual void Allocate(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t)
  {
    Error("Allocate", "Dummy");
    return;
  }

  static Bool_t IsZero(Double_t x, Double_t thresh = 1e-64) { return x > 0 ? (x < thresh) : (x > -thresh); }

 protected:
  void Swap(int& r, int& c) const
  {
    int t = r;
    r = c;
    c = t;
  }

 protected:
  Bool_t fSymmetric; ///< is the matrix symmetric? Only lower triangle is filled

  ClassDef(MatrixSq, 1);
};

//___________________________________________________________
inline void MatrixSq::MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const
{
  MultiplyByVec(vecIn.GetMatrixArray(), vecOut.GetMatrixArray());
}

} // namespace mft
} // namespace o2

#endif
