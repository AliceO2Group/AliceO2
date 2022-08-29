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

/// @file SymMatrix.cxx

#include <iostream>

#include <TClass.h>
#include <TMath.h>
#include "MFTAlignment/SymMatrix.h"
#include "Framework/Logger.h"

using namespace o2::mft;

ClassImp(SymMatrix);

SymMatrix* SymMatrix::fgBuffer = nullptr;
Int_t SymMatrix::fgCopyCnt = 0;

//___________________________________________________________
SymMatrix::SymMatrix()
  : fElems(nullptr),
    fElemsAdd(nullptr)
{
  fSymmetric = kTRUE;
  fgCopyCnt++;
}

//___________________________________________________________
SymMatrix::SymMatrix(Int_t size)
  : MatrixSq(),
    fElems(nullptr),
    fElemsAdd(nullptr)
{
  fNrows = 0;
  fNrowIndex = fNcols = fRowLwb = size;
  fElems = new Double_t[fNcols * (fNcols + 1) / 2];
  fSymmetric = kTRUE;
  Reset();
  fgCopyCnt++;
}

//___________________________________________________________
SymMatrix::SymMatrix(const SymMatrix& src)
  : MatrixSq(src),
    fElems(nullptr),
    fElemsAdd(nullptr)
{
  fNrowIndex = fNcols = src.GetSize();
  fNrows = 0;
  fRowLwb = src.GetSizeUsed();
  if (fNcols) {
    int nmainel = fNcols * (fNcols + 1) / 2;
    fElems = new Double_t[nmainel];
    nmainel = src.fNcols * (src.fNcols + 1) / 2;
    memcpy(fElems, src.fElems, nmainel * sizeof(Double_t));
    if (src.GetSizeAdded()) { // transfer extra rows to main matrix
      Double_t* pnt = fElems + nmainel;
      int ncl = src.GetSizeBooked() + 1;
      for (int ir = 0; ir < src.GetSizeAdded(); ir++) {
        memcpy(pnt, src.fElemsAdd[ir], ncl * sizeof(Double_t));
        pnt += ncl;
        ncl++;
      }
    }
  } else {
    fElems = nullptr;
  }
  fElemsAdd = nullptr;
  fgCopyCnt++;
}

//___________________________________________________________
SymMatrix::~SymMatrix()
{
  Clear();
  if (--fgCopyCnt < 1 && fgBuffer) {
    delete fgBuffer;
    fgBuffer = nullptr;
  }
}

//___________________________________________________________
SymMatrix& SymMatrix::operator=(const SymMatrix& src)
{
  if (this != &src) {
    TObject::operator=(src);
    if (GetSizeBooked() != src.GetSizeBooked() && GetSizeAdded() != src.GetSizeAdded()) {
      // recreate the matrix
      if (fElems) {
        delete[] fElems;
      }
      for (int i = 0; i < GetSizeAdded(); i++) {
        delete[] fElemsAdd[i];
      }
      delete[] fElemsAdd;

      fNrowIndex = src.GetSize();
      fNcols = src.GetSize();
      fNrows = 0;
      fRowLwb = src.GetSizeUsed();
      fElems = new Double_t[GetSize() * (GetSize() + 1) / 2];
      int nmainel = src.GetSizeBooked() * (src.GetSizeBooked() + 1);
      memcpy(fElems, src.fElems, nmainel * sizeof(Double_t));
      if (src.GetSizeAdded()) {           // transfer extra rows to main matrix
        Double_t* pnt = fElems + nmainel; //*sizeof(Double_t);
        int ncl = src.GetSizeBooked() + 1;
        for (int ir = 0; ir < src.GetSizeAdded(); ir++) {
          ncl += ir;
          memcpy(pnt, src.fElemsAdd[ir], ncl * sizeof(Double_t));
          pnt += ncl; //*sizeof(Double_t);
        }
      }

    } else {
      memcpy(fElems, src.fElems, GetSizeBooked() * (GetSizeBooked() + 1) / 2 * sizeof(Double_t));
      int ncl = GetSizeBooked() + 1;
      for (int ir = 0; ir < GetSizeAdded(); ir++) { // dynamic rows
        ncl += ir;
        memcpy(fElemsAdd[ir], src.fElemsAdd[ir], ncl * sizeof(Double_t));
      }
    }
  }
  return *this;
}

//___________________________________________________________
SymMatrix& SymMatrix::operator+=(const SymMatrix& src)
{
  if (GetSizeUsed() != src.GetSizeUsed()) {
    LOG(error) << "Matrix sizes are different";
    return *this;
  }
  for (int i = 0; i < GetSizeUsed(); i++) {
    for (int j = i; j < GetSizeUsed(); j++) {
      (*this)(j, i) += src(j, i);
    }
  }
  return *this;
}

//___________________________________________________________
SymMatrix& SymMatrix::operator-=(const SymMatrix& src)
{
  if (GetSizeUsed() != src.GetSizeUsed()) {
    LOG(error) << "Matrix sizes are different";
    return *this;
  }
  for (int i = 0; i < GetSizeUsed(); i++) {
    for (int j = i; j < GetSizeUsed(); j++) {
      (*this)(j, i) -= src(j, i);
    }
  }
  return *this;
}

//___________________________________________________________
void SymMatrix::Clear(Option_t*)
{
  if (fElems) {
    delete[] fElems;
    fElems = nullptr;
  }

  if (fElemsAdd) {
    for (int i = 0; i < GetSizeAdded(); i++) {
      delete[] fElemsAdd[i];
    }
    delete[] fElemsAdd;
    fElemsAdd = nullptr;
  }
  fNrowIndex = fNcols = fNrows = fRowLwb = 0;
}

//___________________________________________________________
Float_t SymMatrix::GetDensity() const
{
  Int_t nel = 0;
  for (int i = GetSizeUsed(); i--;) {
    for (int j = i + 1; j--;) {
      if (!IsZero(GetEl(i, j))) {
        nel++;
      }
    }
  }
  return 2. * nel / ((GetSizeUsed() + 1) * GetSizeUsed());
}

//___________________________________________________________
void SymMatrix::Print(Option_t* option) const
{
  printf("Symmetric Matrix: Size = %d (%d rows added dynamically), %d used\n", GetSize(), GetSizeAdded(), GetSizeUsed());
  TString opt = option;
  opt.ToLower();
  if (opt.IsNull()) {
    return;
  }
  opt = "%";
  opt += 1 + int(TMath::Log10(double(GetSize())));
  opt += "d|";
  for (Int_t i = 0; i < GetSizeUsed(); i++) {
    printf(opt, i);
    for (Int_t j = 0; j <= i; j++) {
      printf("%+.3e|", GetEl(i, j));
    }
    printf("\n");
  }
}

//___________________________________________________________
void SymMatrix::MultiplyByVec(const Double_t* vecIn, Double_t* vecOut) const
{
  for (int i = GetSizeUsed(); i--;) {
    vecOut[i] = 0.0;
    for (int j = GetSizeUsed(); j--;) {
      vecOut[i] += vecIn[j] * GetEl(i, j);
    }
  }
}

//___________________________________________________________
Bool_t SymMatrix::Multiply(const SymMatrix& right)
{
  int sz = GetSizeUsed();
  if (sz != right.GetSizeUsed()) {
    LOG(error) << "Matrix sizes are different";
    return kFALSE;
  }
  if (!fgBuffer || fgBuffer->GetSizeUsed() != sz) {
    delete fgBuffer;
    fgBuffer = new SymMatrix(*this);
  } else {
    (*fgBuffer) = *this;
  }

  for (int i = sz; i--;) {
    for (int j = i + 1; j--;) {
      double val = 0.;
      for (int k = sz; k--;) {
        val += fgBuffer->GetEl(i, k) * right.GetEl(k, j);
      }
      SetEl(i, j, val);
    }
  }

  return kTRUE;
}

//___________________________________________________________
SymMatrix* SymMatrix::DecomposeChol()
{
  if (!fgBuffer || fgBuffer->GetSizeUsed() != GetSizeUsed()) {
    delete fgBuffer;
    fgBuffer = new SymMatrix(*this);
  } else {
    (*fgBuffer) = *this;
  }

  SymMatrix& mchol = *fgBuffer;

  for (int i = 0; i < GetSizeUsed(); i++) {
    Double_t* rowi = mchol.GetRow(i);
    for (int j = i; j < GetSizeUsed(); j++) {
      Double_t* rowj = mchol.GetRow(j);
      double sum = rowj[i];
      for (int k = i - 1; k >= 0; k--) {
        if (rowi[k] && rowj[k])
          sum -= rowi[k] * rowj[k];
      }
      if (i == j) {
        if (sum <= 0.0) { // not positive-definite
          LOG(debug) << "The matrix is not positive definite [" << sum
                     << "]: Choleski decomposition is not possible";
          // Print("l");
          return 0;
        }
        rowi[i] = TMath::Sqrt(sum);
      } else {
        rowj[i] = sum / rowi[i];
      }
    }
  }
  return fgBuffer;
}

//___________________________________________________________
Bool_t SymMatrix::InvertChol()
{
  SymMatrix* mchol = DecomposeChol();
  if (!mchol) {
    LOG(error) << "Failed to invert the matrix";
    return kFALSE;
  }

  InvertChol(mchol);
  return kTRUE;
}

//___________________________________________________________
void SymMatrix::InvertChol(SymMatrix* pmchol)
{
  Double_t sum;
  SymMatrix& mchol = *pmchol;

  // Invert decomposed triangular L matrix (Lower triangle is filled)
  for (int i = 0; i < GetSizeUsed(); i++) {
    mchol(i, i) = 1.0 / mchol(i, i);
    for (int j = i + 1; j < GetSizeUsed(); j++) {
      Double_t* rowj = mchol.GetRow(j);
      sum = 0.0;
      for (int k = i; k < j; k++) {
        if (rowj[k]) {
          double& mki = mchol(k, i);
          if (mki) {
            sum -= rowj[k] * mki;
          }
        }
      }
      rowj[i] = sum / rowj[j];
    }
  }

  // take product of the inverted Choleski L matrix with its transposed
  for (int i = GetSizeUsed(); i--;) {
    for (int j = i + 1; j--;) {
      sum = 0;
      for (int k = i; k < GetSizeUsed(); k++) {
        double& mik = mchol(i, k);
        if (mik) {
          double& mjk = mchol(j, k);
          if (mjk) {
            sum += mik * mjk;
          }
        }
      }
      (*this)(j, i) = sum;
    }
  }
}

//___________________________________________________________
Bool_t SymMatrix::SolveChol(Double_t* b, Bool_t invert)
{
  Int_t i, k;
  Double_t sum;

  SymMatrix* pmchol = DecomposeChol();
  if (!pmchol) {
    LOG(debug) << "SolveChol failed";
    //    Print("l");
    return kFALSE;
  }
  SymMatrix& mchol = *pmchol;

  for (i = 0; i < GetSizeUsed(); i++) {
    Double_t* rowi = mchol.GetRow(i);
    for (sum = b[i], k = i - 1; k >= 0; k--)
      if (rowi[k] && b[k]) {
        sum -= rowi[k] * b[k];
      }
    b[i] = sum / rowi[i];
  }

  for (i = GetSizeUsed() - 1; i >= 0; i--) {
    for (sum = b[i], k = i + 1; k < GetSizeUsed(); k++)
      if (b[k]) {
        double& mki = mchol(k, i);
        if (mki) {
          sum -= mki * b[k];
        }
      }
    b[i] = sum / mchol(i, i);
  }

  if (invert)
    InvertChol(pmchol);
  return kTRUE;
}

//___________________________________________________________
Bool_t SymMatrix::SolveCholN(Double_t* bn, int nRHS, Bool_t invert)
{
  int sz = GetSizeUsed();
  Int_t i, k;
  Double_t sum;

  SymMatrix* pmchol = DecomposeChol();
  if (!pmchol) {
    LOG(debug) << "SolveChol failed";
    //    Print("l");
    return kFALSE;
  }
  SymMatrix& mchol = *pmchol;

  for (int ir = 0; ir < nRHS; ir++) {
    double* b = bn + ir * sz;

    for (i = 0; i < sz; i++) {
      Double_t* rowi = mchol.GetRow(i);
      for (sum = b[i], k = i - 1; k >= 0; k--)
        if (rowi[k] && b[k]) {
          sum -= rowi[k] * b[k];
        }
      b[i] = sum / rowi[i];
    }

    for (i = sz - 1; i >= 0; i--) {
      for (sum = b[i], k = i + 1; k < sz; k++)
        if (b[k]) {
          double& mki = mchol(k, i);
          if (mki) {
            sum -= mki * b[k];
          }
        }
      b[i] = sum / mchol(i, i);
    }
  }

  if (invert)
    InvertChol(pmchol);
  return kTRUE;
}

//___________________________________________________________
Bool_t SymMatrix::SolveChol(TVectorD& b, Bool_t invert)
{
  return SolveChol((Double_t*)b.GetMatrixArray(), invert);
}

//___________________________________________________________
Bool_t SymMatrix::SolveChol(Double_t* brhs, Double_t* bsol, Bool_t invert)
{
  memcpy(bsol, brhs, GetSizeUsed() * sizeof(Double_t));
  return SolveChol(bsol, invert);
}

//___________________________________________________________
Bool_t SymMatrix::SolveChol(const TVectorD& brhs, TVectorD& bsol, Bool_t invert)
{
  bsol = brhs;
  return SolveChol(bsol, invert);
}

//___________________________________________________________
void SymMatrix::AddRows(int nrows)
{
  if (nrows < 1)
    return;
  Double_t** pnew = new Double_t*[nrows + fNrows];
  for (int ir = 0; ir < fNrows; ir++) {
    pnew[ir] = fElemsAdd[ir]; // copy old extra rows
  }
  for (int ir = 0; ir < nrows; ir++) {
    int ncl = GetSize() + 1;
    pnew[fNrows] = new Double_t[ncl];
    memset(pnew[fNrows], 0, ncl * sizeof(Double_t));
    fNrows++;
    fNrowIndex++;
    fRowLwb++;
  }
  delete[] fElemsAdd;
  fElemsAdd = pnew;
}

//___________________________________________________________
void SymMatrix::Reset()
{
  // if additional rows exist, regularize it
  if (fElemsAdd) {
    delete[] fElems;
    for (int i = 0; i < fNrows; i++) {
      delete[] fElemsAdd[i];
    }
    delete[] fElemsAdd;
    fElemsAdd = nullptr;
    fNcols = fRowLwb = fNrowIndex;
    fElems = new Double_t[GetSize() * (GetSize() + 1) / 2];
    fNrows = 0;
  }
  if (fElems) {
    memset(fElems, 0, GetSize() * (GetSize() + 1) / 2 * sizeof(Double_t));
  }
}

//___________________________________________________________
/*
void SymMatrix::AddToRow(Int_t r, Double_t* valc, Int_t* indc, Int_t n)
{
  //   for (int i=n;i--;) {
  //     (*this)(indc[i],r) += valc[i];
  //   }
  //   return;

  double* row;
  if (r >= fNrowIndex) {
    AddRows(r - fNrowIndex + 1);
    row = &((fElemsAdd[r - fNcols])[0]);
  } else {
    row = &fElems[GetIndex(r, 0)];
  }

  int nadd = 0;
  for (int i = n; i--;) {
    if (indc[i] > r) {
      continue;
    }
    row[indc[i]] += valc[i];
    nadd++;
  }
  if (nadd == n) {
    return;
  }

  // add to col>row
  for (int i = n; i--;) {
    if (indc[i] > r)
      (*this)(indc[i], r) += valc[i];
  }
}
*/

//___________________________________________________________
Double_t* SymMatrix::GetRow(Int_t r)
{
  if (r >= GetSize()) {
    int nn = GetSize();
    AddRows(r - GetSize() + 1);
    LOG(debug) << Form("create %d of %d\n", r, nn);
    return &((fElemsAdd[r - GetSizeBooked()])[0]);
  } else {
    return &fElems[GetIndex(r, 0)];
  }
}

//___________________________________________________________
int SymMatrix::SolveSpmInv(double* vecB, Bool_t stabilize)
{
  Int_t nRank = 0;
  int iPivot;
  double vPivot = 0.;
  double eps = 1e-14;
  int nGlo = GetSizeUsed();
  bool* bUnUsed = new bool[nGlo];
  double *rowMax, *colMax = nullptr;
  rowMax = new double[nGlo];

  if (stabilize) {
    colMax = new double[nGlo];
    for (Int_t i = nGlo; i--;) {
      rowMax[i] = colMax[i] = 0.0;
    }
    for (Int_t i = nGlo; i--;) {
      for (Int_t j = i + 1; j--;) {
        double vl = TMath::Abs(Query(i, j));
        if (IsZero(vl)) {
          continue;
        }
        if (vl > rowMax[i]) {
          rowMax[i] = vl; // Max elemt of row i
        }
        if (vl > colMax[j]) {
          colMax[j] = vl; // Max elemt of column j
        }
        if (i == j) {
          continue;
        }
        if (vl > rowMax[j]) {
          rowMax[j] = vl; // Max elemt of row j
        }
        if (vl > colMax[i]) {
          colMax[i] = vl; // Max elemt of column i
        }
      }
    }

    for (Int_t i = nGlo; i--;) {
      if (!IsZero(rowMax[i])) {
        rowMax[i] = 1. / rowMax[i]; // Max elemt of row i
      }
      if (!IsZero(colMax[i])) {
        colMax[i] = 1. / colMax[i]; // Max elemt of column i
      }
    }
  }

  for (Int_t i = nGlo; i--;) {
    bUnUsed[i] = true;
  }

  if (!fgBuffer || fgBuffer->GetSizeUsed() != GetSizeUsed()) {
    delete fgBuffer;
    fgBuffer = new SymMatrix(*this);
  } else {
    (*fgBuffer) = *this;
  }

  if (stabilize) {
    for (int i = 0; i < nGlo; i++) { // Small loop for matrix equilibration (gives a better conditioning)
      for (int j = 0; j <= i; j++) {
        double vl = Query(i, j);
        if (!IsZero(vl)) {
          SetEl(i, j, TMath::Sqrt(rowMax[i]) * vl * TMath::Sqrt(colMax[j])); // Equilibrate the V matrix
        }
      }
      for (int j = i + 1; j < nGlo; j++) {
        double vl = Query(j, i);
        if (!IsZero(vl)) {
          fgBuffer->SetEl(j, i, TMath::Sqrt(rowMax[i]) * vl * TMath::Sqrt(colMax[j])); // Equilibrate the V matrix
        }
      }
    }
  }
  for (Int_t j = nGlo; j--;) {
    fgBuffer->DiagElem(j) = TMath::Abs(QueryDiag(j)); // save diagonal elem absolute values
  }
  for (Int_t i = 0; i < nGlo; i++) {
    vPivot = 0.0;
    iPivot = -1;

    for (Int_t j = 0; j < nGlo; j++) { // First look for the pivot, ie max unused diagonal element
      double vl;
      if (bUnUsed[j] && (TMath::Abs(vl = QueryDiag(j)) > TMath::Max(TMath::Abs(vPivot), eps * fgBuffer->QueryDiag(j)))) {
        vPivot = vl;
        iPivot = j;
      }
    }

    if (iPivot >= 0) { // pivot found
      nRank++;
      bUnUsed[iPivot] = false; // This value is used
      vPivot = 1.0 / vPivot;
      DiagElem(iPivot) = -vPivot; // Replace pivot by its inverse
      //
      for (Int_t j = 0; j < nGlo; j++) {
        for (Int_t jj = 0; jj < nGlo; jj++) {
          if (j != iPivot && jj != iPivot) { // Other elements (!!! do them first as you use old matV[k][j]'s !!!)
            double& r = j >= jj ? (*this)(j, jj) : (*fgBuffer)(jj, j);
            r -= vPivot * (j > iPivot ? Query(j, iPivot) : fgBuffer->Query(iPivot, j)) * (iPivot > jj ? Query(iPivot, jj) : fgBuffer->Query(jj, iPivot));
          }
        }
      }

      for (Int_t j = 0; j < nGlo; j++)
        if (j != iPivot) { // Pivot row or column elements
          (*this)(j, iPivot) *= vPivot;
          (*fgBuffer)(iPivot, j) *= vPivot;
        }

    } else { // No more pivot value (clear those elements)
      for (Int_t j = 0; j < nGlo; j++) {
        if (bUnUsed[j]) {
          vecB[j] = 0.0;
          for (Int_t k = 0; k < nGlo; k++) {
            (*this)(j, k) = 0.;
            if (j != k) {
              (*fgBuffer)(j, k) = 0;
            }
          }
        }
      }
      break; // No more pivots anyway, stop here
    }
  }

  if (stabilize) {
    for (Int_t i = 0; i < nGlo; i++) {
      for (Int_t j = 0; j < nGlo; j++) {
        double vl = TMath::Sqrt(colMax[i]) * TMath::Sqrt(rowMax[j]); // Correct matrix V
        if (i >= j) {
          (*this)(i, j) *= vl;
        } else {
          (*fgBuffer)(j, i) *= vl;
        }
      }
    }
  }
  for (Int_t j = 0; j < nGlo; j++) {
    rowMax[j] = 0.0;
    for (Int_t jj = 0; jj < nGlo; jj++) { // Reverse matrix elements
      double vl;
      if (j >= jj) {
        vl = (*this)(j, jj) = -Query(j, jj);
      } else {
        vl = (*fgBuffer)(j, jj) = -fgBuffer->Query(j, jj);
      }
      rowMax[j] += vl * vecB[jj];
    }
  }

  for (Int_t j = 0; j < nGlo; j++) {
    vecB[j] = rowMax[j]; // The final result
  }

  delete[] bUnUsed;
  delete[] rowMax;
  if (stabilize) {
    delete[] colMax;
  }

  return nRank;
}
