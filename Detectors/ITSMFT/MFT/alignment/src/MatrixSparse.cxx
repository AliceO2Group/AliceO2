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

/// @file MatrixSparse.cxx

#include <iomanip>
#include <TStopwatch.h>

#include "Framework/Logger.h"
#include "MFTAlignment/MatrixSparse.h"

using namespace o2::mft;

ClassImp(MatrixSparse);

//___________________________________________________________
MatrixSparse::MatrixSparse(Int_t sz)
  : MatrixSq(),
    fVecs(0)
{
  fNcols = fNrows = sz;

  fVecs = new VectorSparse*[sz];
  for (int i = GetSize(); i--;) {
    fVecs[i] = new VectorSparse();
  }
}

//___________________________________________________________
MatrixSparse::MatrixSparse(const MatrixSparse& src)
  : MatrixSq(src),
    fVecs(0)
{
  fVecs = new VectorSparse*[src.GetSize()];
  for (int i = GetSize(); i--;) {
    fVecs[i] = new VectorSparse(*src.GetRow(i));
  }
}

//___________________________________________________________
VectorSparse* MatrixSparse::GetRowAdd(Int_t ir)
{
  // get row, add if needed
  if (ir >= fNrows) {
    VectorSparse** arrv = new VectorSparse*[ir + 1];
    for (int i = GetSize(); i--;) {
      arrv[i] = fVecs[i];
    }
    delete[] fVecs;
    fVecs = arrv;
    for (int i = GetSize(); i <= ir; i++) {
      fVecs[i] = new VectorSparse();
    }
    fNrows = ir + 1;
    if (IsSymmetric() && fNcols < fNrows) {
      fNcols = fNrows;
    }
  }
  return fVecs[ir];
}

//___________________________________________________________
MatrixSparse& MatrixSparse::operator=(const MatrixSparse& src)
{
  if (this == &src)
    return *this;
  MatrixSq::operator=(src);

  Clear();
  fNcols = src.GetNCols();
  fNrows = src.GetNRows();
  SetSymmetric(src.IsSymmetric());
  fVecs = new VectorSparse*[fNrows];
  for (int i = fNrows; i--;) {
    fVecs[i] = new VectorSparse(*src.GetRow(i));
  }
  return *this;
}

//___________________________________________________________
void MatrixSparse::Clear(Option_t*)
{
  for (int i = fNrows; i--;) {
    delete GetRow(i);
  }
  delete[] fVecs;
  fNcols = fNrows = 0;
}

//___________________________________________________________
void MatrixSparse::Print(Option_t* opt) const
{
  LOG(info) << "Sparse Matrix of size " << fNrows << " x " << fNcols;
  if (IsSymmetric()) {
    LOG(info) << " (Symmetric)\n";
  }
  for (int i = 0; i < fNrows; i++) {
    VectorSparse* row = GetRow(i);
    if (!row->GetNElems()) {
      continue;
    }
    printf("%3d: ", i);
    row->Print(opt);
  }
}

//___________________________________________________________
void MatrixSparse::MultiplyByVec(const Double_t* vecIn, Double_t* vecOut) const
{
  memset(vecOut, 0, GetSize() * sizeof(Double_t));

  for (int rw = GetSize(); rw--;) { // loop over rows >>>
    const VectorSparse* rowV = GetRow(rw);
    Int_t nel = rowV->GetNElems();
    if (!nel) {
      continue;
    }

    UShort_t* indV = rowV->GetIndices();
    Double_t* elmV = rowV->GetElems();

    if (IsSymmetric()) {
      // treat diagonal term separately. If filled, it should be the last one
      if (indV[--nel] == rw) {
        vecOut[rw] += vecIn[rw] * elmV[nel];
      } else {
        nel = rowV->GetNElems(); // diag elem was not filled
      }
      for (int iel = nel; iel--;) { // less element retrieval for symmetric case
        if (elmV[iel]) {
          vecOut[rw] += vecIn[indV[iel]] * elmV[iel];
          vecOut[indV[iel]] += vecIn[rw] * elmV[iel];
        }
      }
    } else {
      for (int iel = nel; iel--;) {
        if (elmV[iel]) {
          vecOut[rw] += vecIn[indV[iel]] * elmV[iel];
        }
      }
    }
  } // loop over rows <<<
}

//___________________________________________________________
void MatrixSparse::SortIndices(Bool_t valuesToo)
{
  TStopwatch sw;
  sw.Start();
  LOG(info) << "MatrixSparse:SortIndices >>";
  for (int i = GetSize(); i--;) {
    GetRow(i)->SortIndices(valuesToo);
  }
  sw.Stop();
  sw.Print();
  LOG(info) << "MatrixSparse:SortIndices <<";
}

//___________________________________________________________
void MatrixSparse::AddToRow(Int_t r, Double_t* valc, Int_t* indc, Int_t n)
{
  // for sym. matrix count how many elems to add have row >= col and assign excplicitly
  // those which have row < col

  // range in increasing order of indices
  for (int i = n; i--;) {
    for (int j = i; j >= 0; j--) {
      if (indc[j] > indc[i]) { // swap
        int ti = indc[i];
        indc[i] = indc[j];
        indc[j] = ti;
        double tv = valc[i];
        valc[i] = valc[j];
        valc[j] = tv;
      }
    }
  }

  int ni = n;
  if (IsSymmetric()) {
    while (ni--) {
      if (indc[ni] > r) {
        (*this)(indc[ni], r) += valc[ni];
      } else {
        break; // use the fact that the indices are ranged in increasing order
      }
    }
  }

  if (ni < 0) {
    return;
  }
  VectorSparse* row = GetRowAdd(r);
  row->Add(valc, indc, ni + 1);
}

//___________________________________________________________
Float_t MatrixSparse::GetDensity() const
{

  Int_t nel = 0;
  for (int i = GetSize(); i--;) {
    nel += GetRow(i)->GetNElems();
  }
  int den = IsSymmetric() ? (GetSize() + 1) * GetSize() / 2 : GetSize() * GetSize();
  return float(nel) / den;
}
