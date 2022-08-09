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

/// @file RectMatrix.cxx

#include <TString.h>
#include "MFTAlignment/RectMatrix.h"

using namespace o2::mft;

ClassImp(RectMatrix);

//___________________________________________________________
RectMatrix::RectMatrix()
  : fNRows(0),
    fNCols(0),
    fRows(0)
{
}

//___________________________________________________________
RectMatrix::RectMatrix(Int_t nrow, Int_t ncol)
  : fNRows(nrow),
    fNCols(ncol),
    fRows(0)
{
  fRows = new Double_t*[fNRows];
  for (int i = fNRows; i--;) {
    fRows[i] = new Double_t[fNCols];
    memset(fRows[i], 0, fNCols * sizeof(Double_t));
  }
}

//___________________________________________________________
RectMatrix::RectMatrix(const RectMatrix& src)
  : TObject(src),
    fNRows(src.fNRows),
    fNCols(src.fNCols),
    fRows(0)
{
  fRows = new Double_t*[fNRows];
  for (int i = fNRows; i--;) {
    fRows[i] = new Double_t[fNCols];
    memcpy(fRows[i], src.fRows[i], fNCols * sizeof(Double_t));
  }
}

//___________________________________________________________
RectMatrix::~RectMatrix()
{
  if (fNRows) {
    for (int i = fNRows; i--;) {
      delete[] fRows[i];
    }
  }
  delete[] fRows;
}

//___________________________________________________________
RectMatrix& RectMatrix::operator=(const RectMatrix& src)
{
  if (&src == this) {
    return *this;
  }
  if (fNRows) {
    for (int i = fNRows; i--;) {
      delete[] fRows[i];
    }
  }
  delete[] fRows;
  fNRows = src.fNRows;
  fNCols = src.fNCols;
  fRows = new Double_t*[fNRows];
  for (int i = fNRows; i--;) {
    fRows[i] = new Double_t[fNCols];
    memcpy(fRows[i], src.fRows[i], fNCols * sizeof(Double_t));
  }

  return *this;
}

//___________________________________________________________
void RectMatrix::Print(Option_t* option) const
{
  printf("Rectangular Matrix:  %d rows %d columns\n", fNRows, fNCols);
  TString opt = option;
  opt.ToLower();
  if (opt.IsNull()) {
    return;
  }
  for (int i = 0; i < fNRows; i++) {
    for (Int_t j = 0; j <= fNCols; j++) {
      printf("%+.3e|", Query(i, j));
    }
    printf("\n");
  }
}

//___________________________________________________________
void RectMatrix::Reset() const
{
  for (int i = fNRows; i--;) {
    double* row = GetRow(i);
    for (int j = fNCols; j--;) {
      row[j] = 0.;
    }
  }
}
