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

/// @file MatrixSq.cxx

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "TClass.h"
#include "TMath.h"
#include "MFTAlignment/MatrixSq.h"

using namespace o2::mft;

ClassImp(MatrixSq);

//___________________________________________________________
MatrixSq& MatrixSq::operator=(const MatrixSq& src)
{
  if (this == &src) {
    return *this;
  }
  TMatrixDBase::operator=(src);
  fSymmetric = src.fSymmetric;
  return *this;
}

//___________________________________________________________
void MatrixSq::MultiplyByVec(const Double_t* vecIn, Double_t* vecOut) const
{
  for (int i = GetSize(); i--;) {
    vecOut[i] = 0.0;
    for (int j = GetSize(); j--;) {
      vecOut[i] += vecIn[j] * (*this)(i, j);
    }
  }
}

//___________________________________________________________
void MatrixSq::PrintCOO() const
{
  // get number of non-zero elements
  int nnz = 0;
  int sz = GetSize();
  for (int ir = 0; ir < sz; ir++) {
    for (int ic = 0; ic < sz; ic++) {
      if (Query(ir, ic) != 0) {
        nnz++;
      }
    }
  }

  printf("%d %d %d\n", sz, sz, nnz);
  double vl;
  for (int ir = 0; ir < sz; ir++) {
    for (int ic = 0; ic < sz; ic++) {
      if ((vl = Query(ir, ic)) != 0) {
        printf("%d %d %f\n", ir, ic, vl);
      }
    }
  }
}
