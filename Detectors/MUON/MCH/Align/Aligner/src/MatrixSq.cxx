#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "TClass.h"
#include "TMath.h"

#include "Framework/Logger.h"
#include "MCHAlign/MatrixSq.h"

using namespace o2::mch;

ClassImp(MatrixSq);

//___________________________________________________________
MatrixSq::MatrixSq(const MatrixSq& src)
  : TMatrixDBase(src),
    fSymmetric(src.fSymmetric)
{
  LOG(debug) << "copy ctor";
}

//___________________________________________________________
MatrixSq& MatrixSq::operator=(const MatrixSq& src)
{
  // = operator
  if (this == &src) {
    return *this;
  }
  TMatrixDBase::operator=(src);
  fSymmetric = src.fSymmetric;
  return *this;
}

//___________________________________________________________
void MatrixSq::MultiplyByVec(const double* vecIn, double* vecOut) const
{
  // fill vecOut by matrix*vecIn
  // vector should be of the same size as the matrix
  for (int i = GetSize(); i--;) {
    vecOut[i] = 0.0;
    for (int j = GetSize(); j--;) {
      vecOut[i] += vecIn[j] * (*this)(i, j);
    }
  }
  //
}

//___________________________________________________________
void MatrixSq::PrintCOO() const
{
  // print matrix in COO sparse format
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
