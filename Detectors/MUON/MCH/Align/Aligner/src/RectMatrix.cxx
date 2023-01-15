#include "MCHAlign/RectMatrix.h"
#include <TString.h>

using namespace o2::mch;

ClassImp(RectMatrix);

//___________________________________________________________
RectMatrix::RectMatrix()
  : fNRows(0),
    fNCols(0),
    fRows(nullptr)
{
}

//___________________________________________________________
RectMatrix::RectMatrix(int nrow, int ncol)
  : fNRows(nrow),
    fNCols(ncol),
    fRows(nullptr)
{
  // c-tor
  fRows = new double*[fNRows];
  for (int i = fNRows; i--;) {
    fRows[i] = new double[fNCols];
    memset(fRows[i], 0, fNCols * sizeof(double));
  }
  //
}

//___________________________________________________________
RectMatrix::RectMatrix(const RectMatrix& src)
  : TObject(src),
    fNRows(src.fNRows),
    fNCols(src.fNCols),
    fRows(nullptr)
{
  // copy c-tor
  fRows = new double*[fNRows];
  for (int i = fNRows; i--;) {
    fRows[i] = new double[fNCols];
    memcpy(fRows[i], src.fRows[i], fNCols * sizeof(double));
  }
}

//___________________________________________________________
RectMatrix::~RectMatrix()
{
  // dest-tor
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
  // assignment op-r
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
  fRows = new double*[fNRows];
  for (int i = fNRows; i--;) {
    fRows[i] = new double[fNCols];
    memcpy(fRows[i], src.fRows[i], fNCols * sizeof(double));
  }
  //
  return *this;
}

//___________________________________________________________
void RectMatrix::Print(Option_t* option) const
{
  // print itself
  printf("Rectangular Matrix:  %d rows %d columns\n", fNRows, fNCols);
  TString opt = option;
  opt.ToLower();
  if (opt.IsNull()) {
    return;
  }
  for (int i = 0; i < fNRows; i++) {
    for (int j = 0; j <= fNCols; j++) {
      printf("%+.3e|", Query(i, j));
    }
    printf("\n");
  }
}

//___________________________________________________________
void RectMatrix::Reset() const
{
  // reset all
  for (int i = fNRows; i--;) {
    double* row = GetRow(i);
    for (int j = fNCols; j--;) {
      row[j] = 0.;
    }
  }
}
