#ifndef ALICEO2_MCH_RECTMATRIX_H
#define ALICEO2_MCH_RECTMATRIX_H

#include "TObject.h"
class TString;

namespace o2
{
namespace mch
{

class RectMatrix : public TObject
{

 public:
  RectMatrix();
  RectMatrix(int nrow, int ncol);
  RectMatrix(const RectMatrix& src);
  ~RectMatrix() override;

  int GetNRows() const { return fNRows; }
  int GetNCols() const { return fNCols; }

  double Query(int rown, int coln) const { return operator()(rown, coln); }

  RectMatrix& operator=(const RectMatrix& src);
  double operator()(int rown, int coln) const;
  double& operator()(int rown, int coln);
  double* operator()(int row) const { return GetRow(row); }
  double* GetRow(int row) const { return fRows[row]; }

  void Reset() const;

  void Print(Option_t* option = "") const override;

 protected:
  int fNRows;     // Number of rows
  int fNCols;     // Number of columns
  double** fRows; // pointers on rows

  ClassDefOverride(RectMatrix, 0) // Rectangular Matrix Class
};

//___________________________________________________________
inline double RectMatrix::operator()(int row, int col) const
{
  return (const double&)GetRow(row)[col];
}

//___________________________________________________________
inline double& RectMatrix::operator()(int row, int col)
{
  return (double&)fRows[row][col];
}

} // namespace mch
} // namespace o2

#endif
