#ifndef ALICEO2_MCH_SYMBDMATRIX_H
#define ALICEO2_MCH_SYMBDMATRIX_H

#include <TObject.h>
#include <TVectorD.h>
#include "MCHAlign/MatrixSq.h"

namespace o2
{
namespace mch
{

class SymBDMatrix : public MatrixSq
{

 public:
  enum { kDecomposedBit = 0x1 };

  SymBDMatrix();
  SymBDMatrix(int size, int w = 0);
  SymBDMatrix(const SymBDMatrix& mat);
  ~SymBDMatrix() override;

  int GetBandHWidth() const { return fNrows; }
  int GetNElemsStored() const { return fNelems; }
  void Clear(Option_t* option = "") override;
  void Reset() override;

  float GetDensity() const override;
  SymBDMatrix& operator=(const SymBDMatrix& src);
  double operator()(int rown, int coln) const override;
  double& operator()(int rown, int coln) override;
  double operator()(int rown) const;
  double& operator()(int rown);

  double DiagElem(int r) const override { return (*(const SymBDMatrix*)this)(r, r); }
  double& DiagElem(int r) override { return (*this)(r, r); }
  void DecomposeLDLT();
  void Solve(double* rhs);
  void Solve(const double* rhs, double* sol);
  void Solve(TVectorD& rhs) { Solve(rhs.GetMatrixArray()); }
  void Solve(const TVectorD& rhs, TVectorD& sol) { Solve(rhs.GetMatrixArray(), sol.GetMatrixArray()); }

  void Print(Option_t* option = "") const override;
  void SetDecomposed(bool v = true) { SetBit(kDecomposedBit, v); }
  bool IsDecomposed() const { return TestBit(kDecomposedBit); }

  void MultiplyByVec(const double* vecIn, double* vecOut) const override;
  void MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const override;
  void AddToRow(int r, double* valc, int* indc, int n);

  virtual int GetIndex(int row, int col) const;
  virtual int GetIndex(int diagID) const;
  double GetEl(int row, int col) const { return operator()(row, col); }
  void SetEl(int row, int col, double val) { operator()(row, col) = val; }
  //
 protected:
  double* fElems; //   Elements booked by constructor
  //
  ClassDefOverride(SymBDMatrix, 0) // Symmetric Matrix Class
};

//___________________________________________________________
inline int SymBDMatrix::GetIndex(int row, int col) const
{
  // lower triangle band is actually filled
  if (row < col) {
    Swap(row, col);
  }
  col -= row;
  if (col < -GetBandHWidth()) {
    return -1;
  }
  return GetIndex(row) + col;
}

//___________________________________________________________
inline int SymBDMatrix::GetIndex(int diagID) const
{
  // Get index of the diagonal element on row diagID
  return (diagID + 1) * fRowLwb - 1;
}

//___________________________________________________________
inline double SymBDMatrix::operator()(int row, int col) const
{
  // query element
  int idx = GetIndex(row, col);
  return (const double&)idx < 0 ? 0.0 : fElems[idx];
}

//___________________________________________________________
inline double& SymBDMatrix::operator()(int row, int col)
{
  // get element for assingment; assignment outside of the stored range has no effect
  int idx = GetIndex(row, col);
  if (idx >= 0) {
    return fElems[idx];
  }
  fTol = 0;
  return fTol;
}

//___________________________________________________________
inline double SymBDMatrix::operator()(int row) const
{
  // query diagonal
  return (const double&)fElems[GetIndex(row)];
}

//___________________________________________________________
inline double& SymBDMatrix::operator()(int row)
{
  // get diagonal for assingment; assignment outside of the stored range has no effect
  return fElems[GetIndex(row)];
}

//___________________________________________________________
inline void SymBDMatrix::MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const
{
  MultiplyByVec(vecIn.GetMatrixArray(), vecOut.GetMatrixArray());
}

} // namespace mch
} // namespace o2

#endif
