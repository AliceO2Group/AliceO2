#ifndef ALICEO2_MCH_SYMMATRIX_H
#define ALICEO2_MCH_SYMMATRIX_H

#include <TVectorD.h>
#include <TString.h>
#include "MCHAlign/MatrixSq.h"

namespace o2
{
namespace mch
{

class SymMatrix : public MatrixSq
{
  //
 public:
  SymMatrix();
  SymMatrix(int size);
  SymMatrix(const SymMatrix& mat);
  ~SymMatrix() override;

  void Clear(Option_t* option = "") override;
  void Reset() override;

  int GetSize() const override { return fNrowIndex; }
  int GetSizeUsed() const { return fRowLwb; }
  int GetSizeBooked() const { return fNcols; }
  int GetSizeAdded() const { return fNrows; }
  float GetDensity() const override;
  SymMatrix& operator=(const SymMatrix& src);
  SymMatrix& operator+=(const SymMatrix& src);
  SymMatrix& operator-=(const SymMatrix& src);

  double operator()(int rown, int coln) const override;
  double& operator()(int rown, int coln) override;

  double DiagElem(int r) const override { return (*(const SymMatrix*)this)(r, r); }
  double& DiagElem(int r) override { return (*this)(r, r); }

  double* GetRow(int r);

  void Print(const Option_t* option = "") const override;
  void AddRows(int nrows = 1);
  void SetSizeUsed(int sz) { fRowLwb = sz; }

  void Scale(double coeff);
  bool Multiply(const SymMatrix& right);
  void MultiplyByVec(const double* vecIn, double* vecOut) const override;
  void MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const override;
  void AddToRow(int r, double* valc, int* indc, int n) override;

  // ---------------------------------- Dummy methods of MatrixBase
  const double* GetMatrixArray() const override { return fElems; };
  double* GetMatrixArray() override { return (double*)fElems; }
  const int* GetRowIndexArray() const override
  {
    Error("GetRowIndexArray", "Dummy");
    return nullptr;
  };
  int* GetRowIndexArray() override
  {
    Error("GetRowIndexArray", "Dummy");
    return nullptr;
  };
  const int* GetColIndexArray() const override
  {
    Error("GetColIndexArray", "Dummy");
    return nullptr;
  };
  int* GetColIndexArray() override
  {
    Error("GetColIndexArray", "Dummy");
    return nullptr;
  };
  TMatrixDBase& SetRowIndexArray(int*) override
  {
    Error("SetRowIndexArray", "Dummy");
    return *this;
  }
  TMatrixDBase& SetColIndexArray(int*) override
  {
    Error("SetColIndexArray", "Dummy");
    return *this;
  }
  TMatrixDBase& GetSub(int, int, int, int, TMatrixDBase&, Option_t*) const override
  {
    Error("GetSub", "Dummy");
    return *((TMatrixDBase*)this);
  }
  TMatrixDBase& SetSub(int, int, const TMatrixDBase&) override
  {
    Error("GetSub", "Dummy");
    return *this;
  }
  TMatrixDBase& ResizeTo(int, int, int) override
  {
    Error("ResizeTo", "Dummy");
    return *this;
  }
  TMatrixDBase& ResizeTo(int, int, int, int, int) override
  {
    Error("ResizeTo", "Dummy");
    return *this;
  }

  // ----------------------------- Choleski methods ----------------------------------------
  SymMatrix* DecomposeChol();        // Obtain Cholesky decomposition L matrix
  void InvertChol(SymMatrix* mchol); // Invert using provided Choleski decomposition
  bool InvertChol();                 // Invert
  bool SolveChol(double* brhs, bool invert = false);
  bool SolveChol(double* brhs, double* bsol, bool invert = false);
  bool SolveChol(TVectorD& brhs, bool invert = false);
  bool SolveChol(const TVectorD& brhs, TVectorD& bsol, bool invert = false);
  bool SolveCholN(double* bn, int nRHS, bool invert = false);
  //
  int SolveSpmInv(double* vecB, bool stabilize = true);

 protected:
  virtual int GetIndex(int row, int col) const;
  double GetEl(int row, int col) const { return operator()(row, col); }
  void SetEl(int row, int col, double val) { operator()(row, col) = val; }
  //
 protected:
  double* fElems;     //   Elements booked by constructor
  double** fElemsAdd; //   Elements (rows) added dynamicaly
  //
  static SymMatrix* fgBuffer;    // buffer for fast solution
  static int fgCopyCnt;          // matrix copy counter
  ClassDefOverride(SymMatrix, 0) // Symmetric Matrix Class
};

//___________________________________________________________
inline int SymMatrix::GetIndex(int row, int col) const
{
  // lower triangle is actually filled
  return ((row * (row + 1)) >> 1) + col;
}

//___________________________________________________________
inline double SymMatrix::operator()(int row, int col) const
{
  //
  if (row < col) {
    Swap(row, col);
  }
  if (row >= fNrowIndex) {
    return 0;
  }
  return (const double&)(row < fNcols ? fElems[GetIndex(row, col)] : (fElemsAdd[row - fNcols])[col]);
}

//___________________________________________________________
inline double& SymMatrix::operator()(int row, int col)
{
  if (row < col) {
    Swap(row, col);
  }
  if (row >= fNrowIndex) {
    AddRows(row - fNrowIndex + 1);
  }
  return (row < fNcols ? fElems[GetIndex(row, col)] : (fElemsAdd[row - fNcols])[col]);
}

//___________________________________________________________
inline void SymMatrix::MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const
{
  MultiplyByVec(vecIn.GetMatrixArray(), vecOut.GetMatrixArray());
}

//___________________________________________________________
inline void SymMatrix::Scale(double coeff)
{
  for (int i = fNrowIndex; i--;) {
    for (int j = i; j--;) {
      double& el = operator()(i, j);
      if (el) {
        el *= coeff;
      }
    }
  }
}

//___________________________________________________________
inline void SymMatrix::AddToRow(int r, double* valc, int* indc, int n)
{
  for (int i = n; i--;) {
    (*this)(indc[i], r) += valc[i];
  }
}

} // namespace mch
} // namespace o2

#endif
