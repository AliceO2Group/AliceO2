#ifndef ALICEO2_MCH_MATRIXSQ_H
#define ALICEO2_MCH_MATRIXSQ_H

#include <TMatrixDBase.h>
#include <TVectorD.h>

namespace o2
{
namespace mch
{

/// \class MatrixSq
class MatrixSq : public TMatrixDBase
{

 public:
  MatrixSq() : fSymmetric(false) {}
  MatrixSq(const MatrixSq& src);
  ~MatrixSq() override = default;

  MatrixSq& operator=(const MatrixSq& src);

  virtual int GetSize() const { return fNcols; }
  virtual float GetDensity() const = 0;

  void Clear(Option_t* option = "") override = 0;

  virtual double Query(int rown, int coln) const { return operator()(rown, coln); }
  double operator()(int rown, int coln) const override = 0;
  double& operator()(int rown, int coln) override = 0;

  virtual double QueryDiag(int rc) const { return DiagElem(rc); }
  virtual double DiagElem(int r) const = 0;
  virtual double& DiagElem(int r) = 0;
  virtual void AddToRow(int r, double* valc, int* indc, int n) = 0;

  virtual void Print(Option_t* option = "") const override = 0;
  virtual void Reset() = 0;
  virtual void PrintCOO() const;

  virtual void MultiplyByVec(const double* vecIn, double* vecOut) const;
  virtual void MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const;

  bool IsSymmetric() const override { return fSymmetric; }
  void SetSymmetric(bool v = true) { fSymmetric = v; }

  // ---------------------------------- Dummy methods of MatrixBase
  const double* GetMatrixArray() const override
  {
    Error("GetMatrixArray", "Dummy");
    return nullptr;
  };
  double* GetMatrixArray() override
  {
    Error("GetMatrixArray", "Dummy");
    return nullptr;
  };
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
  virtual void Allocate(int, int, int, int, int, int)
  {
    Error("Allocate", "Dummy");
    return;
  }

  static bool IsZero(double x, double thresh = 1e-64) { return x > 0 ? (x < thresh) : (x > -thresh); }

 protected:
  void Swap(int& r, int& c) const
  {
    int t = r;
    r = c;
    c = t;
  }

 protected:
  bool fSymmetric; ///< is the matrix symmetric? Only lower triangle is filled

  ClassDefOverride(MatrixSq, 1);
};

//___________________________________________________________
inline void MatrixSq::MultiplyByVec(const TVectorD& vecIn, TVectorD& vecOut) const
{
  MultiplyByVec(vecIn.GetMatrixArray(), vecOut.GetMatrixArray());
}

} // namespace mch
} // namespace o2

#endif
