#ifndef ALICEO2_MCH_VECTORSPARSE_H
#define ALICEO2_MCH_VECTORSPARSE_H

#include <TObject.h>
#include <TMath.h>

namespace o2
{
namespace mch
{

class VectorSparse : public TObject
{
 public:
  VectorSparse();
  VectorSparse(const VectorSparse& src);
  ~VectorSparse() override { Clear(); }

  void Print(Option_t* option = "") const override;

  int GetNElems() const { return fNElems; }
  unsigned short int* GetIndices() const { return fIndex; }
  double* GetElems() const { return fElems; }
  unsigned short int& GetIndex(int i) { return fIndex[i]; }
  double& GetElem(int i) const { return fElems[i]; }
  void Clear(Option_t* option = "");
  void Reset() { memset(fElems, 0, fNElems * sizeof(double)); }
  void ReSize(int sz, bool copy = false);
  void SortIndices(bool valuesToo = false);
  void Add(double* valc, int* indc, int n);

  VectorSparse& operator=(const VectorSparse& src);

  virtual double operator()(int ind) const;
  virtual double& operator()(int ind);
  virtual void SetToZero(int ind);
  double FindIndex(int ind) const;
  double& FindIndexAdd(int ind);

  int GetLastIndex() const { return fIndex[fNElems - 1]; }
  double GetLastElem() const { return fElems[fNElems - 1]; }
  double& GetLastElem() { return fElems[fNElems - 1]; }

 protected:
  int fNElems;                // Number of elements
  unsigned short int* fIndex; // Index of stored elems
  double* fElems;             // pointer on elements

  ClassDefOverride(VectorSparse, 0)
};

//___________________________________________________
inline double VectorSparse::operator()(int ind) const
{
  return FindIndex(ind);
}

//___________________________________________________
inline double& VectorSparse::operator()(int ind)
{
  return FindIndexAdd(ind);
}

} // namespace mch
} // namespace o2

#endif
