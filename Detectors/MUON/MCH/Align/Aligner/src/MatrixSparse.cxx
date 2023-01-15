#include "MCHAlign/MatrixSparse.h"

#include "TStopwatch.h"

using namespace o2::mch;
//___________________________________________________________
ClassImp(MatrixSparse);

//___________________________________________________________
MatrixSparse::MatrixSparse(int sz)
  : MatrixSq(),
    fVecs(nullptr)
{

  // constructor
  fNcols = fNrows = sz;
  //
  fVecs = new VectorSparse*[sz];
  for (int i = GetSize(); i--;) {
    fVecs[i] = new VectorSparse();
  }
}

//___________________________________________________________
MatrixSparse::MatrixSparse(const MatrixSparse& src)
  : MatrixSq(src),
    fVecs(nullptr)
{
  // copy c-tor
  fVecs = new VectorSparse*[src.GetSize()];
  for (int i = GetSize(); i--;) {
    fVecs[i] = new VectorSparse(*src.GetRow(i));
  }
}

//___________________________________________________________
VectorSparse* MatrixSparse::GetRowAdd(int ir)
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
  // assignment op-r
  if (this == &src) {
    return *this;
  }
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
  // clear
  for (int i = fNrows; i--;) {
    delete GetRow(i);
  }
  delete[] fVecs;
  fNcols = fNrows = 0;
}

//___________________________________________________________
void MatrixSparse::Print(Option_t* opt) const
{
  // print itself
  printf("Sparse Matrix of size %d x %d %s\n", fNrows, fNcols,
         IsSymmetric() ? " (Symmetric)" : "");
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
void MatrixSparse::MultiplyByVec(const double* vecIn,
                                 double* vecOut) const
{
  // fill vecOut by matrix*vecIn
  // vector should be of the same size as the matrix
  //
  memset(vecOut, 0, GetSize() * sizeof(double));
  //
  for (int rw = GetSize(); rw--;) { // loop over rows >>>
    const VectorSparse* rowV = GetRow(rw);
    int nel = rowV->GetNElems();
    if (!nel) {
      continue;
    }
    //
    unsigned short int* indV = rowV->GetIndices();
    double* elmV = rowV->GetElems();
    //
    if (IsSymmetric()) {
      // treat diagonal term separately. If filled, it should be the last one
      if (indV[--nel] == rw) {
        vecOut[rw] += vecIn[rw] * elmV[nel];
      } else {
        nel = rowV->GetNElems(); // diag elem was not filled
      }
      //
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
    //
  } // loop over rows <<<
  //
}

//___________________________________________________________
void MatrixSparse::SortIndices(bool valuesToo)
{
  // sort columns in increasing order. Used to fix the matrix after ILUk
  // decompostion
  TStopwatch sw;
  sw.Start();
  printf("MatrixSparse:sort>>\n");
  for (int i = GetSize(); i--;)
    GetRow(i)->SortIndices(valuesToo);
  sw.Stop();
  sw.Print();
  printf("MatrixSparse:sort<<\n");
}

//___________________________________________________________
void MatrixSparse::AddToRow(int r, double* valc, int* indc, int n)
{
  // for sym. matrix count how many elems to add have row>=col and assign
  // excplicitly those which have row<col
  //
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
  //
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
  //
  if (ni < 0) {
    return;
  }
  VectorSparse* row = GetRowAdd(r);
  row->Add(valc, indc, ni + 1);
}

//___________________________________________________________
float MatrixSparse::GetDensity() const
{
  // get fraction of non-zero elements
  int nel = 0;
  for (int i = GetSize(); i--;) {
    nel += GetRow(i)->GetNElems();
  }
  int den =
    IsSymmetric() ? (GetSize() + 1) * GetSize() / 2 : GetSize() * GetSize();
  return float(nel) / den;
}
