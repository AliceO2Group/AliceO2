#include <string.h>
#include "MCHAlign/VectorSparse.h"
#include <TString.h>

using namespace o2::mch;

ClassImp(VectorSparse);

//___________________________________________________________
VectorSparse::VectorSparse()
  : fNElems(0),
    fIndex(nullptr),
    fElems(nullptr)
{
}

//___________________________________________________________
VectorSparse::VectorSparse(const VectorSparse& src)
  : TObject(src),
    fNElems(src.fNElems),
    fIndex(nullptr),
    fElems(nullptr)
{
  // copy c-tor
  fIndex = new unsigned short int[fNElems];
  fElems = new double[fNElems];
  memcpy(fIndex, src.fIndex, fNElems * sizeof(unsigned short int));
  memcpy(fElems, src.fElems, fNElems * sizeof(double));
}

//___________________________________________________________
void VectorSparse::Clear(Option_t*)
{
  // clear all
  delete[] fIndex;
  fIndex = nullptr;
  delete[] fElems;
  fElems = nullptr;
  fNElems = 0;
}

//___________________________________________________________
VectorSparse& VectorSparse::operator=(const VectorSparse& src)
{
  // assignment op-tor
  if (&src == this) {
    return *this;
  }
  Clear();
  TObject::operator=(src);
  fNElems = src.fNElems;
  fIndex = new unsigned short int[fNElems];
  fElems = new double[fNElems];
  memcpy(fIndex, src.fIndex, fNElems * sizeof(unsigned short int));
  memcpy(fElems, src.fElems, fNElems * sizeof(double));
  //
  return *this;
}

//___________________________________________________________
double VectorSparse::FindIndex(int ind) const
{
  // return an element with given index
  // printf("V: findindex\n");
  int first = 0;
  int last = fNElems - 1;
  while (first <= last) {
    int mid = (first + last) >> 1;
    if (ind > fIndex[mid]) {
      first = mid + 1;
    } else if (ind < fIndex[mid]) {
      last = mid - 1;
    } else {
      return fElems[mid];
    }
  }
  return 0.0;
}

//___________________________________________________________
void VectorSparse::SetToZero(int ind)
{
  // set element to 0 if it was already defined
  int first = 0;
  int last = fNElems - 1;
  while (first <= last) {
    int mid = (first + last) >> 1;
    if (ind > fIndex[mid]) {
      first = mid + 1;
    } else if (ind < fIndex[mid]) {
      last = mid - 1;
    } else {
      fElems[mid] = 0.;
      return;
    }
  }
}

//___________________________________________________________
double& VectorSparse::FindIndexAdd(int ind)
{
  // increment an element with given index
  // printf("V: findindexAdd\n");
  int first = 0;
  int last = fNElems - 1;
  while (first <= last) {
    int mid = (first + last) >> 1;
    if (ind > fIndex[mid]) {
      first = mid + 1;
    } else if (ind < fIndex[mid]) {
      last = mid - 1;
    } else {
      return fElems[mid];
    }
  }
  // need to insert a new element
  unsigned short int* arrI = new unsigned short int[fNElems + 1];
  memcpy(arrI, fIndex, first * sizeof(unsigned short int));
  arrI[first] = ind;
  memcpy(arrI + first + 1, fIndex + first, (fNElems - first) * sizeof(unsigned short int));
  delete[] fIndex;
  fIndex = arrI;
  //
  double* arrE = new double[fNElems + 1];
  memcpy(arrE, fElems, first * sizeof(double));
  arrE[first] = 0;
  memcpy(arrE + first + 1, fElems + first, (fNElems - first) * sizeof(double));
  delete[] fElems;
  fElems = arrE;
  //
  fNElems++;
  return fElems[first];
  //
}

//__________________________________________________________
void VectorSparse::ReSize(int sz, bool copy)
{
  // change the size
  if (sz < 1) {
    Clear();
    return;
  }
  // need to insert a new element
  unsigned short int* arrI = new unsigned short int[sz];
  double* arrE = new double[sz];
  memset(arrI, 0, sz * sizeof(unsigned short int));
  memset(arrE, 0, sz * sizeof(double));
  //
  if (copy && fIndex) {
    int cpsz = TMath::Min(fNElems, sz);
    memcpy(arrI, fIndex, cpsz * sizeof(unsigned short int));
    memcpy(arrE, fElems, cpsz * sizeof(double));
  }
  delete[] fIndex;
  delete[] fElems;
  fIndex = arrI;
  fElems = arrE;
  fNElems = sz;
  //
}

//__________________________________________________________
void VectorSparse::SortIndices(bool valuesToo)
{
  // sort indices in increasing order. Used to fix the row after ILUk decomposition
  for (int i = fNElems; i--;) {
    for (int j = i; j--;)
      if (fIndex[i] < fIndex[j]) { // swap
        unsigned short int tmpI = fIndex[i];
        fIndex[i] = fIndex[j];
        fIndex[j] = tmpI;
        if (valuesToo) {
          double tmpV = fElems[i];
          fElems[i] = fElems[j];
          fElems[j] = tmpV;
        }
      }
  }
}

//__________________________________________________________
void VectorSparse::Print(Option_t* opt) const
{
  // print itself
  TString sopt = opt;
  sopt.ToLower();
  int ndig = sopt.Atoi();
  if (ndig <= 1) {
    ndig = 2;
  }
  sopt = "%2d:%+.";
  sopt += ndig;
  sopt += "e |";
  printf("|");
  for (int i = 0; i < fNElems; i++) {
    printf(sopt.Data(), fIndex[i], fElems[i]);
  }
  printf("\n");
}

//___________________________________________________________
void VectorSparse::Add(double* valc, int* indc, int n)
{
  // add indiced array to row. Indices must be in increasing order
  int indx;
  int nadd = 0;
  //
  int last = fNElems - 1;
  int mid = 0;
  for (int i = n; i--;) {
    // if the element with this index is already defined, just add the value
    int first = 0;
    bool toAdd = true;
    indx = indc[i];
    while (first <= last) {
      mid = (first + last) >> 1;
      if (indx > fIndex[mid]) {
        first = mid + 1;
      } else if (indx < fIndex[mid]) {
        last = mid - 1;
      } else {
        fElems[mid] += valc[i];
        indc[i] = -1;
        toAdd = false;
        last = mid - 1; // profit from the indices being ordered
        break;
      }
    }
    if (toAdd) {
      nadd++;
    }
  }
  //
  if (nadd < 1) {
    return; // nothing to do anymore
  }
  //
  // need to expand the row
  unsigned short int* arrI = new unsigned short int[fNElems + nadd];
  double* arrE = new double[fNElems + nadd];
  // copy old elems embedding the new ones
  int inew = 0, iold = 0;
  for (int i = 0; i < n; i++) {
    if ((indx = indc[i]) < 0) {
      continue;
    }
    while (iold < fNElems && fIndex[iold] < indx) {
      arrI[inew] = fIndex[iold];
      arrE[inew++] = fElems[iold++];
    }
    arrI[inew] = indx;
    arrE[inew++] = valc[i];
  }
  // copy the rest
  while (iold < fNElems) {
    arrI[inew] = fIndex[iold];
    arrE[inew++] = fElems[iold++];
  }
  //
  delete[] fIndex;
  delete[] fElems;
  fIndex = arrI;
  fElems = arrE;
  //
  fNElems += nadd;
  //
}
