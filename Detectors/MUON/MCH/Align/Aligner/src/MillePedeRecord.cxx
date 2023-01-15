#include "MCHAlign/MillePedeRecord.h"
#include <TMath.h>
#include "Framework/Logger.h"

using namespace o2::mch;

ClassImp(MillePedeRecord);

//_____________________________________________________________________________________________
MillePedeRecord::MillePedeRecord()
  : fSize(0),
    fNGroups(0),
    fRunID(0),
    fGroupID(nullptr),
    fIndex(nullptr),
    fValue(nullptr),
    fWeight(1)
{
  SetUniqueID(0);
}

//_____________________________________________________________________________________________
MillePedeRecord::MillePedeRecord(const MillePedeRecord& src)
  : TObject(src),
    fSize(src.fSize),
    fNGroups(src.fNGroups),
    fRunID(src.fRunID),
    fGroupID(nullptr),
    fIndex(nullptr),
    fValue(nullptr),
    fWeight(src.fWeight)
{
  // copy ct-r
  fIndex = new int[GetDtBufferSize()];
  memcpy(fIndex, src.fIndex, fSize * sizeof(int));
  fValue = new double[GetDtBufferSize()];
  memcpy(fValue, src.fValue, fSize * sizeof(double));
  fGroupID = new unsigned short int[GetGrBufferSize()];
  memcpy(fGroupID, src.fGroupID, GetGrBufferSize() * sizeof(unsigned short int));
}

//_____________________________________________________________________________________________
MillePedeRecord& MillePedeRecord::operator=(const MillePedeRecord& rhs)
{
  // assignment op-r
  if (this != &rhs) {
    Reset();
    for (int i = 0; i < rhs.GetSize(); i++) {
      double val;
      int ind;
      rhs.GetIndexValue(i, ind, val);
      AddIndexValue(ind, val);
    }
    fWeight = rhs.fWeight;
    fRunID = rhs.fRunID;
    for (int i = 0; i < rhs.GetNGroups(); i++) {
      MarkGroup(rhs.GetGroupID(i));
    }
  }
  return *this;
}

//_____________________________________________________________________________________________
MillePedeRecord::~MillePedeRecord()
{
  delete[] fIndex;
  delete[] fValue;
  delete[] fGroupID;
}

//_____________________________________________________________________________________________
void MillePedeRecord::Reset()
{
  // reset all
  fSize = 0;
  for (int i = fNGroups; i--;) {
    fGroupID[i] = 0;
  }
  fNGroups = 0;
  fRunID = 0;
  fWeight = 1.;
}

//_____________________________________________________________________________________________
void MillePedeRecord::Print(const Option_t*) const
{
  // print itself
  if (!fSize) {
    LOG(info) << "No data";
    return;
  }
  int cnt = 0, point = 0;
  //
  if (fNGroups) {
    printf("Groups: ");
  }
  for (int i = 0; i < fNGroups; i++) {
    printf("%4d |", GetGroupID(i));
  }
  printf("Run: %9d Weight: %+.2e\n", fRunID, fWeight);
  while (cnt < fSize) {
    //
    double resid = fValue[cnt++];
    double* derLoc = GetValue() + cnt;
    int* indLoc = GetIndex() + cnt;
    int nLoc = 0;
    while (!IsWeight(cnt)) {
      nLoc++;
      cnt++;
    }
    double weight = GetValue(cnt++);
    double* derGlo = GetValue() + cnt;
    int* indGlo = GetIndex() + cnt;
    int nGlo = 0;
    while (!IsResidual(cnt) && cnt < fSize) {
      nGlo++;
      cnt++;
    }
    //
    printf("\n*** Point#%2d | Residual = %+.4e | Weight = %+.4e\n", point++, resid, weight);
    printf("Locals : ");
    for (int i = 0; i < nLoc; i++) {
      printf("[%5d] %+.4e|", indLoc[i], derLoc[i]);
    }
    printf("\n");
    printf("Globals: ");
    for (int i = 0; i < nGlo; i++) {
      printf("[%5d] %+.4e|", indGlo[i], derGlo[i]);
    }
    printf("\n");
    //
  }
  //
}

//_____________________________________________________________________________________________
double MillePedeRecord::GetGloResWProd(int indx) const
{
  // get sum of derivative over global variable indx * res. at point * weight
  if (!fSize) {
    LOG(info) << "No data";
    return 0;
  }
  int cnt = 0;
  double prodsum = 0.0;
  //
  while (cnt < fSize) {
    //
    double resid = fValue[cnt++];
    while (!IsWeight(cnt)) {
      cnt++;
    }
    double weight = GetValue(cnt++);
    double* derGlo = GetValue() + cnt;
    int* indGlo = GetIndex() + cnt;
    int nGlo = 0;
    while (!IsResidual(cnt) && cnt < fSize) {
      nGlo++;
      cnt++;
    }
    for (int i = nGlo; i--;) {
      if (indGlo[i] == indx) {
        prodsum += resid * weight * derGlo[i];
      }
    }
    //
  }
  return prodsum;
}

//_____________________________________________________________________________________________
double MillePedeRecord::GetGlobalDeriv(int pnt, int indx) const
{
  // get derivative over global variable indx at point pnt
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  //
  while (cnt < fSize) {
    //
    cnt++;
    while (!IsWeight(cnt)) {
      cnt++;
    }
    cnt++;
    double* derGlo = GetValue() + cnt;
    int* indGlo = GetIndex() + cnt;
    int nGlo = 0;
    while (!IsResidual(cnt) && cnt < fSize) {
      nGlo++;
      cnt++;
    }
    //
    if (pnt != point++) {
      continue;
    }
    for (int i = nGlo; i--;) {
      if (indGlo[i] == indx) {
        return derGlo[i];
      }
    }
    break;
  }
  return 0;
  //
}

//_____________________________________________________________________________________________
double MillePedeRecord::GetLocalDeriv(int pnt, int indx) const
{
  // get derivative over local variable indx at point pnt
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  //
  while (cnt < fSize) {
    //
    cnt++;
    double* derLoc = GetValue() + cnt;
    int* indLoc = GetIndex() + cnt;
    int nLoc = 0;
    while (!IsWeight(cnt)) {
      nLoc++;
      cnt++;
    }
    cnt++;
    while (!IsResidual(cnt) && cnt < fSize) {
      cnt++;
    }
    if (pnt != point++) {
      continue;
    }
    for (int i = nLoc; i--;) {
      if (indLoc[i] == indx) {
        return derLoc[i];
      }
    }
    break;
  }
  return 0;
  //
}

//_____________________________________________________________________________________________
double MillePedeRecord::GetResidual(int pnt) const
{
  // get residual at point pnt
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  //
  while (cnt < fSize) {
    //
    double resid = fValue[cnt++];
    while (!IsWeight(cnt)) {
      cnt++;
    }
    cnt++;
    while (!IsResidual(cnt) && cnt < fSize) {
      cnt++;
    }
    if (pnt != point++) {
      continue;
    }
    return resid;
  }
  return 0;
  //
}

//_____________________________________________________________________________________________
double MillePedeRecord::GetWeight(int pnt) const
{
  // get weight of point pnt
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  //
  while (cnt < fSize) {
    //
    cnt++;
    while (!IsWeight(cnt)) {
      cnt++;
    }
    if (point == pnt) {
      return GetValue(cnt);
    }
    cnt++;
    while (!IsResidual(cnt) && cnt < fSize) {
      cnt++;
    }
    point++;
  }
  return -1;
  //
}

//_____________________________________________________________________________________________
void MillePedeRecord::ExpandDtBuffer(int bfsize)
{
  // add extra space for derivatives data
  bfsize = TMath::Max(bfsize, GetDtBufferSize());
  int* tmpI = new int[bfsize];
  memcpy(tmpI, fIndex, fSize * sizeof(int));
  delete[] fIndex;
  fIndex = tmpI;
  //
  double* tmpD = new double[bfsize];
  memcpy(tmpD, fValue, fSize * sizeof(double));
  delete[] fValue;
  fValue = tmpD;
  //
  SetDtBufferSize(bfsize);
}

//_____________________________________________________________________________________________
void MillePedeRecord::ExpandGrBuffer(int bfsize)
{
  // add extra space for groupID data
  bfsize = TMath::Max(bfsize, GetGrBufferSize());
  unsigned short int* tmpI = new unsigned short int[bfsize];
  memcpy(tmpI, fGroupID, fNGroups * sizeof(unsigned short int));
  delete[] fGroupID;
  fGroupID = tmpI;
  for (int i = fNGroups; i < bfsize; i++) {
    fGroupID[i] = 0;
  }
  //
  SetGrBufferSize(bfsize);
}

//_____________________________________________________________________________________________
void MillePedeRecord::MarkGroup(int id)
{
  // mark the presence of the detector group
  id++; // groupID is stored as realID+1
  if (fNGroups > 0 && fGroupID[fNGroups - 1] == id) {
    return; // already there
  }
  if (fNGroups >= GetGrBufferSize()) {
    ExpandGrBuffer(2 * (fNGroups + 1));
  }
  fGroupID[fNGroups++] = id;
}
