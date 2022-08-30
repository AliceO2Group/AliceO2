// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file MillePedeRecord.cxx

#include "MFTAlignment/MillePedeRecord.h"
#include <TMath.h>
#include "Framework/Logger.h"

using namespace o2::mft;

ClassImp(MillePedeRecord);

//_____________________________________________________________________________
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

//_____________________________________________________________________________
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
  fIndex = new Int_t[GetDtBufferSize()];
  memcpy(fIndex, src.fIndex, fSize * sizeof(Int_t));
  fValue = new Double_t[GetDtBufferSize()];
  memcpy(fValue, src.fValue, fSize * sizeof(Double_t));
  fGroupID = new UShort_t[GetGrBufferSize()];
  memcpy(fGroupID, src.fGroupID, GetGrBufferSize() * sizeof(UShort_t));
}

//_____________________________________________________________________________
MillePedeRecord& MillePedeRecord::operator=(const MillePedeRecord& rhs)
{
  if (this != &rhs) {
    Reset();
    for (int i = 0; i < rhs.GetSize(); i++) {
      Double_t val;
      Int_t ind;
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

//_____________________________________________________________________________
MillePedeRecord::~MillePedeRecord()
{
  delete[] fIndex;
  delete[] fValue;
  delete[] fGroupID;
}

//_____________________________________________________________________________
void MillePedeRecord::Reset()
{
  fSize = 0;
  for (int i = fNGroups; i--;) {
    fGroupID[i] = 0;
  }
  fNGroups = 0;
  fRunID = 0;
  fWeight = 1.;
}

//_____________________________________________________________________________
void MillePedeRecord::Print(const Option_t*) const
{
  if (!fSize) {
    LOG(info) << "No data";
    return;
  }
  int cnt = 0, point = 0;

  if (fNGroups) {
    printf("Groups: ");
  }
  for (int i = 0; i < fNGroups; i++) {
    printf("%4d |", GetGroupID(i));
  }
  printf("Run: %9d Weight: %+.2e\n", fRunID, fWeight);
  while (cnt < fSize) {
    Double_t resid = fValue[cnt++];
    Double_t* derLoc = GetValue() + cnt;
    int* indLoc = GetIndex() + cnt;
    int nLoc = 0;
    while (!IsWeight(cnt)) {
      nLoc++;
      cnt++;
    }
    Double_t weight = GetValue(cnt++);
    Double_t* derGlo = GetValue() + cnt;
    int* indGlo = GetIndex() + cnt;
    int nGlo = 0;
    while (!IsResidual(cnt) && cnt < fSize) {
      nGlo++;
      cnt++;
    }

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
  }
}

//_____________________________________________________________________________
Double_t MillePedeRecord::GetGloResWProd(Int_t indx) const
{
  if (!fSize) {
    LOG(info) << "No data";
    return 0;
  }
  int cnt = 0;
  double prodsum = 0.0;

  while (cnt < fSize) {
    Double_t resid = fValue[cnt++];
    while (!IsWeight(cnt)) {
      cnt++;
    }
    Double_t weight = GetValue(cnt++);
    Double_t* derGlo = GetValue() + cnt;
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
  }
  return prodsum;
}

//_____________________________________________________________________________
Double_t MillePedeRecord::GetGlobalDeriv(Int_t pnt, Int_t indx) const
{
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  while (cnt < fSize) {
    cnt++;
    while (!IsWeight(cnt)) {
      cnt++;
    }
    cnt++;
    Double_t* derGlo = GetValue() + cnt;
    int* indGlo = GetIndex() + cnt;
    int nGlo = 0;
    while (!IsResidual(cnt) && cnt < fSize) {
      nGlo++;
      cnt++;
    }
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
}

//_____________________________________________________________________________
Double_t MillePedeRecord::GetLocalDeriv(Int_t pnt, Int_t indx) const
{
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  while (cnt < fSize) {
    cnt++;
    Double_t* derLoc = GetValue() + cnt;
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
}

//_____________________________________________________________________________
Double_t MillePedeRecord::GetResidual(Int_t pnt) const
{
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  while (cnt < fSize) {
    Double_t resid = fValue[cnt++];
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
}

//_____________________________________________________________________________
Double_t MillePedeRecord::GetWeight(Int_t pnt) const
{
  if (!fSize) {
    LOG(error) << "No data";
    return 0;
  }
  int cnt = 0, point = 0;
  while (cnt < fSize) {
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
}

//_____________________________________________________________________________
void MillePedeRecord::ExpandDtBuffer(Int_t bfsize)
{
  bfsize = TMath::Max(bfsize, GetDtBufferSize());
  Int_t* tmpI = new Int_t[bfsize];
  memcpy(tmpI, fIndex, fSize * sizeof(Int_t));
  delete[] fIndex;
  fIndex = tmpI;

  Double_t* tmpD = new Double_t[bfsize];
  memcpy(tmpD, fValue, fSize * sizeof(Double_t));
  delete[] fValue;
  fValue = tmpD;

  SetDtBufferSize(bfsize);
}

//_____________________________________________________________________________
void MillePedeRecord::ExpandGrBuffer(Int_t bfsize)
{
  bfsize = TMath::Max(bfsize, GetGrBufferSize());
  UShort_t* tmpI = new UShort_t[bfsize];
  memcpy(tmpI, fGroupID, fNGroups * sizeof(UShort_t));
  delete[] fGroupID;
  fGroupID = tmpI;
  for (int i = fNGroups; i < bfsize; i++) {
    fGroupID[i] = 0;
  }

  SetGrBufferSize(bfsize);
}

//_____________________________________________________________________________
void MillePedeRecord::MarkGroup(Int_t id)
{
  id++; // groupID is stored as realID+1
  if (fNGroups > 0 && fGroupID[fNGroups - 1] == id) {
    return; // already there
  }
  if (fNGroups >= GetGrBufferSize()) {
    ExpandGrBuffer(2 * (fNGroups + 1));
  }
  fGroupID[fNGroups++] = id;
}
