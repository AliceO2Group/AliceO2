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

/// \file MillePedeRecord.h
/// \author ruben.shahoyan@cern.ch
/// \brief Class to store the data of single track processing
///
/// The records for all processed tracks are stored in the temporary tree in orgder to be
/// reused for multiple iterations of MillePede

#ifndef ALICEO2_MFT_MILLEPEDERECORD_H
#define ALICEO2_MFT_MILLEPEDERECORD_H

#include <TObject.h>

namespace o2
{
namespace mft
{
/// \brief Store residuals and local/global deriavtives from a single track processing
///
/// Format: for each measured point the data is stored consecutively
///
/// INDEX                                VALUE
/// -1                                   residual
/// Local_param_id                       dResidual/dLocal_param
/// ...                                  ...
/// -2                                   weight of the measurement
/// Global_param_od                      dResidual/dGlobal_param
/// ...                                  ...
class MillePedeRecord : public TObject
{
 public:
  /// \brief default c-tor
  MillePedeRecord();

  /// \brief copy c-tor
  MillePedeRecord(const MillePedeRecord& src);

  /// \brief assignment op-r
  MillePedeRecord& operator=(const MillePedeRecord& rhs);

  /// \brief destuctor
  ~MillePedeRecord() override;

  /// \brief reset all
  void Reset();

  /// \brief print itself
  void Print(const Option_t* opt = "") const override;

  Int_t GetSize() const { return fSize; }
  Int_t* GetIndex() const { return fIndex; }
  Int_t GetIndex(int i) const { return fIndex[i]; }

  void GetIndexValue(Int_t i, Int_t& ind, Double_t& val) const
  {
    ind = fIndex[i];
    val = fValue[i];
  }

  /// \brief add new pair of index/value
  void AddIndexValue(Int_t ind, Double_t val);

  void AddResidual(Double_t val) { AddIndexValue(-1, val); }
  void AddWeight(Double_t val) { AddIndexValue(-2, val); }
  void SetWeight(Double_t w = 1) { fWeight = w; }
  Bool_t IsResidual(Int_t i) const { return fIndex[i] == -1; }
  Bool_t IsWeight(Int_t i) const { return fIndex[i] == -2; }

  Double_t* GetValue() const { return fValue; }
  Double_t GetValue(Int_t i) const { return fValue[i]; }
  Double_t GetWeight() const { return fWeight; }

  /// \brief mark the presence of the detector group
  void MarkGroup(Int_t id);
  Int_t GetNGroups() const { return fNGroups; }
  Int_t GetGroupID(Int_t i) const { return fGroupID[i] - 1; }

  /// \brief check if group is defined
  Bool_t IsGroupPresent(Int_t id) const;

  UInt_t GetRunID() const { return fRunID; }
  void SetRunID(UInt_t run) { fRunID = run; }

  /// \brief get derivative over global variable indx at point pnt
  Double_t GetGlobalDeriv(Int_t pnt, Int_t indx) const;

  /// \brief get derivative over local variable indx at point pnt
  Double_t GetLocalDeriv(Int_t pnt, Int_t indx) const;

  /// \brief get residual at point pnt
  Double_t GetResidual(Int_t pnt) const;

  /// \brief get sum of derivative over global variable indx * res. at point * weight
  Double_t GetGloResWProd(Int_t indx) const;

  /// \brief get weight of point pnt
  Double_t GetWeight(Int_t indx) const;

 protected:
  Int_t GetDtBufferSize() const { return GetUniqueID() & 0x0000ffff; }
  Int_t GetGrBufferSize() const { return GetUniqueID() >> 16; }
  void SetDtBufferSize(Int_t sz) { SetUniqueID((GetGrBufferSize() << 16) + sz); }
  void SetGrBufferSize(Int_t sz) { SetUniqueID(GetDtBufferSize() + (sz << 16)); }

  /// \brief add extra space for derivatives data
  void ExpandDtBuffer(Int_t bfsize);

  /// \brief add extra space for groupID data
  void ExpandGrBuffer(Int_t bfsize);

 protected:
  Int_t fSize;        ///< size of the record
  Int_t fNGroups;     ///< number of groups (e.g. detectors) contributing
  UInt_t fRunID;      ///< run ID
  UShort_t* fGroupID; ///< [fNGroups] groups id's+1 (in increasing order)
  Int_t* fIndex;      ///< [fSize] index of variables
  Double32_t* fValue; ///< [fSize] array of values: derivs,residuals
  Double32_t fWeight; ///< global weight for the record

  ClassDef(MillePedeRecord, 3);
};

//_____________________________________________________________________________
inline void MillePedeRecord::AddIndexValue(Int_t ind, Double_t val)
{
  if (fSize >= GetDtBufferSize()) {
    ExpandDtBuffer(2 * (fSize + 1));
  }
  fIndex[fSize] = ind;
  fValue[fSize++] = val;
}

//_____________________________________________________________________________
inline Bool_t MillePedeRecord::IsGroupPresent(Int_t id) const
{
  id++;
  for (int i = fNGroups; i--;) {
    if (fGroupID[i] == id) {
      return kTRUE;
    }
  }
  return kFALSE;
}

} // namespace mft
} // namespace o2

#endif
