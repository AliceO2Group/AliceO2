// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSensTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD sensor

#ifndef ALIALGSENSTRD_H
#define ALIALGSENSTRD_H

#include "AliAlgSens.h"
class AliTrackPointArray;
class AliESDtrack;
class AliAlgPoint;
class TObjArray;

namespace o2
{
namespace align
{

class AliAlgSensTRD : public AliAlgSens
{
 public:
  AliAlgSensTRD(const char* name = 0, Int_t vid = 0, Int_t iid = 0, Int_t isec = 0);
  virtual ~AliAlgSensTRD();
  //
  Int_t GetSector() const { return fSector; }
  void SetSector(UInt_t sc) { fSector = (UChar_t)sc; }
  //
  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //
  virtual void DPosTraDParCalib(const AliAlgPoint* pnt, double* deriv, int calibID, const AliAlgVol* parent = 0) const;
  //
  //  virtual void   SetTrackingFrame();
  virtual void PrepareMatrixT2L();
  //
 protected:
  //
  UChar_t fSector; // sector ID

  ClassDef(AliAlgSensTRD, 1)
};
} // namespace align
} // namespace o2
#endif
