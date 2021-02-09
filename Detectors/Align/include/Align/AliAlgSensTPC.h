// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSensTPC.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TPC sensor (chamber)

#ifndef ALIALGSENSTPC_H
#define ALIALGSENSTPC_H

#include "Align/AliAlgSens.h"

class TObjArray;
//class AliTrackPointArray;
//class AliESDtrack;
class AliAlgPoint;

namespace o2
{
namespace align
{

class AliAlgSensTPC : public AliAlgSens
{
 public:
  AliAlgSensTPC(const char* name = 0, Int_t vid = 0, Int_t iid = 0, Int_t isec = 0);
  virtual ~AliAlgSensTPC();
  //
  Int_t GetSector() const { return fSector; }
  void SetSector(UInt_t sc) { fSector = (UChar_t)sc; }
  //
  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //  virtual void   SetTrackingFrame();
  virtual void PrepareMatrixT2L();
  //
 protected:
  //
  UChar_t fSector; // sector ID

  ClassDef(AliAlgSensTPC, 1)
};
} // namespace align
} // namespace o2
#endif
