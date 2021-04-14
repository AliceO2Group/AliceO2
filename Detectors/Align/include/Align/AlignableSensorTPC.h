// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableSensorTPC.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TPC sensor (chamber)

#ifndef ALIGNABLESENSORTPC_H
#define ALIGNABLESENSORTPC_H

#include "Align/AlignableSensor.h"

class TObjArray;
//class AliTrackPointArray;
//class AliESDtrack;
class AlignmentPoint;

namespace o2
{
namespace align
{

class AlignableSensorTPC : public AlignableSensor
{
 public:
  AlignableSensorTPC(const char* name = 0, int vid = 0, int iid = 0, int isec = 0);
  virtual ~AlignableSensorTPC();
  //
  int GetSector() const { return fSector; }
  void SetSector(uint32_t sc) { fSector = (uint8_t)sc; }
  //
  virtual AlignmentPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //  virtual void   setTrackingFrame();
  virtual void prepareMatrixT2L();
  //
 protected:
  //
  uint8_t fSector; // sector ID

  ClassDef(AlignableSensorTPC, 1)
};
} // namespace align
} // namespace o2
#endif
