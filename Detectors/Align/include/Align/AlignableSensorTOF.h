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

/// @file   AlignableSensorTOF.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TOF sensor

#ifndef ALIGNABLESENSORTOF_H
#define ALIGNABLESENSORTOF_H

#include "Align/AlignableSensor.h"

//class AliTrackPointArray;
//class AliESDtrack;
class AlignmentPoint;
class TObjArray;

namespace o2
{
namespace align
{

class AlignableSensorTOF : public AlignableSensor
{
 public:
  AlignableSensorTOF(const char* name = 0, int vid = 0, int iid = 0, int isec = 0);
  ~AlignableSensorTOF() = final;
  //
  void prepareMatrixT2L() final;
  //
  int GetSector() const { return fSector; }
  void SetSector(uint32_t sc) { fSector = (uint8_t)sc; }
  //
 protected:
  //
  uint8_t fSector; // sector ID
  //
  ClassDef(AlignableSensorTOF, 1)
};
} // namespace align
} // namespace o2
#endif
