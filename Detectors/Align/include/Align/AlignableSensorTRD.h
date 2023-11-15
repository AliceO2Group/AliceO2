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

/// @file   AlignableSensorTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD sensor

#ifndef ALIGNABLESENSORTRD_H
#define ALIGNABLESENSORTRD_H

#include "Align/AlignableSensor.h"
class AlignmentPoint;
class TObjArray;

namespace o2
{
namespace align
{

class AlignableSensorTRD : public AlignableSensor
{
 public:
  AlignableSensorTRD() = default;
  AlignableSensorTRD(const char* name, int vid, int iid, int isec, Controller* ctr);
  ~AlignableSensorTRD() final = default;
  int getSector() const { return mSector; }
  void setSector(int sc) { mSector = (uint8_t)sc; }
  void dPosTraDParCalib(const AlignmentPoint* pnt, double* deriv, int calibID, const AlignableVolume* parent = nullptr) const final;
  void prepareMatrixT2L() final;

 protected:
  uint8_t mSector = 0; // sector ID

  ClassDef(AlignableSensorTRD, 1)
};
} // namespace align
} // namespace o2
#endif
