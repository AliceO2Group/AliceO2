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

/// @file   AlignableSensorITS.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  ITS sensor

#ifndef ALIGNABLESENSORITS_H
#define ALIGNABLESENSORITS_H

#include "Align/AlignableSensor.h"

class TObjArray;
class AliTrackPointArray;
class AliESDtrack;
class AlignmentPoint;

namespace o2
{
namespace align
{

class AlignableSensorITS : public AlignableSensor
{
 public:
  AlignableSensorITS() = default;
  AlignableSensorITS(const char* name, int vid, int iid, Controller* ctr);
  ~AlignableSensorITS() final = default;
  void prepareMatrixT2L() final;
  //
 protected:
  //
  ClassDefOverride(AlignableSensorITS, 1)
};
} // namespace align
} // namespace o2
#endif
