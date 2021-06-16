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

/// \file PowerSupplyUnit.h
/// \brief MFT heat exchanger builder
/// \author P. Demongandin, Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_POWERSUPPLYUNIT_H_
#define ALICEO2_MFT_POWERSUPPLYUNIT_H_

#include "TGeoVolume.h"
#include "TGeoMatrix.h"

namespace o2
{
namespace mft
{

class PowerSupplyUnit
{

 public:
  PowerSupplyUnit();
  //PowerSupplyUnit(Double_t Rwater, Double_t DRPipe, Double_t PowerSupplyUnitThickness, Double_t CarbonThickness);

  ~PowerSupplyUnit() = default;

  TGeoVolumeAssembly* create();

 private:
  ClassDef(PowerSupplyUnit, 2);
};
} // namespace mft
} // namespace o2

#endif
