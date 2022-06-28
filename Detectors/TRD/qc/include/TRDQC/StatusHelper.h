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

/// \file StatusHelper.h
/// \brief Helper with 'permanent' status of TRD half-chambers to support QC plots
/// \author Ole Schmidt

#ifndef O2_TRD_STATUSHELPER_H
#define O2_TRD_STATUSHELPER_H

#include "DataFormatsTRD/Constants.h"

#include "Rtypes.h"

#include <bitset>

namespace o2
{

namespace trd
{

class HalfChamberStatusQC
{

 public:
  HalfChamberStatusQC() = default;
  void maskChamber(int sec, int stack, int ly);
  void maskHalfChamberA(int sec, int stack, int ly);
  void maskHalfChamberB(int sec, int stack, int ly);
  bool isMasked(int hcId) const { return mStatus.test(hcId); }
  void print();

 private:
  std::bitset<constants::MAXHALFCHAMBER> mStatus{};

  ClassDefNV(HalfChamberStatusQC, 1);
};

} // namespace trd

} // namespace o2

#endif // O2_TRD_STATUSHELPER_H
