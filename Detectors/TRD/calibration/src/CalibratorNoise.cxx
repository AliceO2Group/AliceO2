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

/// \file CalibratorNoise.cxx
/// \brief TRD pad calibration

#include "TRDCalibration/CalibratorNoise.h"

namespace o2::trd
{

using Slot = o2::calibration::TimeSlot<PadAdcInfo>;

Slot& CalibratorNoise::emplaceNewSlot(bool front, TFType tStart, TFType tEnd)
{
  auto& container = getSlots();
  auto& slot = front ? container.emplace_front(tStart, tEnd) : container.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<PadAdcInfo>());
  return slot;
}

} // namespace o2::trd
