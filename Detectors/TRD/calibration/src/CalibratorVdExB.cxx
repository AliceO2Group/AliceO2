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

/// \file CalibratorVdExB.cxx
/// \brief TimeSlot-based calibration of vDrift and ExB
/// \author Ole Schmidt

#include "TRDCalibration/CalibratorVdExB.h"
#include <memory>

namespace o2::trd
{

using Slot = o2::calibration::TimeSlot<AngularResidHistos>;

void CalibratorVdExB::initOutput()
{
  // prepare output objects which will go to CCDB
}

void CalibratorVdExB::finalizeSlot(Slot& slot)
{
  // do actual calibration for the data provided in the given slot
  // TODO!
}

Slot& CalibratorVdExB::emplaceNewSlot(bool front, uint64_t tStart, uint64_t tEnd)
{
  auto& container = getSlots();
  auto& slot = front ? container.emplace_front(tStart, tEnd) : container.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<AngularResidHistos>());
  return slot;
}

} // namespace o2::trd
