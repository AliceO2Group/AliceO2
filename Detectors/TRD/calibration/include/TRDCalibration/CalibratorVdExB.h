// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibratorVdExB.h
/// \brief TimeSlot-based calibration of vDrift and ExB
/// \author Ole Schmidt

#ifndef O2_TRD_CALIBRATORVDEXB_H
#define O2_TRD_CALIBRATORVDEXB_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/AngularResidHistos.h"

#include "Rtypes.h"

#include <array>
#include <cstdlib>

namespace o2
{
namespace trd
{

class CalibratorVdExB final : public o2::calibration::TimeSlotCalibration<o2::trd::AngularResidHistos, o2::trd::AngularResidHistos>
{
  using Slot = o2::calibration::TimeSlot<o2::trd::AngularResidHistos>;

 public:
  CalibratorVdExB(size_t nMin = 40'000) : mMinEntries(nMin) {}
  ~CalibratorVdExB() final = default;

  bool hasEnoughData(const Slot& slot) const final { return slot.getContainer()->getNEntries() >= mMinEntries; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, uint64_t tStart, uint64_t tEnd) final;

  // TODO add calibration objects (vDrift and ExB values for each chamber) to this class and implement calibration in finalizeSlot()

 private:
  size_t mMinEntries; ///< minimum total number of angular deviations (on average ~3 entries per bin for each TRD chamber)
  ClassDefOverride(CalibratorVdExB, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CALIBRATORVDEXB_H
