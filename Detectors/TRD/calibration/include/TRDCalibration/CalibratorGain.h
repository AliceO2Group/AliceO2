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

/// \file CalibratorVdExB.h
/// \brief TimeSlot-based calibration of vDrift and ExB
/// \author Ole Schmidt

#ifndef O2_TRD_CALIBRATORGAIN_H
#define O2_TRD_CALIBRATORGAIN_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/CalGain.h"
#include "DataFormatsTRD/GainCalibration.h"
#include "CCDB/CcdbObjectInfo.h"

#include "Rtypes.h"
#include "TProfile.h"

#include <array>
#include <cstdlib>

namespace o2
{
namespace trd
{

class CalibratorGain final : public o2::calibration::TimeSlotCalibration<o2::trd::GainCalibration, o2::trd::GainCalibration>
{
  using Slot = o2::calibration::TimeSlot<o2::trd::GainCalibration>;

 public:
  CalibratorGain(bool enableOut = false) : mEnableOutput(enableOut) {}
  ~CalibratorGain() final = default;

  bool hasEnoughData(const Slot& slot) const final { return true; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tStart, TFType tEnd) final;

  // TODO
  const std::vector<o2::trd::CalGain>& getCcdbObjectVector() const { return mObjectVector; }
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbObjectInfoVector() { return mInfoVector; }

  void initProcessing();

 private:
  bool mInitDone{false};                             ///< flag to avoid creating the TProfiles multiple times
  bool mEnableOutput;                                ///< enable output of calibration fits and tprofiles in a root file instead of the ccdb
  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector; ///< vector of CCDB infos; each element is filled with CCDB description of accompanying CCDB calibration object
  std::vector<o2::trd::CalGain> mObjectVector;       ///< vector of CCDB calibration objects; the extracted ... TODO

  ClassDefOverride(CalibratorGain, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CALIBRATORGAIN_H
