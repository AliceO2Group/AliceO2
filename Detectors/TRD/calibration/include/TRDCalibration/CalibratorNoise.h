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

/// \file CalibratorNoise.h
/// \brief TRD pad calibration

#ifndef O2_TRD_CALIBRATORNOISE_H
#define O2_TRD_CALIBRATORNOISE_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "TRDBase/PadCalibrationsAliases.h"
#include "DataFormatsTRD/NoiseCalibration.h"
#include "CCDB/CcdbObjectInfo.h"
#include "Rtypes.h"

namespace o2
{
namespace trd
{

class CalibratorNoise final : public o2::calibration::TimeSlotCalibration<PadAdcInfo, PadAdcInfo>
{
  using Slot = o2::calibration::TimeSlot<PadAdcInfo>;

 public:
  CalibratorNoise() = default;
  ~CalibratorNoise() final = default;

  // TODO implement these methods
  bool hasEnoughData(const Slot& slot) const final { return false; }
  void initOutput() final {}
  void finalizeSlot(Slot& slot) final {}
  Slot& emplaceNewSlot(bool front, TFType tStart, TFType tEnd) final;

  const std::vector<PadNoise>& getCcdbObjectVector() const { return mObjectVector; }
  std::vector<o2::ccdb::CcdbObjectInfo>& getCcdbObjectInfoVector() { return mInfoVector; }

 private:
  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector; ///< vector of CCDB infos; each element is filled with CCDB description of accompanying CCDB calibration object
  std::vector<PadNoise> mObjectVector;               ///< vector of CCDB calibration objects; the extracted pad noise value for given slot
  ClassDefOverride(CalibratorNoise, 1);
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CALIBRATORNOISE_H
