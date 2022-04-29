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

#ifndef TOF_DIAGNOSTIC_CALIBRATION_H_
#define TOF_DIAGNOSTIC_CALIBRATION_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTOF/Diagnostic.h"
#include "CCDB/CcdbObjectInfo.h"

namespace o2
{
namespace tof
{

class TOFDiagnosticCalibrator final : public o2::calibration::TimeSlotCalibration<o2::tof::Diagnostic, o2::tof::Diagnostic>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::tof::Diagnostic>;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;
  int mRunNumber = -1;

 public:
  TOFDiagnosticCalibrator() = default;
  ~TOFDiagnosticCalibrator() final = default;
  bool hasEnoughData(const Slot& slot) const final { return true; }
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  void setRunNumber(int run) { mRunNumber = run; }
  int getRunNumber() const { return mRunNumber; }

  const std::vector<Diagnostic>& getDiagnosticVector() const { return mDiagnosticVector; }
  const CcdbObjectInfoVector& getDiagnosticInfoVector() const { return mccdbInfoVector; }
  CcdbObjectInfoVector& getDiagnosticInfoVector() { return mccdbInfoVector; }

 private:
  CcdbObjectInfoVector mccdbInfoVector;
  std::vector<Diagnostic> mDiagnosticVector;

  ClassDefOverride(TOFDiagnosticCalibrator, 1);
};

} // end namespace tof
} // end namespace o2

#endif /* TOF_DIAGNOSTIC_CALIBRATION_H_ */
