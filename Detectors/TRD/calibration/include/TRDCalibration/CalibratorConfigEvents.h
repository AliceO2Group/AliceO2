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

#ifndef O2_TRD_CALIBRATORCONFIGEVENTS_H
#define O2_TRD_CALIBRATORCONFIGEVENTS_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DataFormatsTRD/CalVdriftExB.h"
#include "TRDCalibration/CalibrationParams.h"

#include "Rtypes.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"

#include <array>
#include <cstdlib>
#include <memory>

namespace o2::trd
{

class CalibratorConfigEvents final : public o2::calibration::TimeSlotCalibration<o2::trd::TrapConfigEventSlot>
{
  using Slot = o2::calibration::TimeSlot<o2::trd::TrapConfigEventSlot>;

 public:
  CalibratorConfigEvents() = default;
  ~CalibratorConfigEvents() final = default;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tStart, TFType tEnd) final;

  void createFile();

  void closeFile();

  const o2::trd::TrapConfigEvent& getCcdbObject() const { return mCCDBObject; }
  o2::ccdb::CcdbObjectInfo& getCcdbObjectInfo() { return mCCDBInfo; }

  void initProcessing();

  /// Initialize the fit values once with the previous valid ones if they are
  /// available.
  void retrievePrev(o2::framework::ProcessingContext& pc);

 private:
  bool mInitCompleted;
  const TRDCalibParams& mParams{TRDCalibParams::Instance()}; ///< reference to calibration parameters
  size_t mTimeBeforeComparison{mParams.configEventAccumulationTime};///< time of accumulating data and before comparison will be done
  bool mEnableOutput{false};                                 ///< enable output of configevent to a root file instead of the ccdb
  bool mSaveAllChanges=false;                                ///< Do we save all the changes to configs as they come in.
  std::unique_ptr<TFile> mOutFile{nullptr};                  ///< output file
  std::unique_ptr<TTree> mOutTree{nullptr};                  ///< output tree
  o2::ccdb::CcdbObjectInfo mCCDBInfo;                        ///< CCDB infos filled with CCDB description of accompanying CCDB calibration object
  o2::trd::TrapConfigEvent mCCDBObject;                      ///< CCDB calibration  object of TrapConfigEvent
  std::vector<o2::ccdb::CcdbObjectInfo> mInfoVector;         ///< vector of CCDB infos; each element is filled with CCDB description of accompanying CCDB calibration object
  std::vector<o2::trd::TrapConfigEventSlot> mObjectVector;       ///< vector of CCDB calibration objects waiting to be merged 
  
  ClassDefOverride(CalibratorConfigEvents, 1);
};

} // namespace o2::trd

#endif // O2_TRD_CALIBRATORVDEXB_H
