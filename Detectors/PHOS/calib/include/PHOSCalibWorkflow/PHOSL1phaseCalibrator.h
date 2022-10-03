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

#ifndef O2_CALIBRATION_PHOSL1PHASE_CALIBRATOR_H
#define O2_CALIBRATION_PHOSL1PHASE_CALIBRATOR_H

/// @file   PHOSRunbyrunCalibDevice.h
/// @brief  Device to calculate PHOS time shift (L1phase)

#include "Framework/Task.h"
#include "Framework/ProcessingContext.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DataFormatsPHOS/Cell.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSL1phaseSlot
{
 public:
  PHOSL1phaseSlot();
  PHOSL1phaseSlot(const PHOSL1phaseSlot& other);

  ~PHOSL1phaseSlot() = default;

  void print() const;
  void fill(const gsl::span<const Cell>& cells, const gsl::span<const TriggerRecord>& trs);
  void fill(const gsl::span<const Cell>& /*cells*/){}; // not used
  void merge(const PHOSL1phaseSlot* prev);
  void clear();

  void addMeanRms(std::array<std::array<float, 4>, 14>& sumMean, std::array<std::array<float, 4>, 14>& sumRMS, std::array<float, 14>& sumNorm);

  void setRunStartTime(long tf) { mRunStartTime = tf; }

 private:
  static constexpr int mDDL = 14;               /// Number of PHOS DDLs
  long mRunStartTime = 0;                       /// start time of the run (sec)
  float mEmin = 1.5;                            /// Emin for time calculation, GeV
  float mTimeMin = -200.e-9;                    /// Time window for RMS calculation
  float mTimeMax = 200.e-9;                     /// Time window for RMS calculation
  std::array<std::array<float, 4>, mDDL> mRMS;  /// Collected RMS
  std::array<std::array<float, 4>, mDDL> mMean; /// Collected RMS
  std::array<float, mDDL> mNorm;                /// Normalization
  BadChannelsMap* mBadMap = nullptr;            /// Latest bad channels map owned by CCDB manager
  CalibParams* mCalibParams = nullptr;          /// Calibration parameters owned by CCDB manager
  ClassDefNV(PHOSL1phaseSlot, 1);
};

//==========================================================================================
class PHOSL1phaseCalibrator final : public o2::calibration::TimeSlotCalibration<o2::phos::Cell, o2::phos::PHOSL1phaseSlot>
{
  using Slot = o2::calibration::TimeSlot<o2::phos::PHOSL1phaseSlot>;

 public:
  PHOSL1phaseCalibrator();
  ~PHOSL1phaseCalibrator() final = default;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;
  bool process(TFType tf, const gsl::span<const Cell>& clu, const gsl::span<const TriggerRecord>& trs);
  void endOfStream();

  int getCalibration() { return mL1phase; }

 private:
  static constexpr int mDDL = 14;               /// Number of PHOS DDLs
  long mRunStartTime = 0;                       /// start time of the run (sec)
  std::array<std::array<float, 4>, mDDL> mRMS;  /// Collected RMS
  std::array<std::array<float, 4>, mDDL> mMean; /// Collected RMS
  std::array<float, mDDL> mNorm;                /// Normalization
  int mL1phase = 0;                             /// Final calibration

  ClassDefOverride(PHOSL1phaseCalibrator, 1);
};

} // namespace phos
} // namespace o2

#endif
