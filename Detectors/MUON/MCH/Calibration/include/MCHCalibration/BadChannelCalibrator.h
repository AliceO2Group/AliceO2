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

/// \file BadChannelCalibrator.h
/// \brief MCH calibrator to produce a bad channel map, using pedestal data
///
/// \author Andrea Ferrero, CEA-Saclay

#ifndef O2_MCH_CALIBRATION_BADCHANNEL_CALIBRATOR_H_
#define O2_MCH_CALIBRATION_BADCHANNEL_CALIBRATOR_H_

#include "DataFormatsMCH/DsChannelId.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "MCHCalibration/PedestalData.h"
#include "MCHCalibration/PedestalDigit.h"
#include <array>

namespace o2::mch::calibration
{

/**
 * @class BadChannelCalibrator
 * @brief Compute bad channel map from pedestal data
 *
 * Calibrator that checks the computed mean and RMS of the MCH pedestals
 * and compares the values with (configurable) thresholds.
 * The channels whose values exceed one of the thresholds are
 * considered bad/noisy and they are stored into a
 * "bad channels" list that is sent to the CDDB populator(s).
 */
class BadChannelCalibrator final : public o2::calibration::TimeSlotCalibration<o2::mch::calibration::PedestalData>
{
  using TFType = o2::calibration::TFType;
  using Slot = o2::calibration::TimeSlot<o2::mch::calibration::PedestalData>;
  using BadChannelsVector = std::vector<o2::mch::DsChannelId>;
  using PedestalsVector = std::vector<PedestalChannel>;

 public:
  BadChannelCalibrator() = default;

  ~BadChannelCalibrator() final = default;

  /** Decides whether the Slot has enough data to compute the calibration.
   * Decision depends both on the data itself and on the BadChannelCalibratorParam
   * parameters.
   */
  bool hasEnoughData(const Slot& slot) const final;

  void initOutput() final;

  void finalizeSlot(Slot& slot) final;

  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final;

  bool readyToSend(std::string& reason) const;

  const BadChannelsVector& getBadChannelsVector() const { return mBadChannelsVector; }
  const PedestalsVector& getPedestalsVector() const { return mPedestalsVector; }

 private:
  TFType mTFStart;

  BadChannelsVector mBadChannelsVector; ///< vector containing the unique IDs of the bad/noisy channels
  PedestalsVector mPedestalsVector;     ///< vector containing the source pedestal information used for bad channel decision

  ClassDefOverride(BadChannelCalibrator, 1);
};

} // namespace o2::mch::calibration
#endif
