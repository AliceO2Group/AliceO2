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

///
/// \file Digitizer.h
/// \brief Definition of the ALICE3 TOF digitizer
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-03-17
///

#ifndef ALICEO2_IOTOF_DIGITIZER_H
#define ALICEO2_IOTOF_DIGITIZER_H

#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsIOTOF/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "IOTOFSimulation/Segmentation.h"

namespace o2::iotof
{

/// \class Digitizer
/// \brief Digitizer for the ALICE3 Inner/Outer TOF detector
///
/// Converts MC hits into detector digits by:
/// - Applying time smearing according to detector resolution
/// - Converting energy loss to charge
/// - Applying charge threshold
/// - Managing readout frames (ROF)
class Digitizer
{
 public:
  void setDigits(std::vector<o2::iotof::Digit>* dig) { mDigits = dig; }
  void setMCLabels(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mclb) { mMCLabels = mclb; }
  void setROFRecords(std::vector<o2::itsmft::ROFRecord>* rec) { mROFRecords = rec; }

  /// Initialize the digitizer
  void init();

  /// Steer conversion of hits to digits
  void process(const std::vector<o2::itsmft::Hit>* hits, int evID, int srcID);

  /// Set the event time
  void setEventTime(const o2::InteractionTimeRecord& irt) { mEventTime = irt; }

  /// Set continuous readout mode
  void setContinuous(bool v) { mContinuous = v; }
  bool isContinuous() const { return mContinuous; }

  /// Flush the output container
  void fillOutputContainer();

  // Provide the common iotof::GeometryTGeo to access matrices and segmentation
  void setGeometry(const o2::iotof::GeometryTGeo* gm) { mGeometry = gm; }

  // Setters for digitization parameters
  void setChargeThreshold(float thr) { mChargeThreshold = thr; }
  void setTimeResolution(float res) { mTimeResolution = res; }
  void setEfficiency(float eff) { mEfficiency = eff; }
  void setEnergyToCharge(float e2c) { mEnergyToCharge = e2c; }

  // Getters
  float getChargeThreshold() const { return mChargeThreshold; }
  float getTimeResolution() const { return mTimeResolution; }
  float getEfficiency() const { return mEfficiency; }

 private:
  /// Process a single hit
  void processHit(const o2::itsmft::Hit& hit, int evID, int srcID);

  /// Apply time smearing to simulate detector resolution
  double smearTime(double time) const;

  /// Convert energy loss to charge
  int energyToCharge(float energyLoss) const;

  /// Check if the hit passes efficiency cut
  bool isEfficient() const;

  static constexpr float sec2ns = 1e9f; ///< seconds to nanoseconds conversion

  const o2::iotof::GeometryTGeo* mGeometry = nullptr; ///< IOTOF geometry

  std::vector<o2::iotof::Digit>* mDigits = nullptr;                        //! output digits
  std::vector<o2::itsmft::ROFRecord>* mROFRecords = nullptr;               //! output ROF records
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCLabels = nullptr; //! output labels

  o2::InteractionTimeRecord mEventTime; ///< global event time and interaction record
  bool mContinuous = true;              ///< continuous readout mode

  // Digitization parameters
  float mChargeThreshold = 100.f;  ///< charge threshold for digit creation (electrons)
  float mTimeResolution = 0.020f;  ///< time resolution sigma in ns (20 ps default)
  float mEfficiency = 0.98f;       ///< detection efficiency
  float mEnergyToCharge = 3.6e-9f; ///< energy loss to electrons conversion (3.6 eV per e-h pair in Si)

  static o2::iotof::Segmentation* sSegmentation; ///< IOTOF segmentation instance (singleton)
};
} // namespace o2::iotof

#endif