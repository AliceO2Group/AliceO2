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

#ifndef ALICEO2_EMCAL_DIGITSVECTORSTREAM_H_
#define ALICEO2_EMCAL_DIGITSVECTORSTREAM_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <deque>
#include <list>
#include <optional>
#include <gsl/span>
#include "TRandom3.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALSimulation/LabeledDigit.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "EMCALSimulation/SimParam.h"
#include "EMCALSimulation/DigitTimebin.h"

namespace o2
{
namespace emcal
{

/// \class DigitsVectorStream
/// \brief Container class for Digits, MC lebels, and trigger records
/// \ingroup EMCALsimulation
/// \author Hadi Hassan, ORNL
/// \author Markus Fasel, ORNL
/// \date 16/02/2022

class DigitsVectorStream
{

 public:
  /// Default constructor
  DigitsVectorStream() = default;

  /// Destructor
  ~DigitsVectorStream() = default;

  /// clear the container
  void clear();

  /// Initialize the streamer
  void init();

  /// Fill all the containers, digits, labels, and trigger records
  void fill(std::deque<o2::emcal::DigitTimebin>& digitlist, o2::InteractionRecord record);

  /// Getters for the finals data vectors, digits vector, labels vector, and trigger records vector
  const std::vector<o2::emcal::Digit>& getDigits() const { return mDigits; }
  const std::vector<o2::emcal::TriggerRecord>& getTriggerRecords() const { return mTriggerRecords; }
  const o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>& getMCLabels() const { return mLabels; }

  /// Flag whether to simulate noise to the digits
  void doSimulateNoiseDigits(bool doNoise = true) { mSimulateNoiseDigits = doNoise; }
  /// Add noise to this digit
  void addNoiseDigits(LabeledDigit& d1);

  /// Remove digits below the threshold
  void doRemoveDigitsBelowThreshold(bool doThreshold = true) { mRemoveDigitsBelowThreshold = doThreshold; }

 private:
  unsigned int mStartIndex = 0; ///< Start index for the digits in the trigger recod for every readout window

  std::vector<o2::emcal::Digit> mDigits;                         ///< Output vector for the digits
  o2::dataformats::MCTruthContainer<o2::emcal::MCLabel> mLabels; ///< Output vector for the MC labels
  std::vector<o2::emcal::TriggerRecord> mTriggerRecords;         ///< Output vector for the trigger records

  bool mSimulateNoiseDigits = true;        ///< simulate noise digits
  bool mRemoveDigitsBelowThreshold = true; ///< remove digits below threshold

  const SimParam* mSimParam = nullptr;  ///< SimParam object
  TRandom3* mRandomGenerator = nullptr; ///< random number generator

  ClassDefNV(DigitsVectorStream, 1);
};

} // namespace emcal

} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITSVECTORSTREAM_H_ */
