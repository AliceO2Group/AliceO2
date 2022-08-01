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
#ifndef ALICEO2_EMCAL_BUNCH_H
#define ALICEO2_EMCAL_BUNCH_H

#include <cstdint>
#include <vector>
#include <gsl/span>
#include "Rtypes.h"

namespace o2
{

namespace emcal
{
/// \class Bunch
/// \brief ALTRO bunch information
/// \ingroup EMCALreconstruction
///
/// The bunch contains the ADC values of a given
/// data bunch for a channel in the ALTRO stream.
/// The ADC values are stored in reversed order in
/// time both in the ALTRO stream and in the bunch
/// object.
///
/// For iteration one should assume that the end time
/// is 0, however it can also be larger than 0. In this
/// case the first value has to be mapped to the end timebin
/// and the last value to the start timebin, still iterating
/// only over the number of samples.
class Bunch
{
 public:
  /// \brief Constructor
  Bunch() = default;

  /// \brief Initialize the bunch with start time and bunch length
  /// \param length Length of the bunch
  /// \param start Start time of the bunch
  Bunch(uint8_t length, uint8_t start) : mBunchLength(length), mStartTime(start), mADC() {}

  /// \brief
  ~Bunch() = default;

  /// \brief Add ADC value to the bunch
  /// \param adc Next ADC value
  ///
  /// ADC values are stored in reversed order. The next ADC value
  /// has to be earlier in time compared to the previous one.
  void addADC(uint16_t adc) { mADC.emplace_back(adc); }

  /// \brief Initialize the ADC values in the bunch from a range
  /// \param range Range of ADC values
  ///
  /// The ADC values are stored in reversed order in time. Therefore
  /// the last entry is the one earliest in time.
  void initFromRange(gsl::span<uint16_t> range);

  /// \brief Get range of ADC values in the bunch
  /// \return ADC values in the bunch
  ///
  /// The ADC values are stored in reversed order in time. Therefore
  /// the last entry is the one earliest in time.
  const std::vector<uint16_t>& getADC() const { return mADC; }

  /// \brief Get the length of the bunch (number of time bins)
  /// \return Length of the bunch
  uint8_t getBunchLength() const { return mBunchLength; }

  /// \brief Get the start time bin
  /// \return Start timebin
  ///
  /// The start timebin is the higher of the two,
  /// the samples are in reversed order.
  uint8_t getStartTime() const { return mStartTime; }

  /// \brief Get the end time bin
  /// \return End timebin
  ///
  /// The end timebin is the lower of the two,
  /// the samples are in reversed order.
  uint8_t getEndTime() const { return mStartTime - mBunchLength + 1; }

  /// \brief Set the length of the ALTRO bunch
  /// \param length Bunch length
  void setBunchLength(uint8_t length) { mBunchLength = length; }

  /// \brief Set the start time bin
  /// \param start timebin
  ///
  /// The start timebin is the higher of the two,
  /// the samples are in reversed order.
  void setStartTime(uint8_t start) { mStartTime = start; }

 private:
  uint8_t mBunchLength = 0;   ///< Number of ADC samples in buffer
  uint8_t mStartTime = 0;     ///< Start timebin (larger time bin, samples are in reversed order)
  std::vector<uint16_t> mADC; ///< ADC samples in bunch

  ClassDefNV(Bunch, 1);
};

} // namespace emcal

} // namespace o2

#endif
