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
#ifndef ALICEO2_EMCAL_FASTORTIMESERIES_H
#define ALICEO2_EMCAL_FASTORTIMESERIES_H

#include <vector>
#include <gsl/span>
#include "Rtypes.h"

namespace o2::emcal
{

/// \class FastORTimeSeries
/// \brief Container for FastOR time series
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \ingroup EMCALreconstruction
/// \since April 19, 2024
///
/// Time series are encoded in bunches in the raw data, which are usually time-reversed.
/// The FastORTimeSeries handles the time series of all bunches in the readout window,
/// in proper future-direction time order, correcting the time-reversal from the Fake-ALTRO.
/// Consequently the ADC samples are expected in time-reversed format. The function
/// calculateL1TimeSum calculates the timesum of the timeseries as 4-integral with respect to
/// a given L0 time, which is expected at the end of the time integration range.
class FastORTimeSeries
{
 public:
  /// @brief Dummy constructor
  FastORTimeSeries() = default;

  /// \brief Construcor
  /// \param maxsamples Maximum number of time samples
  /// \param timesamples Time-reversed raw ADC samples
  /// \param starttime Start time
  FastORTimeSeries(int maxsamples, const gsl::span<const uint16_t> timesamples, uint8_t starttime)
  {
    setSize(maxsamples);
    fillReversed(timesamples, starttime);
  }

  /// \brief Destructor
  ~FastORTimeSeries() = default;

  void setTimeSamples(const gsl::span<const uint16_t> timesamples, uint8_t starttime) { fillReversed(timesamples, starttime); }

  /// \brief Calculate L0 timesum (4-integral of the ADC series) with respect to a given L0 time
  /// \param l0time L0 time (end of the time series)
  /// \return Timesum of the time series
  uint16_t calculateL1TimeSum(uint8_t l0time) const;

  /// \brief Access raw ADC values (in forward time order)
  /// \return ADC values of the time series in forward time order
  const gsl::span<const uint16_t> getADCs() const { return mTimeSamples; }

  /// \brief Clear ADC samples in the time series
  void clear();

 private:
  /// \brief Set the container size for the ADC samples
  /// \param maxsamples Max. amount of samples to be handled
  void setSize(int maxsamples);

  /// \brief Fill the internal time samples in proper time order
  /// \param timesamples Time-reversed time samples
  /// \param starttime Start time
  void fillReversed(const gsl::span<const uint16_t> timesamples, uint8_t starttime);

  std::vector<uint16_t> mTimeSamples; ///< Raw ADC time samples (in forward time order)

  ClassDef(FastORTimeSeries, 1);
};

} // namespace o2::emcal

#endif