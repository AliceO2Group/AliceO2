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

/// \file   MIDCalibration/ChannelCalibratorFinalizer.h
/// \brief  MID noise and dead channels calibrator finalizer
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   31 October 2022

#ifndef O2_MID_CHANNELCALIBRATORFINALIZER_H
#define O2_MID_CHANNELCALIBRATORFINALIZER_H

#include <string>
#include <vector>
#include <gsl/span>
#include "DataFormatsMID/ColumnData.h"

namespace o2
{
namespace mid
{

class ChannelCalibratorFinalizer
{
 public:
  /// Process the noisy and dead channels
  /// \param noise Vector of noisy channels
  /// \param dead Vector of dead channels
  void process(const gsl::span<const ColumnData> noise, const gsl::span<const ColumnData> dead);

  /// Returns the bad channels
  const std::vector<ColumnData>& getBadChannels() const { return mBadChannels; }

  /// Returns mask as string
  /// \return Masks as a string
  std::string getMasksAsString() { return mMasksString; }

  /// Sets reference masks
  /// \param refMasks Reference masks
  void setReferenceMasks(const std::vector<ColumnData>& refMasks) { mRefMasks = refMasks; }

 private:
  std::vector<ColumnData> mBadChannels; /// List of bad channels
  std::vector<ColumnData> mRefMasks;    /// Reference masks
  std::string mMasksString;             /// Masks as string
};
} // namespace mid
} // namespace o2

#endif
