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

#ifndef O2_MCH_CALIBRATION_PEDESTAL_CHANNEL_H_
#define O2_MCH_CALIBRATION_PEDESTAL_CHANNEL_H_

#include "DataFormatsMCH/DsChannelId.h"

namespace o2::mch::calibration
{
/**
 * @class PedestalChannel
 * @brief Pedestal mean and sigma for one channel
 *
 * A PedestalChannel stores the mean and sigma of the pedestal of one MCH channel,
 * as well as the number of entries (digits) used to compute those values.
 */
struct PedestalChannel {
  int mEntries{0};         ///< number of entries used so far for the mean and variance
  double mPedestal{0};     ///< mean
  double mVariance{0};     ///< variance
  DsChannelId dsChannelId; ///< identifier of the channel

  /** return the RMS of the pedestal */
  double getRms() const;

  bool isValid() const; ///< true if the channel is associated to a detector pad

  std::string asString() const;
  friend std::ostream& operator<<(std::ostream&, const PedestalChannel&);
};
} // namespace o2::mch::calibration

#endif
