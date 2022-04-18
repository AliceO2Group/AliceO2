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
#ifndef ALICEO2_EMCAL_ALTROHELPER_H_
#define ALICEO2_EMCAL_ALTROHELPER_H_

#include <Rtypes.h>
#include <iosfwd>
#include <exception>
#include <cstdint>
#include <vector>
#include <map>
#include "DataFormatsEMCAL/MCLabel.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALReconstruction/Bunch.h"

namespace o2
{
namespace emcal
{

/// \struct AltroBunch
/// \brief ALTRO bunch information obtained from digits
struct AltroBunch {
  int mStarttime;         ///< Start time of the bunch
  std::vector<int> mADCs; ///< ADCs belonging to the bunch
  std::multimap<int, std::vector<o2::emcal::MCLabel>> mEnergyLabels;
  std::vector<o2::emcal::MCLabel> mLabels;
};

/// \struct ChannelData
/// \brief Structure for mapping digits to Channels within a SRU
struct ChannelData {
  int mRow;                               ///< Row of the channel
  int mCol;                               ///< Column of the channel
  std::vector<o2::emcal::Digit*> mDigits; ///< Digits for the channel  within the current event
};

/// \struct SRUDigitContainer
/// \brief Structure for organizing digits within the SRU
struct SRUDigitContainer {
  int mSRUid;                           ///< DDL of the SRU
  std::map<int, ChannelData> mChannels; ///< Containers for channels within the SRU
};

/// \struct ChannelDigits
/// \brief Structure for mapping digits to Channels within a SRU
struct ChannelDigits {
  o2::emcal::ChannelType_t mChanType;
  std::vector<const o2::emcal::Digit*> mChannelDigits;
  std::vector<gsl::span<const o2::emcal::MCLabel>> mChannelLabels;
};

/// \struct DigitContainerPerSRU
/// \brief Structure for organizing digits within the SRU and preserves the digit type
struct DigitContainerPerSRU {
  int mSRUid;                                   ///< DDL of the SRU
  std::map<int, ChannelDigits> mChannelsDigits; ///< Containers for channels within the SRU
};

struct ChannelBunchData {
  o2::emcal::ChannelType_t mChanType;
  std::vector<o2::emcal::Bunch> mChannelsBunchesHG;
  std::vector<o2::emcal::Bunch> mChannelsBunchesLG;
  std::vector<std::vector<o2::emcal::MCLabel>> mChannelLabelsHG;
  std::vector<std::vector<o2::emcal::MCLabel>> mChannelLabelsLG;
};

/// \struct SRUBunchContainer
/// \brief Structure for organizing Bunches within the SRU
struct SRUBunchContainer {
  int mSRUid;                                    ///< DDL of the SRU
  std::map<int, ChannelBunchData> mChannelsData; ///< Containers for channels' bunches within the SRU
};

} // namespace emcal
} // namespace o2
#endif
