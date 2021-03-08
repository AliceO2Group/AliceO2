// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DsChannleGroup.h
/// \brief Implementation of a group of DualSampa channels
///
/// \author Andrea Ferrero, CEA-Saclay

#ifndef ALICEO2_MCH_DSCHANNELGROUP_H_
#define ALICEO2_MCH_DSCHANNELGROUP_H_

#include <vector>
#include "Rtypes.h"

namespace o2
{
namespace mch
{

/// Unique 32-bit identifier of a DulaSampa channel. The ID is generated from the following indexes:
/// - the unique ID of the corresponding solar board
/// - the index of the DulaSampa board within the Solar board, from 0 to 39
/// - the channel number, from 0 to 63
class DsChannelId
{
 public:
  DsChannelId() = default;
  DsChannelId(uint32_t channelId) : mChannelId(channelId) {}
  DsChannelId(uint16_t solarId, uint8_t dsId, uint8_t channel)
  {
    set(solarId, dsId, channel);
  }

  void set(uint16_t solarId, uint8_t dsId, uint8_t channel)
  {
    mChannelId = (static_cast<uint32_t>(solarId) << 16) +
                 (static_cast<uint32_t>(dsId) << 8) + channel;
  }

 private:
  uint32_t mChannelId{0};

  ClassDefNV(DsChannelId, 1); // class for MCH readout channel
};

/// A group of DualSampa channels, implemented as a vector of 32-bit channel identifiers
class DsChannelGroup
{
 public:
  DsChannelGroup() = default;

  const std::vector<DsChannelId>& getChannels() const { return mChannels; }
  std::vector<DsChannelId>& getChannels() { return mChannels; }

  void reset() { mChannels.clear(); }

 private:
  std::vector<DsChannelId> mChannels;

  ClassDefNV(DsChannelGroup, 1); // class for MCH bad channels list
};

} // end namespace mch
} // end namespace o2

#endif /* ALICEO2_MCH_DSCHANNELGROUP_H_ */
