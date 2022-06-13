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

#ifndef O2_DATAFORMATS_MCH_DS_CHANNEL_ID_H_
#define O2_DATAFORMATS_MCH_DS_CHANNEL_ID_H_

#include <cstdint>
#include <string>
#include "Rtypes.h"

namespace o2::mch
{
/// Unique 32-bit identifier of a DualSampa channel. The ID is generated from the following indexes:
/// - the unique ID of the corresponding solar board
/// - the index of the DualSampa board within the Solar board, from 0 to 39
/// - the channel number, from 0 to 63
class DsChannelId
{
 public:
  DsChannelId() = default;
  DsChannelId(uint32_t channelId) : mChannelId(channelId) {}
  DsChannelId(uint16_t solarId, uint8_t eLinkId, uint8_t channel)
  {
    set(solarId, eLinkId, channel);
  }

  static uint32_t make(uint16_t solarId, uint8_t eLinkId, uint8_t channel)
  {
    uint32_t id = (static_cast<uint32_t>(solarId) << 16) +
                  (static_cast<uint32_t>(eLinkId) << 8) + channel;
    return id;
  }

  void set(uint16_t solarId, uint8_t eLinkId, uint8_t channel)
  {
    mChannelId = DsChannelId::make(solarId, eLinkId, channel);
  }

  uint16_t getSolarId() const { return static_cast<uint16_t>((mChannelId >> 16) & 0xFFFF); }
  uint8_t getElinkId() const { return static_cast<uint8_t>((mChannelId >> 8) & 0xFF); }

  [[deprecated("use getElinkId instead which better reflects what it is and avoid confusion with dsId from DsDetId")]] uint8_t getDsId() const { return getElinkId(); }
  uint8_t getChannel() const { return static_cast<uint8_t>(mChannelId & 0xFF); }

  std::string asString() const;

 private:
  uint32_t mChannelId{0};

  ClassDefNV(DsChannelId, 1); // class for MCH readout channel
};
} // namespace o2::mch
#endif
