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

#ifndef O2_DATAFORMATS_MCH_DS_CHANNEL_DET_ID_H_
#define O2_DATAFORMATS_MCH_DS_CHANNEL_DET_ID_H_

#include <cstdint>
#include <string>
#include "Rtypes.h"

namespace o2::mch
{
/** Unique 32-bit identifier of a DualSampa channel. The ID is generated from the following indexes:
 * - the detection element id
 * - the dual sampa id
 * - the channel number, from 0 to 63
 *
 * This class serves the same purposes as @ref DsChannelId, but using a different
 * "coordinate system" to reference the elements within the spectrometer.
 * @ref DsChannelId is more readout/online oriented,
 * while DsChannelDetId is more reconstruction/offline oriented.
 * Note that the actual underlying integer value of the two ids are *not*
 * and thus cannot be intermixed.
 */
class DsChannelDetId
{
 public:
  DsChannelDetId() = default;
  DsChannelDetId(uint32_t channelId) : mChannelDetId(channelId) {}
  DsChannelDetId(uint16_t detId, uint16_t dsId, uint8_t channel)
  {
    set(detId, dsId, channel);
  }

  static uint32_t make(uint16_t deId, uint16_t dsId, uint8_t channel)
  {
    uint32_t id = (static_cast<uint32_t>(deId) << 17) +
                  (static_cast<uint32_t>(dsId) << 6) + channel;
    return id;
  }

  void set(uint16_t deId, uint16_t dsId, uint8_t channel)
  {
    mChannelDetId = DsChannelDetId::make(deId, dsId, channel);
  }

  uint16_t getDeId() const { return static_cast<uint16_t>((mChannelDetId >> 17) & 0x7FF); }
  uint16_t getDsId() const { return static_cast<uint16_t>((mChannelDetId >> 6) & 0x7FF); }
  uint8_t getChannel() const { return static_cast<uint8_t>(mChannelDetId & 0x3F); }

  std::string asString() const;

  uint32_t value() const { return mChannelDetId; }

  bool isValid() const { return (mChannelDetId != 0); }

  bool operator==(const DsChannelDetId& chId) const { return mChannelDetId == chId.mChannelDetId; }
  bool operator!=(const DsChannelDetId& chId) const { return mChannelDetId != chId.mChannelDetId; }
  bool operator<(const DsChannelDetId& chId) const { return mChannelDetId < chId.mChannelDetId; }

 private:
  uint32_t mChannelDetId{0};

  ClassDefNV(DsChannelDetId, 1); // An identifier for a MCH channel (detector oriented)
};
} // namespace o2::mch
#endif
