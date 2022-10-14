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

/** @file HeartBeatPacket.h
 * C++ definition of a heart-beat packet
 * @author  Andrea Ferrero, CEE-Saclay
 */

#ifndef O2_MCH_BASE_HEARTBEATPACKET_H_
#define O2_MCH_BASE_HEARTBEATPACKET_H_

#include "Rtypes.h"

namespace o2
{
namespace mch
{

// \class HeartBeatPacket
/// \brief MCH heart-beat packet implementation
class HeartBeatPacket
{
 public:
  HeartBeatPacket() = default;

  HeartBeatPacket(int solarid, int dsid, int chip, uint32_t bc) : mSolarID(solarid), mChipID(dsid * 2 + (chip % 2)), mBunchCrossing(bc) {}
  ~HeartBeatPacket() = default;

  uint16_t getSolarID() const { return mSolarID; }
  uint8_t getDsID() const { return mChipID / 2; }
  uint8_t getChip() const { return mChipID % 2; }

  uint32_t getBunchCrossing() const { return mBunchCrossing; }

 private:
  uint16_t mSolarID{0};
  uint8_t mChipID{0};
  uint32_t mBunchCrossing{0};

  ClassDefNV(HeartBeatPacket, 1);
}; //class HeartBeatPacket

} //namespace mch
} //namespace o2
#endif // O2_MCH_BASE_HEARTBEATPACKET_H_
