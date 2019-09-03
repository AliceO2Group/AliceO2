// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digit.h
/// \brief Definition of the ZDC Digit class

#ifndef ALICEO2_ZDC_DIGIT_H_
#define ALICEO2_ZDC_DIGIT_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "ZDCBase/Constants.h"
#include <array>

namespace o2
{
namespace zdc
{

struct ChannelBCData {
  std::array<uint16_t, NTimeBinsPerBC> data = {0};
};

struct ChannelData {
  std::array<ChannelBCData, NBCReadOut> data = {};
};

class Digit
{
 public:
  Digit() = default;

  const ChannelBCData& getChannel(int ch, int bc) const { return mChannels[ch].data[bc]; }
  ChannelBCData& getChannel(int ch, int bc) { return mChannels[ch].data[bc]; }

  const ChannelBCData& getZNA(int twr, int bc) const { return mChannels[toChannel(ZNA, twr)].data[bc]; }
  ChannelBCData& getZNA(int twr, int bc) { return mChannels[toChannel(ZNA, twr)].data[bc]; }

  const ChannelBCData& getZPA(int twr, int bc) const { return mChannels[toChannel(ZPA, twr)].data[bc]; }
  ChannelBCData& getZPA(int twr, int bc) { return mChannels[toChannel(ZPA, twr)].data[bc]; }

  const ChannelBCData& getZNC(int twr, int bc) const { return mChannels[toChannel(ZNC, twr)].data[bc]; }
  ChannelBCData& getZNC(int twr, int bc) { return mChannels[toChannel(ZNC, twr)].data[bc]; }

  const ChannelBCData& getZPC(int twr, int bc) const { return mChannels[toChannel(ZPC, twr)].data[bc]; }
  ChannelBCData& getZPC(int twr, int bc) { return mChannels[toChannel(ZPC, twr)].data[bc]; }

  const ChannelBCData& getZEM(int twr, int bc) const { return mChannels[toChannel(ZEM, twr)].data[bc]; }
  ChannelBCData& getZEM(int twr, int bc) { return mChannels[toChannel(ZEM, twr)].data[bc]; }

  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& getInteractionRecord() { return mIntRecord; }

  void print() const;

 private:
  std::array<ChannelData, NChannels> mChannels = {};
  o2::InteractionRecord mIntRecord;

  ClassDefNV(Digit, 1);
};

} // namespace zdc
} // namespace o2

#endif
