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

#include "FOCALCalib/PadBadChannelMap.h"

using namespace o2::focal;

PadBadChannelMap::PadBadChannelMap()
{
  init();
}

void PadBadChannelMap::init()
{
  // Mark all channels as good channels
  for (std::size_t index = 0; index < mChannelStatus.size(); index++) {
    mChannelStatus[index] = static_cast<uint8_t>(MaskType_t::GOOD_CHANNEL);
  }
}

void PadBadChannelMap::reset()
{
  init();
}

void PadBadChannelMap::setChannelStatus(std::size_t layer, std::size_t channel, MaskType_t masktype)
{
  mChannelStatus[getChannelIndex(layer, channel)] = static_cast<uint8_t>(masktype);
}

PadBadChannelMap::MaskType_t PadBadChannelMap::getChannelStatus(std::size_t layer, std::size_t channel) const
{
  return static_cast<MaskType_t>(mChannelStatus[getChannelIndex(layer, channel)]);
}

std::size_t PadBadChannelMap::getChannelIndex(std::size_t layer, std::size_t channel) const
{
  if (layer >= constants::PADS_NLAYERS || channel >= constants::PADLAYER_MODULE_NCHANNELS) {
    throw ChannelIndexException(layer, channel);
  }
  return channel + constants::PADLAYER_MODULE_NCHANNELS * layer;
}