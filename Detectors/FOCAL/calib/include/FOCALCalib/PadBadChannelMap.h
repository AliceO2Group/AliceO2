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
#ifndef ALICEO2_FOCAL_PADBADCHANNELMAP_H
#define ALICEO2_FOCAL_PADBADCHANNELMAP_H

#include <array>
#include <bitset>
#include <exception>
#include <string>
#include "Rtypes.h"

#include "DataFormatsFOCAL/Constants.h"

namespace o2::focal
{

class PadBadChannelMap
{
 public:
  enum MaskType_t {
    GOOD_CHANNEL = 0,
    WARM_CHANNEL = 1,
    BAD_CHANNEL = 2,
    DEAD_CHANNEL = 3
  };

  class ChannelIndexException : public std::exception
  {
   public:
    ChannelIndexException(std::size_t layer, std::size_t channel) : mLayer(layer), mChannel(channel)
    {
      mMessage = "Access to invalid channel: Layer " + std::to_string(mLayer) + ", channel " + std::to_string(mChannel);
    }
    ~ChannelIndexException() noexcept final = default;

    const char* what() const noexcept final { return mMessage.data(); }

    std::size_t getLayer() const noexcept { return mLayer; }
    std::size_t getChannel() const noexcept { return mChannel; }

   private:
    std::size_t mLayer;
    std::size_t mChannel;
    std::string mMessage;
  };

  PadBadChannelMap();
  ~PadBadChannelMap() = default;

  void reset();

  void setChannelStatus(std::size_t layer, std::size_t channel, MaskType_t channeltype);
  void setGoodChannel(std::size_t layer, std::size_t channel) { setChannelStatus(layer, channel, MaskType_t::WARM_CHANNEL); }
  void setBadChannel(std::size_t layer, std::size_t channel) { setChannelStatus(layer, channel, MaskType_t::BAD_CHANNEL); }
  void setDeadChannel(std::size_t layer, std::size_t channel) { setChannelStatus(layer, channel, MaskType_t::DEAD_CHANNEL); }
  void setWarmChannel(std::size_t layer, std::size_t channel) { setChannelStatus(layer, channel, MaskType_t::WARM_CHANNEL); }

  MaskType_t getChannelStatus(std::size_t layer, std::size_t channel) const;
  bool isGoodChannel(std::size_t layer, std::size_t channel) const { return getChannelStatus(layer, channel) == MaskType_t::GOOD_CHANNEL; }
  bool isBadChannel(std::size_t layer, std::size_t channel) const { return getChannelStatus(layer, channel) == MaskType_t::GOOD_CHANNEL; }
  bool isDeadChannel(std::size_t layer, std::size_t channel) const { return getChannelStatus(layer, channel) == MaskType_t::GOOD_CHANNEL; }
  bool isWarmChannel(std::size_t layer, std::size_t channel) const { return getChannelStatus(layer, channel) == MaskType_t::GOOD_CHANNEL; }

 private:
  void init();
  std::size_t getChannelIndex(std::size_t layer, std::size_t channel) const;
  std::array<MaskType_t, constants::PADS_NLAYERS * constants::PADLAYER_MODULE_NCHANNELS> mChannelStatus;

  ClassDefNV(PadBadChannelMap, 1)
};

} // namespace o2::focal
#endif // ALICEO2_FOCAL_PADBADCHANNELMAP_H