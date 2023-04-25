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
#ifndef ALICEO2_FOCAL_PADPEDESTAL_H
#define ALICEO2_FOCAL_PADPEDESTAL_H

#include <array>
#include <exception>
#include <string>
#include <unordered_map>

#include <boost/container_hash/hash.hpp>

#include "TH1.h"
#include "Rtypes.h"

namespace o2::focal
{

class PadPedestal
{
 public:
  struct ChannelID {
    std::size_t mLayer;
    std::size_t mChannel;

    bool operator==(const ChannelID& other) const
    {
      return mLayer == other.mLayer && mChannel == other.mChannel;
    }
  };

  struct ChannelIDHasher {

    /// \brief Functor implementation
    /// \param s Channel for which to determine a hash value
    /// \return hash value for channel ID
    size_t operator()(const ChannelID& s) const
    {
      std::size_t seed = 0;
      boost::hash_combine(seed, s.mLayer);
      boost::hash_combine(seed, s.mChannel);
      return seed;
    }
  };

  class InvalidChannelException : public std::exception
  {
   public:
    InvalidChannelException(std::size_t layer, std::size_t channel) : mLayer(layer), mChannel(channel)
    {
      mMessage = "Invalid channel: Layer " + std::to_string(mLayer) + ", channel " + std::to_string(mChannel);
    }
    ~InvalidChannelException() noexcept final = default;

    const char* what() const noexcept final { return mMessage.data(); }

    std::size_t getLayer() const noexcept { return mLayer; }
    std::size_t getChannel() const noexcept { return mChannel; }

   private:
    std::size_t mLayer;
    std::size_t mChannel;
    std::string mMessage;
  };

  class InvalidLayerException : public std::exception
  {
   public:
    InvalidLayerException(std::size_t layer) : mLayer(layer)
    {
      mMessage = "Access to invalid layer " + std::to_string(layer);
    }
    ~InvalidLayerException() noexcept final = default;

    const char* what() const noexcept final { return mMessage.data(); }
    std::size_t getLayer() const noexcept { return mLayer; }

   private:
    std::size_t mLayer = 0;
    std::string mMessage;
  };

  PadPedestal() = default;
  ~PadPedestal() = default;

  void clear();
  void setPedestal(std::size_t layer, std::size_t channel, double pedestal);
  double getPedestal(std::size_t layer, std::size_t channel) const;

  TH1* getHistogramRepresentation(int layer) const;
  std::array<TH1*, 18> getLayerHistogramRepresentations() const;

 private:
  std::unordered_map<ChannelID, double, ChannelIDHasher> mPedestalValues;
  ClassDefNV(PadPedestal, 1);
};

} // namespace o2::focal
#endif // ALICEO2_FOCAL_PADPEDESTAL_H