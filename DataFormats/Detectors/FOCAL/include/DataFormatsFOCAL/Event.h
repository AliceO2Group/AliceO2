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
#ifndef ALICEO2_FOCAL_EVENT_H
#define ALICEO2_FOCAL_EVENT_H

#include <array>
#include <cstdint>
#include <gsl/span>
#include "Rtypes.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFOCAL/Constants.h"
#include "DataFormatsFOCAL/PixelChip.h"
#include "DataFormatsFOCAL/PixelChipRecord.h"

namespace o2::focal
{

class PadLayerEvent
{
 public:
  struct Header {
    uint8_t mHeader;
    uint8_t mBC;
    uint8_t mWADD;
    uint8_t mFourbits;
    uint8_t mTrailer;
  };
  struct Channel {
    uint16_t mADC;
    uint16_t mTOA;
    uint16_t mTOT;
  };

  void setHeader(unsigned int half, uint8_t header, uint8_t bc, uint8_t wadd, uint8_t fourbits, uint8_t trialer);
  void setChannel(unsigned int channel, uint16_t adc, uint16_t toa, uint16_t tot);
  void setCMN(unsigned int half, uint16_t adc, uint16_t toa, uint16_t tot);
  void setCalib(unsigned int half, uint16_t adc, uint16_t toa, uint16_t tot);

  const Header& getHeader(unsigned int half) const;
  const Channel& getChannel(unsigned int channel) const;
  const Channel& getCMN(unsigned int half) const;
  const Channel& getCalib(unsigned int half) const;

  std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> getADCs() const;
  std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> getTOAs() const;
  std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> getTOTs() const;

  void reset();

 private:
  void check_halfs(unsigned int half) const;
  void check_channel(unsigned int channel) const;

  std::array<Header, constants::PADLAYER_MODULE_NHALVES> mHeaders;
  std::array<Channel, constants::PADLAYER_MODULE_NCHANNELS> mChannels;
  std::array<Channel, constants::PADLAYER_MODULE_NHALVES> mCMN;
  std::array<Channel, constants::PADLAYER_MODULE_NHALVES> mCalib;
  ClassDefNV(PadLayerEvent, 1);
};

class PixelLayerEvent
{
 public:
  PixelLayerEvent() = default;
  ~PixelLayerEvent() = default;

  void addChip(const PixelChip& chip);
  void addChip(int feeID, int laneID, int chipID, uint16_t statusCode, gsl::span<const PixelHit> hits);
  const std::vector<PixelChip>& getChips() const { return mChips; }

  void reset();

 private:
  std::vector<PixelChip> mChips;
  ClassDefNV(PixelLayerEvent, 1);
};

class Event
{
 public:
  Event() = default;
  ~Event() = default;

  PadLayerEvent& getPadLayer(unsigned int index);
  const PadLayerEvent& getPadLayer(unsigned int index) const;
  void setPadLayer(unsigned int layer, const PadLayerEvent& event);

  PixelLayerEvent& getPixelLayer(unsigned int index);
  const PixelLayerEvent& getPixelLayer(unsigned int index) const;
  void setPixelLayerEvent(unsigned int layer, const PixelLayerEvent& event);

  const InteractionRecord& getInteractionRecord() const { return mInteractionRecord; }
  void setInteractionRecord(const InteractionRecord& ir) { mInteractionRecord = ir; }

  void reset();

  void construct(const o2::InteractionRecord& interaction, gsl::span<const PadLayerEvent> pads, gsl::span<const PixelChipRecord> eventPixels, gsl::span<const PixelHit> pixelHits);

  bool isInitialized() const { return mInitialized; }

 private:
  void check_pad_layers(unsigned int index) const;
  void check_pixel_layers(unsigned int index) const;

  InteractionRecord mInteractionRecord;
  std::array<PadLayerEvent, constants::PADS_NLAYERS> mPadLayers;
  std::array<PixelLayerEvent, constants::PIXELS_NLAYERS> mPixelLayers;
  bool mInitialized = false;

  ClassDefNV(Event, 1);
};

} // namespace o2::focal

#endif // ALICEO2_FOCAL_EVENT_H