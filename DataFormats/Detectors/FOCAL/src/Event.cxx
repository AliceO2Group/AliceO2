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
#include <algorithm>
#include <DataFormatsFOCAL/ErrorHandling.h>
#include <DataFormatsFOCAL/Event.h>
#include <iostream>

using namespace o2::focal;

PadLayerEvent& Event::getPadLayer(unsigned int index)
{
  check_pad_layers(index);
  return mPadLayers[index];
}

const PadLayerEvent& Event::getPadLayer(unsigned int index) const
{
  check_pad_layers(index);
  return mPadLayers[index];
}

PixelLayerEvent& Event::getPixelLayer(unsigned int index)
{
  check_pixel_layers(index);
  return mPixelLayers[index];
}
const PixelLayerEvent& Event::getPixelLayer(unsigned int index) const
{
  check_pixel_layers(index);
  return mPixelLayers[index];
}

void Event::setPadLayer(unsigned int layer, const PadLayerEvent& event)
{
  check_pad_layers(layer);
  mPadLayers[layer] = event;
}

void Event::setPixelLayerEvent(unsigned int layer, const PixelLayerEvent& event)
{
  check_pixel_layers(layer);
  mPixelLayers[layer] = event;
}

void Event::reset()
{
  mInitialized = false;
  for (auto& padlayer : mPadLayers) {
    padlayer.reset();
  }
  for (auto& pixellayer : mPixelLayers) {
    pixellayer.reset();
  }
}

void Event::construct(const o2::InteractionRecord& interaction, gsl::span<const PadLayerEvent> pads, gsl::span<const PixelChipRecord> eventPixels, gsl::span<const PixelHit> pixelHits)
{
  reset();
  mInteractionRecord = interaction;
  int ilayer = 0;
  for (auto& padlayer : pads) {
    mPadLayers[ilayer] = padlayer;
    ilayer++;
  }

  int currentlast = 0;
  for (auto& chip : eventPixels) {
    if (chip.getLayerID() > 1) {
      std::cerr << "Invalid layer ID chip " << chip.getChipID() << ": " << chip.getLayerID() << std::endl;
      continue;
    }
    if (chip.getNumberOfHits()) {
      if (chip.getFirstHit() >= pixelHits.size()) {
        std::cerr << "First hit index " << chip.getFirstHit() << " exceeding hit contiainer " << pixelHits.size() << std::endl;
        continue;
      }
      if (chip.getFirstHit() + chip.getNumberOfHits() - 1 >= pixelHits.size()) {
        std::cerr << "First hit index " << chip.getFirstHit() + chip.getNumberOfHits() - 1 << " exceeding hit contiainer " << pixelHits.size() << std::endl;
        continue;
      }
      mPixelLayers[chip.getLayerID()].addChip(chip.getFeeID(), chip.getLaneID(), chip.getChipID(), chip.getStatusCode(), pixelHits.subspan(chip.getFirstHit(), chip.getNumberOfHits()));
    }
    currentlast = chip.getFirstHit() + chip.getNumberOfHits();
  }
  if (currentlast < pixelHits.size()) {
    std::cerr << "Inconsistent number of hits / event : all chips -> " << currentlast << " hits, container size -> " << pixelHits.size() << std::endl;
  }
  mInitialized = true;
}

void Event::check_pad_layers(unsigned int index) const
{
  if (index >= constants::PADS_NLAYERS) {
    throw IndexExceptionEvent(index, constants::PADS_NLAYERS, IndexExceptionEvent::IndexType_t::PAD_LAYER);
  }
}

void Event::check_pixel_layers(unsigned int index) const
{
  if (index >= constants::PIXELS_NLAYERS) {
    throw IndexExceptionEvent(index, constants::PIXELS_NLAYERS, IndexExceptionEvent::IndexType_t::PIXEL_LAYER);
  }
}

void PadLayerEvent::setHeader(unsigned int half, uint8_t header, uint8_t bc, uint8_t wadd, uint8_t fourbits, uint8_t trailer)
{
  check_halfs(half);
  auto& asicheader = mHeaders[half];
  asicheader.mHeader = header;
  asicheader.mBC = bc;
  asicheader.mFourbits = fourbits;
  asicheader.mWADD = wadd;
  asicheader.mTrailer = trailer;
}

void PadLayerEvent::setChannel(unsigned int channel, uint16_t adc, uint16_t toa, uint16_t tot)
{
  check_channel(channel);
  auto& asicchannel = mChannels[channel];
  asicchannel.mADC = adc;
  asicchannel.mTOA = toa;
  asicchannel.mTOT = tot;
}

void PadLayerEvent::setCMN(unsigned int channel, uint16_t adc, uint16_t toa, uint16_t tot)
{
  check_halfs(channel);
  auto& cmn = mCMN[channel];
  cmn.mADC = adc;
  cmn.mTOA = toa;
  cmn.mTOT = tot;
}

void PadLayerEvent::setCalib(unsigned int channel, uint16_t adc, uint16_t toa, uint16_t tot)
{
  check_halfs(channel);
  auto& calib = mCalib[channel];
  calib.mADC = adc;
  calib.mTOA = toa;
  calib.mTOT = tot;
}

void PadLayerEvent::setTrigger(unsigned int window, uint32_t header0, uint32_t header1, const gsl::span<uint8_t> triggers)
{
  if (window >= constants::PADLAYER_WINDOW_LENGTH) {
    throw IndexExceptionEvent(window, constants::PADLAYER_WINDOW_LENGTH, IndexExceptionEvent::IndexType_t::TRIGGER_WINDOW);
  }
  auto& currenttrigger = mTriggers[window];
  currenttrigger.mHeader0 = header0;
  currenttrigger.mHeader1 = header1;
  std::copy(triggers.begin(), triggers.end(), currenttrigger.mTriggers.begin());
}

const PadLayerEvent::Header& PadLayerEvent::getHeader(unsigned int half) const
{
  check_halfs(half);
  return mHeaders[half];
}

const PadLayerEvent::Channel& PadLayerEvent::getChannel(unsigned int channel) const
{
  check_channel(channel);
  return mChannels[channel];
}
const PadLayerEvent::Channel& PadLayerEvent::getCMN(unsigned int half) const
{
  check_halfs(half);
  return mCMN[half];
}
const PadLayerEvent::Channel& PadLayerEvent::getCalib(unsigned int half) const
{
  check_halfs(half);
  return mCalib[half];
}

const PadLayerEvent::TriggerWindow& PadLayerEvent::getTrigger(unsigned int window) const
{
  if (window >= constants::PADLAYER_WINDOW_LENGTH) {
    throw IndexExceptionEvent(window, constants::PADLAYER_WINDOW_LENGTH, IndexExceptionEvent::IndexType_t::TRIGGER_WINDOW);
  }
  return mTriggers[window];
}

std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> PadLayerEvent::getADCs() const
{
  std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> adcs;
  for (std::size_t ichan = 0; ichan < constants::PADLAYER_MODULE_NCHANNELS; ichan++) {
    adcs[ichan] = mChannels[ichan].mADC;
  }
  return adcs;
}
std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> PadLayerEvent::getTOAs() const
{
  std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> toas;
  for (std::size_t ichan = 0; ichan < constants::PADLAYER_MODULE_NCHANNELS; ichan++) {
    toas[ichan] = mChannels[ichan].mTOA;
  }
  return toas;
}
std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> PadLayerEvent::getTOTs() const
{
  std::array<uint16_t, constants::PADLAYER_MODULE_NCHANNELS> tots;
  for (std::size_t ichan = 0; ichan < constants::PADLAYER_MODULE_NCHANNELS; ichan++) {
    tots[ichan] = mChannels[ichan].mTOT;
  }
  return tots;
}

void PadLayerEvent::reset()
{
  for (auto& header : mHeaders) {
    header.mBC = 0;
    header.mHeader = 0;
    header.mFourbits = 0;
    header.mWADD = 0;
    header.mTrailer = 0;
  }
  for (auto& chan : mChannels) {
    chan.mADC = 0;
    chan.mTOA = 0;
    chan.mTOT = 0;
  }
  for (auto& calib : mCalib) {
    calib.mADC = 0;
    calib.mTOA = 0;
    calib.mTOT = 0;
  }
  for (auto& cmn : mCMN) {
    cmn.mADC = 0;
    cmn.mTOA = 0;
    cmn.mTOT = 0;
  }
  for (auto& trg : mTriggers) {
    trg.mHeader0 = 0;
    trg.mHeader1 = 0;
    std::fill(trg.mTriggers.begin(), trg.mTriggers.end(), 0);
  }
}

void PadLayerEvent::check_halfs(unsigned int half) const
{
  if (half >= constants::PADLAYER_MODULE_NHALVES) {
    throw IndexExceptionEvent(half, constants::PADLAYER_MODULE_NHALVES, IndexExceptionEvent::IndexType_t::PAD_NHALVES);
  }
}

void PadLayerEvent::check_channel(unsigned int channel) const
{
  if (channel >= constants::PADLAYER_MODULE_NCHANNELS) {
    throw IndexExceptionEvent(channel, constants::PADLAYER_MODULE_NCHANNELS, IndexExceptionEvent::IndexType_t::PAD_CHANNEL);
  }
}

void PixelLayerEvent::addChip(const PixelChip& chip)
{
  auto found = std::find_if(mChips.begin(), mChips.end(), [&chip](const PixelChip& testchip) { return chip == testchip; });
  if (found != mChips.end()) {
    std::copy(chip.mHits.begin(), chip.mHits.end(), std::back_inserter(found->mHits));
  } else {
    mChips.push_back(chip);
  }
}

void PixelLayerEvent::addChip(int feeID, int laneID, int chipID, uint16_t statusCode, gsl::span<const PixelHit> hits)
{
  auto found = std::find_if(mChips.begin(), mChips.end(), [laneID, chipID, feeID](const PixelChip& testchip) { return chipID == testchip.mChipID && laneID == testchip.mLaneID && feeID == testchip.mFeeID; });
  if (found != mChips.end()) {
    std::copy(hits.begin(), hits.end(), std::back_inserter(found->mHits));
  } else {
    mChips.push_back({static_cast<uint8_t>(feeID), static_cast<uint8_t>(laneID), static_cast<uint8_t>(chipID), statusCode});
    auto& currentchip = mChips.back();
    std::copy(hits.begin(), hits.end(), std::back_inserter(currentchip.mHits));
  }
}

void PixelLayerEvent::reset()
{
  mChips.clear();
}
