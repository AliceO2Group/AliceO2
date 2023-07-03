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
#include <cfloat>
#include <DataFormatsFOCAL/Constants.h>
#include <FOCALCalib/PadPedestal.h>

#include <fairlogger/Logger.h>

using namespace o2::focal;

void PadPedestal::clear()
{
}

bool PadPedestal::operator==(const PadPedestal& rhs) const
{
  if (mPedestalValues.size() != rhs.mPedestalValues.size()) {
    LOG(debug) << "Error size: this " << mPedestalValues.size() << ", other " << rhs.mPedestalValues.size();
    return false;
  }
  bool failure = false;
  // check equalty of content based on other object
  for (auto [channel, pedestal] : rhs.mPedestalValues) {
    auto found = mPedestalValues.find(channel);
    if (found == mPedestalValues.end()) {
      LOG(debug) << "Key not found this: Layer " << channel.mLayer << ", channel " << channel.mChannel;
      failure = true;
    } else {
      if (std::abs(found->second - pedestal) > DBL_EPSILON) {
        LOG(debug) << "Value error channel layer " << channel.mLayer << ", channel " << channel.mChannel << " this " << found->second << ", other " << pedestal;
        failure = true;
      }
    }
  }
  if (failure) {
    return false;
  }
  // check equality of content based on this object
  for (auto [channel, pedestal] : mPedestalValues) {
    auto found = rhs.mPedestalValues.find(channel);
    if (found == rhs.mPedestalValues.end()) {
      LOG(debug) << "Key not found other: Layer " << channel.mLayer << ", channel: " << channel.mChannel;
      failure = true;
    } else {
      if (std::abs(found->second - pedestal) > DBL_EPSILON) {
        LOG(debug) << "Value error channel layer " << channel.mLayer << ", channel " << channel.mChannel << " other " << found->second << ", this " << pedestal;
        failure = true;
      }
    }
  }
  return !failure;
}

void PadPedestal::setPedestal(std::size_t layer, std::size_t channel, double pedestal)
{
  if (layer >= constants::PADS_NLAYERS) {
    throw InvalidChannelException(layer, channel);
  }
  if (channel >= constants::PADLAYER_MODULE_NCHANNELS) {
    throw InvalidChannelException(layer, channel);
  }
  auto found = mPedestalValues.find({layer, channel});
  if (found == mPedestalValues.end()) {
    mPedestalValues.insert({{layer, channel}, pedestal});
  } else {
    found->second = pedestal;
  }
}

double PadPedestal::getPedestal(std::size_t layer, std::size_t channel) const
{
  auto found = mPedestalValues.find({layer, channel});
  if (found == mPedestalValues.end()) {
    throw InvalidChannelException(layer, channel);
  }
  return found->second;
}

TH1* PadPedestal::getHistogramRepresentation(int layer) const
{
  if (layer > 18) {
    throw InvalidLayerException(layer);
  }
  std::string histname = "PedestalsLayer" + std::to_string(layer),
              histtitle = "Pedestals in layer " + std::to_string(layer);
  auto pedestalHist = new TH1F(histname.data(), histtitle.data(), constants::PADLAYER_MODULE_NCHANNELS, -0.5, constants::PADLAYER_MODULE_NCHANNELS - 0.5);
  pedestalHist->SetDirectory(nullptr);
  pedestalHist->SetStats(false);
  pedestalHist->SetXTitle("Channel");
  pedestalHist->SetYTitle("Pedestal (ADC counts)");
  for (std::size_t channel = 0; channel < constants::PADLAYER_MODULE_NCHANNELS; channel++) {
    double pedestal = 0;
    try {
      pedestal = getPedestal(layer, channel);
    } catch (InvalidChannelException& e) {
      LOG(error) << e.what();
    }
    pedestalHist->SetBinContent(channel + 1, pedestal);
  }
  return pedestalHist;
}
std::array<TH1*, 18> PadPedestal::getLayerHistogramRepresentations() const
{
  std::array<TH1*, 18> pedestalHistograms;
  for (int layer = 0; layer < 18; layer++) {
    pedestalHistograms[layer] = getHistogramRepresentation(layer);
  }
  return pedestalHistograms;
}