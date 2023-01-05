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
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/WorkflowSpec.h"
#include "CommonConstants/Triggers.h"
#include "DetectorsRaw/RDHUtils.h"
#include "FOCALReconstruction/PadWord.h"
#include "FOCALWorkflow/RawDecoderSpec.h"
#include "ITSMFTReconstruction/GBTWord.h"

#include <iostream>

using namespace o2::focal::reco_workflow;

void RawDecoderSpec::init(framework::InitContext& ctx)
{
}

void RawDecoderSpec::run(framework::ProcessingContext& ctx)
{
  LOG(info) << "Running FOCAL decoding";
  resetContainers();

  int inputs = 0;
  std::vector<char> rawbuffer;
  int currentendpoint = 0;
  bool nextHBF = true;
  o2::InteractionRecord currentIR;
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {
    if (rawData.header != nullptr && rawData.payload != nullptr) {
      const auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(rawData);
      auto header = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(rawData);
      LOG(debug) << "Channel " << header->dataOrigin.str << "/" << header->dataDescription.str << "/" << header->subSpecification;

      gsl::span<const char> databuffer(rawData.payload, payloadSize);
      int currentpos = 0;
      while (currentpos < databuffer.size()) {
        auto rdh = reinterpret_cast<const o2::header::RDHAny*>(databuffer.data() + currentpos);
        if (mDebugMode) {
          o2::raw::RDHUtils::printRDH(rdh);
        }
        if (o2::raw::RDHUtils::getMemorySize(rdh) == o2::raw::RDHUtils::getHeaderSize(rdh)) {
          auto trigger = o2::raw::RDHUtils::getTriggerType(rdh);
          if (trigger & o2::trigger::SOT || trigger & o2::trigger::HB) {
            if (o2::raw::RDHUtils::getStop(rdh)) {
              LOG(debug) << "Stop bit received - processing payload";
              // Data ready
              if (currentendpoint == 1) {
                // Pad data
                if (mUsePadData) {
                  LOG(debug) << "Processing PAD data";
                  decodePadData(rawbuffer, currentIR);
                }
              } else if (currentendpoint == 0) {
                // Pixel data
                if (mUsePixelData) {
                  auto feeID = o2::raw::RDHUtils::getFEEID(rdh);
                  LOG(debug) << "Processing Pixel data from FEE " << feeID;
                  decodePixelData(rawbuffer, currentIR, feeID);
                }
              } else {
                LOG(error) << "Unsupported endpoint " << currentendpoint;
              }
              nextHBF = true;
            } else {
              LOG(debug) << "New HBF or Timeframe";
              nextHBF = true;
            }
          }
          currentpos += o2::raw::RDHUtils::getOffsetToNext(rdh);
          continue;
        }

        auto trigger = o2::raw::RDHUtils::getTriggerType(rdh);
        if (trigger & o2::trigger::SOT || trigger & o2::trigger::HB) {
          // HBF trigger could be part of regular payoad RDH (pixels)
          nextHBF = true;
        }

        if (nextHBF) {
          rawbuffer.clear();
          // Get interaction record for HBF
          currentIR.bc = o2::raw::RDHUtils::getTriggerBC(rdh);
          currentIR.orbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
          currentendpoint = o2::raw::RDHUtils::getEndPointID(rdh);
          LOG(debug) << "New HBF " << currentIR.orbit << " / " << currentIR.bc << ", endpoint " << currentendpoint;
          nextHBF = false;
        }

        // non-0 payload size:
        auto payloadsize = o2::raw::RDHUtils::getMemorySize(rdh) - o2::raw::RDHUtils::getHeaderSize(rdh);
        int endpoint = static_cast<int>(o2::raw::RDHUtils::getEndPointID(rdh));
        LOG(debug) << "Next RDH: ";
        LOG(debug) << "Found endpoint              " << endpoint;
        LOG(debug) << "Found trigger BC:           " << o2::raw::RDHUtils::getTriggerBC(rdh);
        LOG(debug) << "Found trigger Oribt:        " << o2::raw::RDHUtils::getTriggerOrbit(rdh);
        LOG(debug) << "Found payload size:         " << payloadsize;
        LOG(debug) << "Found offset to next:       " << o2::raw::RDHUtils::getOffsetToNext(rdh);
        LOG(debug) << "Stop bit:                   " << (o2::raw::RDHUtils::getStop(rdh) ? "yes" : "no");
        LOG(debug) << "Number of GBT words:        " << (payloadsize * sizeof(char) / (endpoint == 1 ? sizeof(o2::focal::PadGBTWord) : sizeof(o2::itsmft::GBTWord)));
        auto page_payload = databuffer.subspan(currentpos + o2::raw::RDHUtils::getHeaderSize(rdh), payloadsize);
        std::copy(page_payload.begin(), page_payload.end(), std::back_inserter(rawbuffer));
        currentpos += o2::raw::RDHUtils::getOffsetToNext(rdh);
      }
    } else {
      LOG(error) << "Input " << inputs << ": Either header or payload is nullptr";
    }
    inputs++;
  }

  LOG(info) << "Found " << mHBFs.size() << " HBFs in timeframe";

  buildEvents();

  LOG(info) << "Found " << mOutputTriggerRecords.size() << " events in timeframe";

  sendOutput(ctx);
}

void RawDecoderSpec::sendOutput(framework::ProcessingContext& ctx)
{
  ctx.outputs().snapshot(framework::Output{o2::header::gDataOriginFOC, "PADLAYERS", mOutputSubspec, framework::Lifetime::Timeframe}, mOutputPadLayers);
  ctx.outputs().snapshot(framework::Output{o2::header::gDataOriginFOC, "PIXELHITS", mOutputSubspec, framework::Lifetime::Timeframe}, mOutputPixelHits);
  ctx.outputs().snapshot(framework::Output{o2::header::gDataOriginFOC, "PIXELCHIPS", mOutputSubspec, framework::Lifetime::Timeframe}, mOutputPixelChips);
  ctx.outputs().snapshot(framework::Output{o2::header::gDataOriginFOC, "TRIGGERS", mOutputSubspec, framework::Lifetime::Timeframe}, mOutputTriggerRecords);
}

void RawDecoderSpec::resetContainers()
{
  mHBFs.clear();
  mOutputPadLayers.clear();
  mOutputPixelChips.clear();
  mOutputPixelHits.clear();
  mOutputTriggerRecords.clear();
}

void RawDecoderSpec::decodePadData(const gsl::span<const char> padWords, o2::InteractionRecord& interaction)
{
  LOG(debug) << "Decoding pad data for Orbit " << interaction.orbit << ", BC " << interaction.bc;
  constexpr std::size_t EVENTSIZEPADGBT = 1180,
                        EVENTSIZECHAR = EVENTSIZEPADGBT * sizeof(PadGBTWord) / sizeof(char);
  auto nevents = padWords.size() / (EVENTSIZECHAR);
  for (int ievent = 0; ievent < nevents; ievent++) {
    decodePadEvent(padWords.subspan(EVENTSIZECHAR * ievent, EVENTSIZECHAR), interaction);
  }
}

void RawDecoderSpec::decodePadEvent(const gsl::span<const char> padWords, o2::InteractionRecord& interaction)
{
  gsl::span<const PadGBTWord> padWordsGBT(reinterpret_cast<const PadGBTWord*>(padWords.data()), padWords.size() / sizeof(PadGBTWord));
  mPadDecoder.reset();
  mPadDecoder.decodeEvent(padWordsGBT);
  std::map<o2::InteractionRecord, HBFData>::iterator foundHBF = mHBFs.find(interaction);
  if (foundHBF == mHBFs.end()) {
    // New event, create new entry
    HBFData nexthbf;
    auto res = mHBFs.insert({interaction, nexthbf});
    foundHBF = res.first;
  }
  foundHBF->second.mPadEvents.push_back(createPadLayerEvent(mPadDecoder.getData()));
}

void RawDecoderSpec::decodePixelData(const gsl::span<const char> pixelWords, o2::InteractionRecord& interaction, int feeID)
{
  LOG(debug) << "Decoding pixel data for Orbit " << interaction.orbit << ", BC " << interaction.bc;
  auto fee = feeID & 0x00FF,
       branch = (feeID & 0x0F00) >> 8;
  int layer = fee < 2 ? 0 : 1;

  gsl::span<const o2::itsmft::GBTWord> pixelpayload(reinterpret_cast<const o2::itsmft::GBTWord*>(pixelWords.data()), sizeof(o2::itsmft::GBTWord) / sizeof(char));
  mPixelDecoder.reset();
  mPixelDecoder.decodeEvent(pixelpayload);

  std::map<o2::InteractionRecord, HBFData>::iterator foundHBF = mHBFs.end();

  for (auto& [trigger, chipdata] : mPixelDecoder.getChipData()) {
    LOG(debug) << "Found trigger orbit " << trigger.orbit << ", BC " << trigger.bc;
    if (foundHBF == mHBFs.end()) {
      // take HBF from the first trigger as BC in RDH for pixel data is unreliable
      foundHBF = mHBFs.find(interaction);
      if (foundHBF == mHBFs.end()) {
        // New event, create new entry
        HBFData nexthbf;
        auto res = mHBFs.insert({trigger, nexthbf});
        foundHBF = res.first;
      }
    }
    // check if the trigger already exists
    auto triggerfound = std::find(foundHBF->second.mPixelTriggers.begin(), foundHBF->second.mPixelTriggers.end(), trigger);
    if (triggerfound != foundHBF->second.mPixelTriggers.end()) {
      //
      auto index = triggerfound - foundHBF->second.mPixelTriggers.begin();
      fillChipsToLayer(foundHBF->second.mPixelEvent[index][layer], chipdata);
    } else {
      // new trigger
      std::array<PixelLayerEvent, constants::PIXELS_NLAYERS> nextevent;
      foundHBF->second.mPixelEvent.push_back(nextevent);
      foundHBF->second.mPixelTriggers.push_back(trigger);
      auto& current = foundHBF->second.mPixelEvent.back();
      fillChipsToLayer(current[layer], chipdata);
    }
  }
}

std::array<o2::focal::PadLayerEvent, o2::focal::constants::PADS_NLAYERS> RawDecoderSpec::createPadLayerEvent(const o2::focal::PadData& data) const
{
  std::array<PadLayerEvent, constants::PADS_NLAYERS> result;
  for (std::size_t ilayer = 0; ilayer < constants::PADS_NLAYERS; ilayer++) {
    auto& asic = data.getDataForASIC(ilayer).getASIC();
    int bc[2];
    for (std::size_t ihalf = 0; ihalf < constants::PADLAYER_MODULE_NHALVES; ihalf++) {
      auto header = asic.getHeader(ihalf);
      bc[ihalf] = header.mBCID;
      auto calib = asic.getCalib(ihalf);
      auto cmn = asic.getCMN(ihalf);
      result[ilayer].setHeader(ihalf, header.getHeader(), header.getBCID(), header.getWadd(), header.getFourbit(), header.getTrailer());
      result[ilayer].setCalib(ihalf, calib.getADC(), calib.getTOA(), calib.getTOT());
      result[ilayer].setCMN(ihalf, cmn.getADC(), cmn.getTOA(), cmn.getTOT());
    }
    for (std::size_t ichannel = 0; ichannel < constants::PADLAYER_MODULE_NCHANNELS; ichannel++) {
      auto channel = asic.getChannel(ichannel);
      result[ilayer].setChannel(ichannel, channel.getADC(), channel.getTOA(), channel.getTOT());
    }
  }

  return result;
}

void RawDecoderSpec::fillChipsToLayer(o2::focal::PixelLayerEvent& pixellayer, const gsl::span<const o2::focal::PixelChip>& chipData)
{
  for (const auto& chip : chipData) {
    pixellayer.addChip(chip);
  }
}

void RawDecoderSpec::fillEventPixeHitContainer(std::vector<PixelHit>& eventHits, std::vector<PixelChipRecord>& eventChips, const PixelLayerEvent& pixelLayer, int layerIndex)
{
  for (auto& chip : pixelLayer.getChips()) {
    auto starthits = eventHits.size();
    auto& chipHits = chip.mHits;
    std::copy(chipHits.begin(), chipHits.end(), std::back_inserter(eventHits));
    eventChips.emplace_back(layerIndex, chip.mLaneID, chip.mChipID, starthits, chipHits.size());
  }
}

void RawDecoderSpec::buildEvents()
{
  LOG(debug) << "Start building events" << std::endl;
  for (const auto& [hbir, hbf] : mHBFs) {
    if (mUsePadData && mUsePixelData) {
      LOG(debug) << "Processing HBF with IR: " << hbir.orbit << " / " << hbir.bc << std::endl;
      // check consistency in number of events between triggers, pixels and pads
      // in case all events are in the stream
      if (!(hbf.mPadEvents.size() == hbf.mPixelEvent.size() == hbf.mPixelTriggers.size())) {
        LOG(error) << "Inconsistent number of events in HBF for pads (" << hbf.mPadEvents.size() << ") and pixels (" << hbf.mPixelEvent.size() << ") - " << hbf.mPixelTriggers.size() << " triggers";
        continue;
      }
      LOG(debug) << "HBF: " << hbf.mPixelTriggers.size() << " triggers, " << hbf.mPadEvents.size() << " pad events, " << hbf.mPixelEvent.size() << " pixel events" << std::endl;
      for (std::size_t itrg = 0; itrg < hbf.mPixelTriggers.size(); itrg++) {
        auto startPads = mOutputPadLayers.size(),
             startHits = mOutputPixelHits.size(),
             startChips = mOutputPixelChips.size();
        for (std::size_t ilayer = 0; ilayer < constants::PADS_NLAYERS; ilayer++) {
          mOutputPadLayers.push_back(hbf.mPadEvents[itrg][ilayer]);
        }
        std::vector<PixelHit> eventHits;
        std::vector<PixelChipRecord> eventPixels;
        for (std::size_t ilayer = 0; ilayer < constants::PIXELS_NLAYERS; ilayer++) {
          fillEventPixeHitContainer(eventHits, eventPixels, hbf.mPixelEvent[itrg][ilayer], ilayer);
        }
        std::copy(eventHits.begin(), eventHits.end(), std::back_inserter(mOutputPixelHits));
        std::copy(eventPixels.begin(), eventPixels.end(), std::back_inserter(mOutputPixelChips));
        mOutputTriggerRecords.emplace_back(hbf.mPixelTriggers[itrg], startPads, constants::PADS_NLAYERS, startHits, eventPixels.size(), startHits, eventHits.size());
      }
    } else if (mUsePixelData) {
      // only pixel data available, merge pixel layers and interaction record
      if (!(hbf.mPixelEvent.size() == hbf.mPixelTriggers.size())) {
        LOG(error) << "Inconsistent number of pixel events (" << hbf.mPixelEvent.size() << ") and triggers (" << hbf.mPixelTriggers.size() << ") in HBF";
        continue;
      }
      for (std::size_t itrg = 0; itrg < hbf.mPixelTriggers.size(); itrg++) {
        auto startPads = mOutputPadLayers.size(),
             startHits = mOutputPixelHits.size(),
             startChips = mOutputPixelChips.size();
        std::vector<PixelHit> eventHits;
        std::vector<PixelChipRecord> eventPixels;
        for (std::size_t ilayer = 0; ilayer < constants::PIXELS_NLAYERS; ilayer++) {
          fillEventPixeHitContainer(eventHits, eventPixels, hbf.mPixelEvent[itrg][ilayer], ilayer);
        }
        std::copy(eventHits.begin(), eventHits.end(), std::back_inserter(mOutputPixelHits));
        std::copy(eventPixels.begin(), eventPixels.end(), std::back_inserter(mOutputPixelChips));
        mOutputTriggerRecords.emplace_back(hbf.mPixelTriggers[itrg], startPads, 0, startHits, eventPixels.size(), startHits, eventHits.size());
      }
    } else if (mUsePadData) {
      // only pad data available, set pad layers and use IR of the HBF
      for (std::size_t itrg = 0; itrg < hbf.mPadEvents.size(); itrg++) {
        auto startPads = mOutputPadLayers.size(),
             startHits = mOutputPixelHits.size(),
             startChips = mOutputPixelChips.size();
        o2::InteractionRecord fakeBC;
        fakeBC.orbit = hbir.orbit;
        fakeBC.bc = hbir.bc + itrg;
        for (std::size_t ilayer = 0; ilayer < constants::PADS_NLAYERS; ilayer++) {
          mOutputPadLayers.push_back(hbf.mPadEvents[itrg][ilayer]);
        }
        mOutputTriggerRecords.emplace_back(hbf.mPixelTriggers[itrg], startPads, constants::PADS_NLAYERS, startHits, 0, startHits, 0);
      }
    }
  }
}

o2::framework::DataProcessorSpec o2::focal::reco_workflow::getRawDecoderSpec(bool askDISTSTF, uint32_t outputSubspec, bool usePadData, bool usePixelData, bool debugMode)
{
  constexpr auto originFOC = o2::header::gDataOriginFOC;
  std::vector<o2::framework::OutputSpec> outputs;

  outputs.emplace_back(originFOC, "PADLAYERS", outputSubspec, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(originFOC, "PIXELHITS", outputSubspec, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(originFOC, "PIXELCHIPS", outputSubspec, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(originFOC, "TRIGGERS", outputSubspec, o2::framework::Lifetime::Timeframe);

  std::vector<o2::framework::InputSpec> inputs{{"stf", o2::framework::ConcreteDataTypeMatcher{originFOC, o2::header::gDataDescriptionRawData}, o2::framework::Lifetime::Optional}};
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{"FOCALRawDecoderSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::focal::reco_workflow::RawDecoderSpec>(outputSubspec, usePadData, usePixelData, debugMode),
                                          o2::framework::Options{}};
}