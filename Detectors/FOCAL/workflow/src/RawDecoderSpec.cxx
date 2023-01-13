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
#include <sstream>
#include <set>

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
  std::unordered_map<int, int> numHBFFEE, numEventsFEE;
  std::unordered_map<int, std::vector<int>> numEventsHBFFEE;
  int numHBFPadsTF = 0, numEventsPadsTF = 0;
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
                  auto nEventsPads = decodePadData(rawbuffer, currentIR);
                  auto found = mNumEventsHBFPads.find(nEventsPads);
                  if (found != mNumEventsHBFPads.end()) {
                    found->second += 1;
                  } else {
                    mNumEventsHBFPads.insert(std::pair<int, int>{nEventsPads, 1});
                  }
                  numEventsPadsTF += nEventsPads;
                  numHBFPadsTF++;
                }
              } else if (currentendpoint == 0) {
                // Pixel data
                if (mUsePixelData) {
                  auto feeID = o2::raw::RDHUtils::getFEEID(rdh);
                  LOG(debug) << "Processing Pixel data from FEE " << feeID;
                  auto neventsPixels = decodePixelData(rawbuffer, currentIR, feeID);
                  auto found = numHBFFEE.find(feeID);
                  if (found != numHBFFEE.end()) {
                    found->second++;
                  } else {
                    numHBFFEE.insert(std::pair<int, int>{feeID, 1});
                  }
                  auto evFound = numEventsFEE.find(feeID);
                  if (evFound != numEventsFEE.end()) {
                    evFound->second += neventsPixels;
                  } else {
                    numEventsFEE.insert(std::pair<int, int>{feeID, neventsPixels});
                  }
                  auto evHBFFound = numEventsHBFFEE.find(feeID);
                  if (evHBFFound != numEventsHBFFEE.end()) {
                    evHBFFound->second.push_back(neventsPixels);
                  } else {
                    std::vector<int> evbuffer;
                    evbuffer.push_back(neventsPixels);
                    numEventsHBFFEE.insert(std::pair<int, std::vector<int>>{feeID, evbuffer});
                  }
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

  // Consistency check PixelEvents
  if (!consistencyCheckPixelFEE(numHBFFEE)) {
    std::cout << "Mismatch in number of HBF / TF between pixel FEEs" << std::endl;
    printCounters(numHBFFEE);
    mNumInconsistencyPixelHBF++;
  }
  int numHBFPixelsTF = maxCounter(numHBFFEE);
  mNumHBFPixels += numHBFPixelsTF;
  if (!consistencyCheckPixelFEE(numEventsFEE)) {
    std::cout << "Mismatch in number of events / TF between pixel FEEs" << std::endl;
    printCounters(numEventsFEE);
    mNumInconsistencyPixelEvent++;
  }
  mNumEventsPixels += maxCounter(numEventsFEE);
  if (!checkEventsHBFConsistency(numEventsHBFFEE)) {
    std::cout << "Mistmatch number of events / HBF between pixel FEEs" << std::endl;
    printEvents(numEventsHBFFEE);
    mNumInconsistencyPixelEventHBF++;
  }
  fillPixelEventHBFCount(numEventsHBFFEE);

  LOG(info)
    << "Found " << mHBFs.size() << " HBFs in timeframe";

  buildEvents();

  LOG(info) << "Found " << mOutputTriggerRecords.size() << " events in timeframe";

  sendOutput(ctx);
  mNumEventsPads += numEventsPadsTF;
  mNumHBFPads += numHBFPadsTF;
  mNumTimeframes++;
  auto foundHBFperTFPads = mNumHBFperTFPads.find(numHBFPadsTF);
  if (foundHBFperTFPads != mNumHBFperTFPads.end()) {
    foundHBFperTFPads->second++;
  } else {
    mNumHBFperTFPads.insert(std::pair<int, int>{numHBFPadsTF, 1});
  }
  auto foundHBFperTFPixels = mNumHBFperTFPixels.find(numHBFPixelsTF);
  if (foundHBFperTFPixels != mNumHBFperTFPixels.end()) {
    foundHBFperTFPixels->second++;
  } else {
    mNumHBFperTFPixels.insert(std::pair<int, int>{numHBFPixelsTF, 1});
  }
}

void RawDecoderSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  std::cout << "Number of timeframes:             " << mNumTimeframes << std::endl;
  std::cout << "Number of pad HBFs:               " << mNumHBFPads << std::endl;
  std::cout << "Number of pixel HBFs:             " << mNumHBFPixels << std::endl;
  for (auto& [hbfs, tfs] : mNumHBFperTFPads) {
    std::cout << "Pads - Number of TFs with " << hbfs << " HBFs: " << tfs << std::endl;
  }
  for (auto& [hbfs, tfs] : mNumHBFperTFPixels) {
    std::cout << "Pixels - Number of TFs with " << hbfs << " HBFs: " << tfs << std::endl;
  }
  std::cout << "Number of pad events:             " << mNumEventsPads << std::endl;
  std::cout << "Number of pixel events:           " << mNumEventsPixels << std::endl;
  for (auto& [nevents, nHBF] : mNumEventsHBFPads) {
    std::cout << "Number of HBFs with " << nevents << " pad events:    " << nHBF << std::endl;
  }
  for (auto& [nevents, nHBF] : mNumEventsHBFPixels) {
    std::cout << "Number of HBFs with " << nevents << " pixel events:    " << nHBF << std::endl;
  }
  std::cout << "Number of inconsistencies between pixel FEEs: " << mNumInconsistencyPixelHBF << " HBFs, " << mNumInconsistencyPixelEvent << " events, " << mNumInconsistencyPixelEventHBF << " events / HBF" << std::endl;
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

int RawDecoderSpec::decodePadData(const gsl::span<const char> padWords, o2::InteractionRecord& interaction)
{
  LOG(debug) << "Decoding pad data for Orbit " << interaction.orbit << ", BC " << interaction.bc;
  constexpr std::size_t EVENTSIZEPADGBT = 1180,
                        EVENTSIZECHAR = EVENTSIZEPADGBT * sizeof(PadGBTWord) / sizeof(char);
  auto nevents = padWords.size() / (EVENTSIZECHAR);
  for (int ievent = 0; ievent < nevents; ievent++) {
    decodePadEvent(padWords.subspan(EVENTSIZECHAR * ievent, EVENTSIZECHAR), interaction);
  }
  return nevents;
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

int RawDecoderSpec::decodePixelData(const gsl::span<const char> pixelWords, o2::InteractionRecord& interaction, int feeID)
{
  LOG(debug) << "Decoding pixel data for Orbit " << interaction.orbit << ", BC " << interaction.bc;
  auto fee = feeID & 0x00FF,
       branch = (feeID & 0x0F00) >> 8;
  int layer = fee < 2 ? 0 : 1;

  gsl::span<const o2::itsmft::GBTWord> pixelpayload(reinterpret_cast<const o2::itsmft::GBTWord*>(pixelWords.data()), sizeof(o2::itsmft::GBTWord) / sizeof(char));
  mPixelDecoder.reset();
  mPixelDecoder.decodeEvent(pixelpayload);

  std::map<o2::InteractionRecord, HBFData>::iterator foundHBF = mHBFs.end();

  int nevents = 0;
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
    nevents++;
  }
  return nevents;
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
        mOutputTriggerRecords.emplace_back(hbir, startPads, constants::PADS_NLAYERS, startHits, 0, startHits, 0);
      }
    }
  }
}

bool RawDecoderSpec::consistencyCheckPixelFEE(const std::unordered_map<int, int>& counters) const
{
  bool initialized = false;
  bool discrepancy = false;
  int current = -1;
  for (auto& [fee, value] : counters) {
    if (!initialized) {
      current = value;
      initialized = true;
    }
    if (value != current) {
      discrepancy = true;
      break;
    }
  }

  return !discrepancy;
}

bool RawDecoderSpec::checkEventsHBFConsistency(const std::unordered_map<int, std::vector<int>>& counters) const
{
  bool initialized = false;
  bool discrepancy = false;
  std::vector<int> current;
  for (auto& [fee, events] : counters) {
    if (!initialized) {
      current = events;
      initialized = true;
    }
    if (events != current) {
      discrepancy = true;
    }
  }
  return !discrepancy;
}

int RawDecoderSpec::maxCounter(const std::unordered_map<int, int>& counters) const
{
  int maxCounter = 0;
  for (auto& [fee, counter] : counters) {
    if (counter > maxCounter) {
      maxCounter = counter;
    }
  }

  return maxCounter;
}

void RawDecoderSpec::printCounters(const std::unordered_map<int, int>& counters) const
{
  for (auto& [fee, counter] : counters) {
    std::cout << "  FEE " << fee << ": " << counter << " counts ..." << std::endl;
  }
}

void RawDecoderSpec::printEvents(const std::unordered_map<int, std::vector<int>>& counters) const
{
  for (auto& [fee, events] : counters) {
    std::stringstream stringbuilder;
    bool first = true;
    for (auto ev : events) {
      if (first) {
        first = false;
      } else {
        stringbuilder << ", ";
      }
      stringbuilder << ev;
    }
    std::cout << "  FEE " << fee << ": " << stringbuilder.str() << " events ..." << std::endl;
  }
}

void RawDecoderSpec::fillPixelEventHBFCount(const std::unordered_map<int, std::vector<int>>& counters)
{
  // take FEE with the max number of events - expecting other FEEs might miss events
  int maxFEE = 0;
  int current = -1;
  for (auto& [fee, events] : counters) {
    int sum = 0;
    for (auto ev : events) {
      sum += ev;
    }
    if (sum > current) {
      maxFEE = fee;
      current = sum;
    }
  }
  auto en = counters.find(maxFEE);
  if (en != counters.end()) {
    for (auto nEventsPixels : en->second) {
      auto found = mNumEventsHBFPixels.find(nEventsPixels);
      if (found != mNumEventsHBFPixels.end()) {
        found->second += 1;
      } else {
        mNumEventsHBFPixels.insert(std::pair<int, int>{nEventsPixels, 1});
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