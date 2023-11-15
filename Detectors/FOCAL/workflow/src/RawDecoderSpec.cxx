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
  if (ctx.options().get<bool>("filterIncomplete")) {
    LOG(info) << "Enabling filtering of incomplete events in the pixel data";
    mFilterIncomplete = true;
  }
  if (ctx.options().get<bool>("displayInconsistent")) {
    LOG(info) << "Display additional information in case of inconsistency between pixel links";
    mDisplayInconsistent = true;
  }
  auto mappingfile = ctx.options().get<std::string>("pixelmapping");
  PixelMapper::MappingType_t mappingtype = PixelMapper::MappingType_t::MAPPING_UNKNOWN;
  auto chiptype = ctx.options().get<std::string>("pixeltype");
  if (chiptype == "IB") {
    LOG(info) << "Using mapping type: IB";
    mappingtype = PixelMapper::MappingType_t::MAPPING_IB;
  } else if (chiptype == "OB") {
    LOG(info) << "Using mapping type: OB";
    mappingtype = PixelMapper::MappingType_t::MAPPING_OB;
  } else {
    LOG(fatal) << "Unkown mapping type for pixels: " << chiptype;
  }
  if (mappingfile == "default") {
    LOG(info) << "Using default pixel mapping for pixel type " << chiptype;
    mPixelMapping = std::make_unique<PixelMapper>(mappingtype);
  } else {
    LOG(info) << "Using user-defined mapping: " << mappingfile;
    mPixelMapping = std::make_unique<PixelMapper>(PixelMapper::MappingType_t::MAPPING_UNKNOWN);
    mPixelMapping->setMappingFile(mappingfile, mappingtype);
  }
}

void RawDecoderSpec::run(framework::ProcessingContext& ctx)
{
  LOG(info) << "Running FOCAL decoding";
  resetContainers();
  mTimeframeHasPadData = false;
  mTimeframeHasPixelData = false;

  int inputs = 0;
  std::vector<char> rawbuffer;
  uint16_t currentfee = 0;
  o2::InteractionRecord currentIR;
  std::unordered_map<int, int> numHBFFEE, numEventsFEE;
  std::unordered_map<int, std::vector<int>> numEventsHBFFEE;
  int numHBFPadsTF = 0, numEventsPadsTF = 0;
  std::vector<int> expectFEEs;
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {
    if (rawData.header != nullptr && rawData.payload != nullptr) {
      const auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(rawData);
      auto header = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(rawData);
      LOG(debug) << "Channel " << header->dataOrigin.str << "/" << header->dataDescription.str << "/" << header->subSpecification;

      gsl::span<const char> databuffer(rawData.payload, payloadSize);
      int currentpos = 0;
      bool firstHBF = true;
      while (currentpos < databuffer.size()) {
        auto rdh = reinterpret_cast<const o2::header::RDHAny*>(databuffer.data() + currentpos);
        if (mDebugMode) {
          o2::raw::RDHUtils::printRDH(rdh);
        }

        if (o2::raw::RDHUtils::getMemorySize(rdh) > o2::raw::RDHUtils::getHeaderSize(rdh)) {
          // non-0 payload size:
          auto payloadsize = o2::raw::RDHUtils::getMemorySize(rdh) - o2::raw::RDHUtils::getHeaderSize(rdh);
          int endpoint = static_cast<int>(o2::raw::RDHUtils::getEndPointID(rdh));
          auto fee = o2::raw::RDHUtils::getFEEID(rdh);
          LOG(debug) << "Next RDH: ";
          LOG(debug) << "Found fee                   0x" << std::hex << fee << std::dec << " (System " << (fee == 0xcafe ? "Pads" : "Pixels") << ")";
          LOG(debug) << "Found trigger BC:           " << o2::raw::RDHUtils::getTriggerBC(rdh);
          LOG(debug) << "Found trigger Oribt:        " << o2::raw::RDHUtils::getTriggerOrbit(rdh);
          LOG(debug) << "Found payload size:         " << payloadsize;
          LOG(debug) << "Found offset to next:       " << o2::raw::RDHUtils::getOffsetToNext(rdh);
          LOG(debug) << "Stop bit:                   " << (o2::raw::RDHUtils::getStop(rdh) ? "yes" : "no");
          LOG(debug) << "Number of GBT words:        " << (payloadsize * sizeof(char) / (fee == 0xcafe ? sizeof(o2::focal::PadGBTWord) : sizeof(o2::itsmft::GBTWord)));
          auto page_payload = databuffer.subspan(currentpos + o2::raw::RDHUtils::getHeaderSize(rdh), payloadsize);
          std::copy(page_payload.begin(), page_payload.end(), std::back_inserter(rawbuffer));
        }

        auto trigger = o2::raw::RDHUtils::getTriggerType(rdh);
        if (trigger & o2::trigger::HB) {
          // HB trigger received
          if (o2::raw::RDHUtils::getStop(rdh)) {
            LOG(debug) << "Stop bit received - processing payload";
            // Data ready
            if (rawbuffer.size()) {
              // Only process if we actually have payload (skip empty HBF)
              if (currentfee == 0xcafe) { // Use FEE ID 0xcafe for PAD data
                // Pad data
                if (mUsePadData) {
                  LOG(debug) << "Processing PAD data";
                  auto nEventsPads = decodePadData(rawbuffer, currentIR);
                  mTimeframeHasPadData = true;
                  auto found = mNumEventsHBFPads.find(nEventsPads);
                  if (found != mNumEventsHBFPads.end()) {
                    found->second += 1;
                  } else {
                    mNumEventsHBFPads.insert(std::pair<int, int>{nEventsPads, 1});
                  }
                  numEventsPadsTF += nEventsPads;
                  numHBFPadsTF++;
                }
              } else { // All other FEEs are pixel FEEs
                // Pixel data
                if (mUsePixelData) {
                  auto feeID = o2::raw::RDHUtils::getFEEID(rdh);
                  if (firstHBF) {
                    auto found = std::find(expectFEEs.begin(), expectFEEs.end(), feeID);
                    if (found == expectFEEs.end()) {
                      expectFEEs.emplace_back(feeID);
                    }
                    firstHBF = false;
                  }
                  LOG(debug) << "Processing Pixel data from FEE " << feeID;
                  auto neventsPixels = decodePixelData(rawbuffer, currentIR, feeID);
                  mTimeframeHasPixelData = true;
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
              }
            } else {
              LOG(debug) << "Payload size 0 - skip empty HBF";
            }
            rawbuffer.clear();
          } else {
            // Get interaction record for HBF
            currentIR.bc = o2::raw::RDHUtils::getTriggerBC(rdh);
            currentIR.orbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
            currentfee = o2::raw::RDHUtils::getFEEID(rdh);
            LOG(debug) << "New HBF " << currentIR.orbit << " / " << currentIR.bc << ", FEE 0x" << std::hex << currentfee << std::dec;
          }
        }

        currentpos += o2::raw::RDHUtils::getOffsetToNext(rdh);
      }
    } else {
      LOG(error) << "Input " << inputs << ": Either header or payload is nullptr";
    }
    inputs++;
  }

  int numHBFPixelsTF = 0;
  if (mTimeframeHasPixelData) {
    // Consistency check PixelEvents
    if (!consistencyCheckPixelFEE(numHBFFEE)) {
      LOG(alarm) << "Mismatch in number of HBF / TF between pixel FEEs";
      if (mDisplayInconsistent) {
        printCounters(numHBFFEE);
      }
      mNumInconsistencyPixelHBF++;
    }
    numHBFPixelsTF = maxCounter(numHBFFEE);
    mNumHBFPixels += numHBFPixelsTF;
    if (!consistencyCheckPixelFEE(numEventsFEE)) {
      LOG(alarm) << "Mismatch in number of events / TF between pixel FEEs";
      if (mDisplayInconsistent) {
        printCounters(numEventsFEE);
      }
      mNumInconsistencyPixelEvent++;
    }
    mNumEventsPixels += maxCounter(numEventsFEE);
    if (!checkEventsHBFConsistency(numEventsHBFFEE)) {
      LOG(alarm) << "Mistmatch number of events / HBF between pixel FEEs";
      if (mDisplayInconsistent) {
        printEvents(numEventsHBFFEE);
      }
      mNumInconsistencyPixelEventHBF++;
    }
    fillPixelEventHBFCount(numEventsHBFFEE);

    if (mFilterIncomplete) {
      LOG(debug) << "Filtering incomplete pixel events";
      for (auto& hbf : mHBFs) {
        auto numErased = filterIncompletePixelsEventsHBF(hbf.second, expectFEEs);
        mNumEventsPixels -= numErased;
      }
    }
  }

  LOG(info) << "Found " << mHBFs.size() << " HBFs in timeframe";

  LOG(debug) << "EventBuilder: Pixels: " << (mTimeframeHasPixelData ? "yes" : "no");
  LOG(debug) << "EventBuilder: Pads:   " << (mTimeframeHasPadData ? "yes" : "no");
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

int RawDecoderSpec::decodePadData(const gsl::span<const char> padWords, o2::InteractionRecord& hbIR)
{
  LOG(debug) << "Decoding pad data for Orbit " << hbIR.orbit << ", BC " << hbIR.bc;
  constexpr std::size_t EVENTSIZEPADGBT = 1180,
                        EVENTSIZECHAR = EVENTSIZEPADGBT * sizeof(PadGBTWord) / sizeof(char);
  auto nevents = padWords.size() / (EVENTSIZECHAR);
  for (int ievent = 0; ievent < nevents; ievent++) {
    decodePadEvent(padWords.subspan(EVENTSIZECHAR * ievent, EVENTSIZECHAR), hbIR);
  }
  return nevents;
}

void RawDecoderSpec::decodePadEvent(const gsl::span<const char> padWords, o2::InteractionRecord& hbIR)
{
  gsl::span<const PadGBTWord> padWordsGBT(reinterpret_cast<const PadGBTWord*>(padWords.data()), padWords.size() / sizeof(PadGBTWord));
  mPadDecoder.reset();
  mPadDecoder.decodeEvent(padWordsGBT);
  std::map<o2::InteractionRecord, HBFData>::iterator foundHBF = mHBFs.find(hbIR);
  if (foundHBF == mHBFs.end()) {
    // New event, create new entry
    HBFData nexthbf;
    auto res = mHBFs.insert({hbIR, nexthbf});
    foundHBF = res.first;
  }
  foundHBF->second.mPadEvents.push_back(createPadLayerEvent(mPadDecoder.getData()));
}

int RawDecoderSpec::decodePixelData(const gsl::span<const char> pixelWords, o2::InteractionRecord& hbIR, int feeID)
{
  LOG(debug) << "Decoding pixel data for Orbit " << hbIR.orbit << ", BC " << hbIR.bc;

  gsl::span<const o2::itsmft::GBTWord> pixelpayload(reinterpret_cast<const o2::itsmft::GBTWord*>(pixelWords.data()), pixelWords.size() / sizeof(o2::itsmft::GBTWord));
  LOG(debug) << pixelWords.size() << " Bytes -> " << pixelpayload.size() << " GBT words";
  mPixelDecoder.reset();
  mPixelDecoder.decodeEvent(pixelpayload);

  std::map<o2::InteractionRecord, HBFData>::iterator foundHBF = mHBFs.end();

  int nevents = 0;
  for (auto& [trigger, chipdata] : mPixelDecoder.getChipData()) {
    LOG(debug) << "Found trigger orbit " << trigger.orbit << ", BC " << trigger.bc;
    if (trigger.orbit != hbIR.orbit) {
      LOG(debug) << "FEE 0x" << std::hex << feeID << std::dec << ": Discarding spurious trigger with Orbit " << trigger.orbit << " (HB " << hbIR.orbit << ")";
      continue;
    }
    int nhits = 0;
    // set fee for all chips
    for (auto& chip : chipdata) {
      nhits += chip.mHits.size();
    }
    // std::cout << "FEE 0x" << std::hex << feeID << std::dec << " (fee " << fee << ", branch " << branch << ", layer " << layer << "): Found " << chipdata.size() << " chips with " << nhits << " hits (total) for Orbit " << trigger.orbit << ", BC " << trigger.bc << std::endl;
    // for (const auto& chip : chipdata) {
    //  std::cout << "Chip " << static_cast<int>(chip.mChipID) << ", lane " << static_cast<int>(chip.mLaneID) << ": " << chip.mHits.size() << " hits ... (Dataframe "
    //            << (chip.isDataframe() ? "yes" : "no") << ", Emptyframe " << (chip.isEmptyframe() ? "yes" : "no") << ", Busy on " << (chip.isBusyOn() ? "yes" : "no") << ", Busy off " << (chip.isBusyOff() ? "yes" : "no") << ")" << std::endl;
    // }
    if (foundHBF == mHBFs.end()) {
      // take HBF from the first trigger as BC in RDH for pixel data is unreliable
      foundHBF = mHBFs.find(hbIR);
      if (foundHBF == mHBFs.end()) {
        // New event, create new entry
        HBFData nexthbf;
        auto res = mHBFs.insert({hbIR, nexthbf});
        foundHBF = res.first;
      }
    }
    // check if the trigger already exists
    auto triggerfound = std::find(foundHBF->second.mPixelTriggers.begin(), foundHBF->second.mPixelTriggers.end(), trigger);
    if (triggerfound != foundHBF->second.mPixelTriggers.end()) {
      //
      auto index = triggerfound - foundHBF->second.mPixelTriggers.begin();
      for (const auto& chip : chipdata) {
        try {
          auto chipPosition = mPixelMapping->getPosition(feeID, chip);
          fillChipToLayer(foundHBF->second.mPixelEvent[index][chipPosition.mLayer], chip, feeID);
        } catch (PixelMapper::InvalidChipException& e) {
          LOG(warning) << e;
        }
      }
      foundHBF->second.mFEEs[index].push_back(feeID);
    } else {
      // new trigger
      std::array<PixelLayerEvent, constants::PIXELS_NLAYERS> nextevent;
      foundHBF->second.mPixelEvent.push_back(nextevent);
      foundHBF->second.mPixelTriggers.push_back(trigger);
      auto& current = foundHBF->second.mPixelEvent.back();
      for (const auto& chip : chipdata) {
        try {
          auto chipPosition = mPixelMapping->getPosition(feeID, chip);
          fillChipToLayer(current[chipPosition.mLayer], chip, feeID);
        } catch (PixelMapper::InvalidChipException& e) {
          LOG(warning) << e;
        }
      }
      foundHBF->second.mFEEs.push_back({feeID});
    }
    nevents++;
  }
  return nevents;
}

std::array<o2::focal::PadLayerEvent, o2::focal::constants::PADS_NLAYERS> RawDecoderSpec::createPadLayerEvent(const o2::focal::PadData& data) const
{
  std::array<PadLayerEvent, constants::PADS_NLAYERS> result;
  std::array<uint8_t, 8> triggertimes;
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
    auto triggers = data.getDataForASIC(ilayer).getTriggerWords();
    for (std::size_t window = 0; window < constants::PADLAYER_WINDOW_LENGTH; window++) {
      std::fill(triggertimes.begin(), triggertimes.end(), 0);
      triggertimes[0] = triggers[window].mTrigger0;
      triggertimes[1] = triggers[window].mTrigger1;
      triggertimes[2] = triggers[window].mTrigger2;
      triggertimes[3] = triggers[window].mTrigger3;
      triggertimes[4] = triggers[window].mTrigger4;
      triggertimes[5] = triggers[window].mTrigger5;
      triggertimes[6] = triggers[window].mTrigger6;
      triggertimes[7] = triggers[window].mTrigger7;
      result[ilayer].setTrigger(window, triggers[window].mHeader0, triggers[window].mHeader1, triggertimes);
    }
  }

  return result;
}

void RawDecoderSpec::fillChipToLayer(o2::focal::PixelLayerEvent& pixellayer, const o2::focal::PixelChip& chipData, int feeID)
{
  int nhitsBefore = 0;
  int nchipsBefore = pixellayer.getChips().size();
  for (auto& chip : pixellayer.getChips()) {
    nhitsBefore += chip.mHits.size();
  }
  pixellayer.addChip(feeID, chipData.mLaneID, chipData.mChipID, chipData.mStatusCode, chipData.mHits);
  int nhitsAfter = 0;
  int nchipsAfter = pixellayer.getChips().size();
  for (auto& chip : pixellayer.getChips()) {
    nhitsAfter += chip.mHits.size();
  }
  // std::cout << "Adding to pixel layer: Chips before " << nchipsBefore << " / after " << nchipsAfter << ", hits before " << nhitsBefore << " / after " << nhitsAfter << std::endl;
}

void RawDecoderSpec::fillEventPixeHitContainer(std::vector<PixelHit>& eventHits, std::vector<PixelChipRecord>& eventChips, const PixelLayerEvent& pixelLayer, int layerIndex)
{
  for (auto& chip : pixelLayer.getChips()) {
    auto starthits = eventHits.size();
    auto& chipHits = chip.mHits;
    std::copy(chipHits.begin(), chipHits.end(), std::back_inserter(eventHits));
    eventChips.emplace_back(layerIndex, chip.mFeeID, chip.mLaneID, chip.mChipID, chip.mStatusCode, starthits, chipHits.size());
  }
}

void RawDecoderSpec::buildEvents()
{
  LOG(debug) << "Start building events" << std::endl;
  for (const auto& [hbir, hbf] : mHBFs) {
    if (mTimeframeHasPadData && mTimeframeHasPixelData) {
      LOG(debug) << "Processing HBF with IR: " << hbir.orbit << " / " << hbir.bc << std::endl;
      // check consistency in number of events between triggers, pixels and pads
      // in case all events are in the stream
      if ((hbf.mPadEvents.size() != hbf.mPixelEvent.size()) || (hbf.mPadEvents.size() != hbf.mPixelTriggers.size()) || (hbf.mPixelEvent.size() != hbf.mPixelTriggers.size())) {
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
        // std::cout << "Orbit " << hbf.mPixelTriggers[itrg].orbit << ", BC " << hbf.mPixelTriggers[itrg].bc << ": " << eventPixels.size() << " chips with " << eventHits.size() << " hits ..." << std::endl;
        mOutputTriggerRecords.emplace_back(hbf.mPixelTriggers[itrg], startPads, constants::PADS_NLAYERS, startChips, eventPixels.size(), startHits, eventHits.size());
      }
    } else if (mTimeframeHasPixelData) {
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
        mOutputTriggerRecords.emplace_back(hbf.mPixelTriggers[itrg], startPads, 0, startChips, eventPixels.size(), startHits, eventHits.size());
      }
    } else if (mTimeframeHasPadData) {
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
        mOutputTriggerRecords.emplace_back(hbir, startPads, constants::PADS_NLAYERS, startChips, 0, startHits, 0);
      }
    }
  }
}

int RawDecoderSpec::filterIncompletePixelsEventsHBF(HBFData& data, const std::vector<int>& expectFEEs)
{
  auto same = [](const std::vector<int>& lhs, const std::vector<int>& rhs) -> bool {
    bool missing = false;
    for (auto entry : lhs) {
      if (std::find(rhs.begin(), rhs.end(), entry) == rhs.end()) {
        missing = true;
        break;
      }
    }
    if (!missing) {
      for (auto entry : rhs) {
        if (std::find(lhs.begin(), lhs.end(), entry) == lhs.end()) {
          missing = true;
          break;
        }
      }
    }
    return missing;
  };
  std::vector<int> indexIncomplete;
  for (auto index = 0; index < data.mFEEs.size(); index++) {
    if (data.mFEEs[index].size() != expectFEEs.size()) {
      indexIncomplete.emplace_back(index);
      continue;
    }
    if (!same(data.mFEEs[index], expectFEEs)) {
      indexIncomplete.emplace_back(index);
    }
  }
  if (indexIncomplete.size()) {
    std::sort(indexIncomplete.begin(), indexIncomplete.end(), std::less<>());
    // start removing from the end, since erase will impact the indexing
    for (auto indexIter = indexIncomplete.rbegin(); indexIter != indexIncomplete.rend(); indexIter++) {
      auto iterPixelEvent = data.mPixelEvent.begin() + *indexIter;
      auto iterTrigger = data.mPixelTriggers.begin() + *indexIter;
      auto iterFEEs = data.mFEEs.begin() + *indexIter;
      data.mPixelEvent.erase(iterPixelEvent);
      data.mPixelTriggers.erase(iterTrigger);
      data.mFEEs.erase(iterFEEs);
    }
  }
  return indexIncomplete.size();
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
    LOG(info) << "  FEE 0x" << std::hex << fee << std::dec << ": " << counter << " counts ...";
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
    LOG(info) << "  FEE 0x" << std::hex << fee << std::dec << ": " << stringbuilder.str() << " events ...";
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

  std::vector<o2::framework::InputSpec> inputs{{"stf", o2::framework::ConcreteDataTypeMatcher{originFOC, o2::header::gDataDescriptionRawData}, o2::framework::Lifetime::Timeframe}};
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{"FOCALRawDecoderSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::focal::reco_workflow::RawDecoderSpec>(outputSubspec, usePadData, usePixelData, debugMode),
                                          o2::framework::Options{
                                            {"filterIncomplete", o2::framework::VariantType::Bool, false, {"Filter incomplete pixel events"}},
                                            {"displayInconsistent", o2::framework::VariantType::Bool, false, {"Display information about inconsistent timeframes"}},
                                            {"pixeltype", o2::framework::VariantType::String, "OB", {"Pixel mapping type"}},
                                            {"pixelmapping", o2::framework::VariantType::String, "default", {"File with pixel mapping"}}}};
}