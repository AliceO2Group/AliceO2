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

#include <fairlogger/Logger.h>
#include <CommonDataFormat/InteractionRecord.h>
#include <CommonConstants/Triggers.h>
#include <FOCALReconstruction/PixelDecoder.h>
#include <iostream>

using namespace o2::focal;

void PixelDecoder::reset()
{
  mPixelData.clear();
  mChipData.clear();
}

void PixelDecoder::decodeEvent(gsl::span<const o2::itsmft::GBTWord> payload)
{
  o2::InteractionRecord currentIR;
  bool physicsTrigger = false;
  mPixelData.clear();
  std::shared_ptr<PixelLaneHandler> currenttrigger;
  for (const auto& word : payload) {
    if (word.isDataHeader()) {
      // to be defined
    }
    if (word.isDataTrailer()) {
      // to be defined
    }
    if (word.isDiagnosticWord()) {
      // to be defined
    }
    if (word.isTriggerWord()) {
      auto lastIR = currentIR;
      currentIR.orbit = word.orbit;
      currentIR.bc = word.bc;
      if (word.triggerType & o2::trigger::PhT) {
        physicsTrigger = true;
      }
      if (lastIR != currentIR || !currenttrigger) {
        auto found = mPixelData.find(currentIR);
        if (found == mPixelData.end()) {
          currenttrigger = std::make_shared<PixelLaneHandler>();
          mPixelData[currentIR] = currenttrigger;
        } else {
          currenttrigger = found->second;
        }
      }
    }
    if (word.isData()) {
      auto* dataword = reinterpret_cast<const o2::itsmft::GBTData*>(&word);
      auto payload = gsl::span<const uint8_t>(reinterpret_cast<const uint8_t*>(word.getW8()), 9);
      int lane = -1;
      if (word.isDataIB()) {
        lane = dataword->getLaneIB();
      } else if (word.isDataOB()) {
        // lane = dataword->getLaneOB();
        // MF treat as if they would be IB lanes
        lane = dataword->getLaneIB();
      }
      if (lane >= PixelLaneHandler::NLANES) {
        // Discarding lanes
        continue;
      }
      currenttrigger->getLane(lane).append(payload);
    }
  }

  mChipData.clear();

  // std::cout << "Found " << mPixelData.size() << " triggers" << std::endl;
  LOG(debug) << "Found " << mPixelData.size() << " triggers";
  for (auto& [trigger, data] : mPixelData) {
    // std::cout << "Found trigger " << trigger.asString() << std::endl;
    LOG(debug) << "Found trigger " << trigger.asString();
    std::vector<PixelChip> combinedChips;
    int foundLanes = 0;
    for (int ilane = 0; ilane < PixelLaneHandler::NLANES; ilane++) {
      const auto& lane = data->getLane(ilane);
      if (lane.getPayload().size()) {
        // std::cout << "[Lane " << ilane << "] " << lane << std::endl;
        LOG(debug) << "[Lane " << ilane << "] " << lane;
        auto laneChips = decodeLane(ilane, lane.getPayload());
        auto chipsBefore = combinedChips.size();
        std::copy(laneChips.begin(), laneChips.end(), std::back_inserter(combinedChips));
        // std::cout << "Merging combined chips, before " << chipsBefore << ", after " << combinedChips.size() << std::endl;
        LOG(debug) << "Merging combined chips, before " << chipsBefore << ", after " << combinedChips.size();
        foundLanes++;
      }
    }
    // std::cout << "Trigger has " << combinedChips.size() << " chips from " << foundLanes << " lanes " << std::endl;
    LOG(debug) << "Trigger has " << combinedChips.size() << " chips from " << foundLanes << " lanes ";
    std::sort(combinedChips.begin(), combinedChips.end(), std::less<>());
    for (auto& chip : combinedChips) {
      // std::cout << "Chip " << static_cast<int>(chip.mChipID) << " [lane " << static_cast<int>(chip.mLaneID) << "], with " << chip.mHits.size() << " hit(s) ... " << std::endl;
      LOG(debug) << "Chip " << static_cast<int>(chip.mChipID) << " [lane " << static_cast<int>(chip.mLaneID) << "], with " << chip.mHits.size() << " hit(s) ... ";
    }
    mChipData[trigger] = combinedChips;
  }
}

PixelWord::PixelWordType PixelDecoder::getWordType(uint8_t payloadword)
{
  // native alpide words
  if (payloadword == 0xff) {
    return PixelWord::PixelWordType::IDLE;
  } else if (payloadword == 0xf1) {
    return PixelWord::PixelWordType::BUSY_ON;
  } else if (payloadword == 0xf0) {
    return PixelWord::PixelWordType::BUSY_OFF;
  } else if ((payloadword & 0xf0) == 0xa0) {
    return PixelWord::PixelWordType::CHIP_HEADER;
  } else if ((payloadword & 0xf0) == 0xb0) {
    return PixelWord::PixelWordType::CHIP_TRAILER;
  } else if ((payloadword & 0xf0) == 0xe0) {
    return PixelWord::PixelWordType::CHIP_EMPTYFRAME;
  } else if ((payloadword & 0xe0) == 0xc0) {
    return PixelWord::PixelWordType::REGION_HEADER;
  } else if ((payloadword & 0xc0) == 0x40) {
    return PixelWord::PixelWordType::DATA_SHORT;
  } else if ((payloadword & 0xc0) == 0x00) {
    return PixelWord::PixelWordType::DATA_LONG;
  }
  return PixelWord::PixelWordType::UNKNOWN;
}

uint16_t PixelDecoder::AlpideY(uint16_t address)
{
  return address / 2;
}

uint16_t PixelDecoder::AlpideX(uint8_t region, uint8_t encoder, uint16_t address)
{
  int x = region * 32 + encoder * 2;
  if (address % 4 == 1) {
    x++;
  }
  if (address % 4 == 2) {
    x++;
  }
  return x;
}

std::vector<PixelChip> PixelDecoder::decodeLane(uint8_t laneID, gsl::span<const uint8_t> laneWords)
{
  bool done = false;
  auto currentptr = laneWords.data();

  uint8_t currentChipID;
  uint8_t currentRegion;
  uint16_t currentChipStatus;
  std::vector<PixelHit> hits;
  std::vector<PixelChip> decodedChips;
  bool activeChip = false;
  while (!done) {
    if (!activeChip && (*currentptr == 0)) {
      // skip 0 outside active chip
      currentptr++;
      if (currentptr - laneWords.data() >= laneWords.size()) {
        done = true;
      }
      continue;
    }
    auto wordtype = getWordType(*currentptr);
    std::size_t wordsize;
    switch (wordtype) {
      case PixelWord::PixelWordType::CHIP_EMPTYFRAME: {
        auto emptyword = reinterpret_cast<const PixelWord::ChipHeader*>(currentptr);
        wordsize = sizeof(PixelWord::ChipHeader) / sizeof(uint8_t);
        hits.clear();
        auto chipID = emptyword->mChipID;
        // std::cout << "Empty chip (" << std::bitset<4>(emptyword->mIdentifier) << ") " << int(emptyword->mChipID) << ", BC (" << int(emptyword->mBunchCrossing) << "), empty " << (emptyword->isEmptyFrame() ? "yes" : "no") << std::endl;
        LOG(debug) << "Empty chip (" << std::bitset<4>(emptyword->mIdentifier) << ") " << int(emptyword->mChipID) << ", BC (" << int(emptyword->mBunchCrossing) << "), empty " << (emptyword->isEmptyFrame() ? "yes" : "no");
        if (std::find_if(decodedChips.begin(), decodedChips.end(), [chipID, laneID](const PixelChip& chip) { return chip.mChipID == chipID && chip.mLaneID == laneID; }) == decodedChips.end()) {
          // Add empty chip to decoded payload (if not yet present)
          // std::cout << "Creating new empty frame" << std::endl;
          LOG(debug) << "Creating new empty frame";
          currentChipStatus = (emptyword->mIdentifier) << 8;
          decodedChips.push_back({0, laneID, static_cast<uint8_t>(chipID), currentChipStatus, hits});
        } else {
          // std::cout << "Skipping existing empty frame" << std::endl;
          LOG(debug) << "Skipping existing empty frame";
        }
        break;
      }
      case PixelWord::PixelWordType::CHIP_HEADER: {
        auto chipheader = reinterpret_cast<const PixelWord::ChipHeader*>(currentptr);
        wordsize = sizeof(PixelWord::ChipHeader) / sizeof(uint8_t);
        hits.clear();
        currentChipID = chipheader->mChipID;
        activeChip = true;
        currentChipStatus = (chipheader->mIdentifier) << 8;
        // std::cout << "New chip (" << std::bitset<4>(chipheader->mIdentifier) << ") " << int(chipheader->mChipID) << ", BC (" << int(chipheader->mBunchCrossing) << "), empty " << (chipheader->isEmptyFrame() ? "yes" : "no") << std::endl;
        LOG(debug) << "New chip (" << std::bitset<4>(chipheader->mIdentifier) << ") " << int(chipheader->mChipID) << ", BC (" << int(chipheader->mBunchCrossing) << "), empty " << (chipheader->isEmptyFrame() ? "yes" : "no");
        break;
      }
      case PixelWord::PixelWordType::CHIP_TRAILER: {
        auto trailer = reinterpret_cast<const PixelWord::ChipTrailer*>(currentptr);
        wordsize = sizeof(PixelWord::ChipTrailer) / sizeof(uint8_t);
        currentChipStatus |= trailer->mReadoutFlags;
        // -> Combine hits to chip
        // -> Write hits to output container
        // std::cout << "Finished chip (" << std::bitset<4>(trailer->mIdentifier) << ") " << int(currentChipID) << " with " << hits.size() << " hits .. (Readout flags " << std::bitset<4>(trailer->mReadoutFlags) << ")" << std::endl;
        LOG(debug) << "Finished chip (" << std::bitset<4>(trailer->mIdentifier) << ") " << int(currentChipID) << " with " << hits.size() << " hits .. (Readout flags " << std::bitset<4>(trailer->mReadoutFlags) << ")";
        auto found = std::find_if(decodedChips.begin(), decodedChips.end(), [currentChipID, laneID](const PixelChip& chip) { return chip.mChipID == currentChipID && chip.mLaneID == laneID; });
        if (found != decodedChips.end()) {
          auto hitsbefore = found->mHits.size();
          std::copy(hits.begin(), hits.end(), std::back_inserter(found->mHits));
          found->mStatusCode = currentChipStatus;
          found->removeEmptyframe();
          // std::cout << "Merging data with existing chip, Hits before: " << hitsbefore << ", after: " << found->mHits.size() << std::endl;
          LOG(debug) << "Merging data with existing chip, Hits before: " << hitsbefore << ", after: " << found->mHits.size();
        } else {
          // std::cout << "Inserting new chip" << std::endl;
          LOG(debug) << "Inserting new chip";
          decodedChips.push_back({0, laneID, currentChipID, currentChipStatus, hits});
        }
        activeChip = false;
        break;
      }
      case PixelWord::PixelWordType::REGION_HEADER: {
        auto regionheader = reinterpret_cast<const PixelWord::RegionHeader*>(currentptr);
        wordsize = sizeof(PixelWord::RegionHeader) / sizeof(uint8_t);
        currentRegion = regionheader->mRegion;
        // std::cout << "New region (" << std::bitset<3>(regionheader->mIdentifier) << ") " << int(regionheader->mRegion) << std::endl;
        LOG(debug) << "New region (" << std::bitset<3>(regionheader->mIdentifier) << ") " << int(regionheader->mRegion);
        break;
      }
      case PixelWord::PixelWordType::DATA_SHORT: {
        PixelWord::DataShort datashort(currentptr);
        // std::cout << "Found DataShort [" << std::bitset<16>(datashort.mData) << "] word (" << std::bitset<2>(datashort.mIdentifier) << ") with encoder " << std::bitset<4>(datashort.mEncoderID) << " and address " << std::bitset<10>(datashort.mAddress) << std::endl;
        LOG(debug) << "Found DataShort [" << std::bitset<16>(datashort.mData) << "] word (" << std::bitset<2>(datashort.mIdentifier) << ") with encoder " << std::bitset<4>(datashort.mEncoderID) << " and address " << std::bitset<10>(datashort.mAddress);
        wordsize = sizeof(PixelWord::DataShort) / sizeof(uint8_t);
        hits.push_back({AlpideX(currentRegion, datashort.mEncoderID, datashort.mAddress), AlpideY(datashort.mAddress)});
        break;
      }
      case PixelWord::PixelWordType::DATA_LONG: {
        // Split word in 2 parts - DataShort and hitmap
        PixelWord::DataShort datapart(currentptr);
        auto hitmappart = reinterpret_cast<const PixelWord::Hitmap*>(currentptr + sizeof(PixelWord::DataShort));
        // std::cout << "Found DataLong [" << std::bitset<16>(datapart.mData) << "] word (" << std::bitset<2>(datapart.mIdentifier) << ") with encoder " << std::bitset<4>(datapart.mEncoderID) << " and address " << std::bitset<10>(datapart.mAddress) << std::endl;
        LOG(debug) << "Found DataLong [" << std::bitset<16>(datapart.mData) << "] word (" << std::bitset<2>(datapart.mIdentifier) << ") with encoder " << std::bitset<4>(datapart.mEncoderID) << " and address " << std::bitset<10>(datapart.mAddress);
        wordsize = (sizeof(PixelWord::DataShort) + (sizeof(PixelWord::Hitmap))) / sizeof(uint8_t);
        auto hitmap = hitmappart->getHitmap();
        for (int bitID = 0; bitID < hitmap.size(); bitID++) {
          if (hitmap.test(bitID)) {
            auto address = datapart.mAddress + bitID + 1;
            hits.push_back({AlpideX(currentRegion, datapart.mEncoderID, address), AlpideY(address)});
          }
        }
        break;
      }
      case PixelWord::PixelWordType::BUSY_OFF:
        // std::cout << "Found busy off" << std::endl;
        wordsize = sizeof(PixelWord::BusyOff) / sizeof(uint8_t);
      case PixelWord::PixelWordType::BUSY_ON:
        // std::cout << "Found busy on" << std::endl;
        wordsize = sizeof(PixelWord::BusyOn) / sizeof(uint8_t);
      case PixelWord::PixelWordType::IDLE:
        // std::cout << "Found idle" << std::endl;
        wordsize = sizeof(PixelWord::Idle) / sizeof(uint8_t);
      default:
        // std::cout << "Found unknown word" << std::endl;
        wordsize = 1;
        break;
    };
    currentptr += wordsize;
    if (currentptr - laneWords.data() >= laneWords.size()) {
      done = true;
    }
  }
  // std::cout << "Lane " << laneID << ": Found " << decodedChips.size() << " chips ... " << std::endl;
  LOG(debug) << "Lane " << laneID << ": Found " << decodedChips.size() << " chips ... ";
  return decodedChips;
}
