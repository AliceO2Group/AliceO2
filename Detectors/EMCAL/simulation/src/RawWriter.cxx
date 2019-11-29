// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <gsl/span>
#include <TSystem.h>
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALBase/Geometry.h"
#include "EMCALSimulation/RawWriter.h"
#include "Headers/RAWDataHeader.h"

using namespace o2::emcal;

void RawWriter::setTriggerRecords(std::vector<o2::emcal::TriggerRecord>* triggers)
{
  mTriggers = triggers;
  mCurrentTrigger = triggers->begin();
}

void RawWriter::init()
{
  mOutputStream.open();

  // initialize mappers
  std::array<char, 4> sides = {{'A', 'C'}};
  for (auto iside = 0; iside < sides.size(); iside++) {
    for (auto isru = 0; isru < 20; isru++) {
      mMappers[iside * 2 + isru].setMapping(Form("%s/share/Detectors/EMC/file/RCU%d%c.data", gSystem->Getenv("O2_ROOT"), isru, sides[iside]));
    }
  }

  // initialize containers for SRU
  for (auto isru = 0; isru < 40; isru++) {
    SRUDigitContainer srucont;
    srucont.mSRUid = isru;
    mSRUdata.push_back(srucont);
  }
}

void RawWriter::processNextTrigger()
{
  for (auto srucont : mSRUdata)
    srucont.mChannels.clear();
  std::vector<o2::emcal::Digit*>* bunchDigits;
  int lasttower = -1;
  for (auto& dig : gsl::span(&mDigits->data()[mCurrentTrigger->getFirstEntry()], mCurrentTrigger->getNumberOfObjects())) {
    auto tower = dig.getTower();
    if (tower != lasttower) {
      lasttower = tower;
      auto onlineindices = getOnlineID(tower);
      int sruID = std::get<0>(onlineindices);
      auto towerdata = mSRUdata[sruID].mChannels.find(tower);
      if (towerdata == mSRUdata[sruID].mChannels.end()) {
        mSRUdata[sruID].mChannels[tower] = {std::get<1>(onlineindices), std::get<2>(onlineindices), std::vector<o2::emcal::Digit*>(mNADCSamples)};
        bunchDigits = &(mSRUdata[sruID].mChannels[tower].mDigits);
        memset(bunchDigits->data(), 0, sizeof(o2::emcal::Digit*) * mNADCSamples);
      } else {
        bunchDigits = &(towerdata->second.mDigits);
      }
      (*bunchDigits)[int(dig.getTimeStamp())] = &dig;
    }
  }

  // Create and fill DMA pages for each channel
  std::vector<char> payload;
  for (auto srucont : mSRUdata) {
    o2::header::RAWDataHeaderV4 rawheader;
    rawheader.triggerBC = mCurrentTrigger->getBCData().bc;
    rawheader.triggerOrbit = mCurrentTrigger->getBCData().orbit;
    // @TODO: Set trigger type
    rawheader.feeId = srucont.mSRUid;

    for (const auto& [tower, channel] : srucont.mChannels) {
      // Find out hardware address of the channel
      auto hwaddress = mMappers[srucont.mSRUid].getHardwareAddress(channel.mRow, channel.mCol, ChannelType_t::HIGH_GAIN); // @TODO distinguish between high- and low-gain cells

      std::vector<int> rawbunches;
      for (auto& bunch : findBunches(channel.mDigits)) {
        rawbunches.push_back(bunch.mStarttime);
        rawbunches.push_back(bunch.mADCs.size());
        for (auto adc : bunch.mADCs) {
          rawbunches.push_back(adc);
        }
      }
      auto encodedbunches = encodeBunchData(rawbunches);
      auto chanhead = createChannelHeader(hwaddress, encodedbunches.size() * 3 - 2, false); /// bad channel status eventually to be added later
      char* chanheadwords = reinterpret_cast<char*>(&chanhead);
      for (int iword = 0; iword < sizeof(ChannelHeader) / sizeof(char); iword++) {
        payload.emplace_back(chanheadwords[iword]);
      }
      char* channelwords = reinterpret_cast<char*>(encodedbunches.data());
      for (auto iword = 0; iword < encodedbunches.size() * sizeof(int) / sizeof(char); iword++) {
        payload.emplace_back(channelwords[iword]);
      }
    }

    // Create RCU trailer
    auto trailerwords = createRCUTrailer();
    for (auto word : trailerwords)
      payload.emplace_back(word);

    // write DMA page to stream
    mOutputStream.writeData(rawheader, payload);
  }
}

std::vector<AltroBunch> RawWriter::findBunches(const std::vector<o2::emcal::Digit*>& channelDigits)
{
  std::vector<AltroBunch> result;
  AltroBunch* currentBunch = nullptr;
  int starttime = 0;
  for (auto ien = channelDigits.size() - 1;; ien--) {
    auto dig = channelDigits[ien];
    if (!dig) {
      starttime++;
      continue;
    }
    int adc = dig->getEnergy() / constants::EMCAL_ADCENERGY; /// conversion Energy <-> ADC := 16 MeV/ADC
    if (adc < mPedestal) {
      // Stop bunch
      currentBunch = nullptr;
      starttime++;
      continue;
    }
    if (!currentBunch) {
      // start new bunch
      AltroBunch bunch;
      bunch.mStarttime = starttime;
      result.push_back(bunch);
      currentBunch = &(result.back());
    }
    currentBunch->mADCs.emplace_back(adc);
    starttime++;
  }
  return result;
}

std::tuple<int, int, int> RawWriter::getOnlineID(int towerID)
{
  auto cellindex = mGeometry->GetCellIndex(towerID);
  auto supermoduleID = std::get<0>(cellindex);
  auto etaphi = mGeometry->GetCellPhiEtaIndexInSModule(supermoduleID, std::get<1>(cellindex), std::get<2>(cellindex), std::get<3>(cellindex));
  auto etaphishift = mGeometry->ShiftOfflineToOnlineCellIndexes(supermoduleID, std::get<0>(etaphi), std::get<1>(etaphi));
  int row = std::get<0>(etaphishift), col = std::get<1>(etaphishift);

  int sruID = -1;
  if (0 <= row && row < 8)
    sruID = 0; // first cable row
  else if (8 <= row && row < 16 && 0 <= col && col < 24)
    sruID = 0; // first half;
  else if (8 <= row && row < 16 && 24 <= col && col < 48)
    sruID = 1; // second half;
  else if (16 <= row && row < 24)
    sruID = 1; // third cable row
  if (supermoduleID % 2 == 1)
    sruID = 1 - sruID; // swap for odd=C side, to allow us to cable both sides the same

  return std::make_tuple(sruID, row, col);
}

std::vector<int> RawWriter::encodeBunchData(const std::vector<int>& data)
{
  std::vector<int> encoded;
  CaloBunchWord currentword;
  int wordnumber = 0;
  for (auto adc : data) {
    switch (wordnumber) {
      case 0:
        currentword.mWord0 = adc;
        break;
      case 1:
        currentword.mWord1 = adc;
        break;
      case 2:
        currentword.mWord2 = adc;
        break;
    };
    if (wordnumber == 2) {
      // start new word;
      encoded.push_back(currentword.mDataWord);
      currentword.mDataWord = 0;
      wordnumber = 0;
    } else {
      wordnumber++;
    }
  }
  return encoded;
}

ChannelHeader RawWriter::createChannelHeader(int hardwareAddress, int payloadSize, bool isBadChannel)
{
  ChannelHeader header;
  header.mHardwareAddress = hardwareAddress;
  header.mPayloadSize = payloadSize;
  header.mBadChannel = isBadChannel ? 1 : 0;
  return header;
}

std::vector<char> RawWriter::createRCUTrailer()
{
  std::vector<char> trailerwords;
  return trailerwords;
}