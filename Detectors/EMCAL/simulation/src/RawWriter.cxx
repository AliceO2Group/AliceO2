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

#include "FairLogger.h"

#include <fmt/core.h>
#include <gsl/span>
#include <TSystem.h>
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/RCUTrailer.h"
#include "EMCALSimulation/RawWriter.h"

using namespace o2::emcal;

void RawWriter::init()
{
  mRawWriter = std::make_unique<o2::raw::RawFileWriter>(o2::header::gDataOriginEMC, false);
  mRawWriter->setCarryOverCallBack(this);

  // initialize mappers
  if (!mMappingHandler) {
    mMappingHandler = std::make_unique<o2::emcal::MappingHandler>();
  }

  for (auto iddl = 0; iddl < 40; iddl++) {
    // For EMCAL set
    // - FEE ID = DDL ID
    // - C-RORC and link increasing with DDL ID
    // @TODO replace with link assignment on production FLPs,
    // eventually storing in CCDB

    // initialize containers for SRU
    SRUDigitContainer srucont;
    srucont.mSRUid = iddl;
    mSRUdata.push_back(srucont);

    // Skip empty links with these ddl IDs,
    // ddl ID 21 and 39 are empty links, while 23 and 36 are connected to LEDmon only
    if (iddl == 21 || iddl == 22 || iddl == 36 || iddl == 39) {
      continue;
    }

    auto [crorc, link] = mGeometry->getLinkAssignment(iddl);
    auto flpID = (iddl <= 23) ? 146 : 147;
    std::string rawfilename = mOutputLocation;
    switch (mFileFor) {
      case FileFor_t::kFullDet:
        rawfilename += "/emcal.raw";
        break;
      case FileFor_t::kSubDet:
        rawfilename += fmt::format("/EMC_alio2-cr1-flp{:d}.raw", flpID);
        break;
      case FileFor_t::kCRORC:
        rawfilename += fmt::format("/EMC_alio2-cr1-flp{:d}_crorc{:d}.raw", flpID, crorc);
        break;
      case FileFor_t::kLink:
        // Pileup simulation based on DigitsWriteoutBuffer (EMCAL-681) - AliceO2 – H. Hassan
        rawfilename += fmt::format("/EMC_alio2-cr1-flp{:d}_crorc{:d}_{:d}.raw", flpID, crorc, link);
        break;
    }
    mRawWriter->registerLink(iddl, crorc, link, 0, rawfilename.data());
  }
}

void RawWriter::digitsToRaw(gsl::span<o2::emcal::Digit> digitsbranch, gsl::span<o2::emcal::TriggerRecord> triggerbranch)
{
  setDigits(digitsbranch);
  for (auto trg : triggerbranch) {
    processTrigger(trg);
  }
}

bool RawWriter::processTrigger(const o2::emcal::TriggerRecord& trg)
{
  for (auto& srucont : mSRUdata) {
    srucont.mChannels.clear();
  }
  std::vector<o2::emcal::Digit*>* bunchDigits;
  int lasttower = -1;
  for (auto& dig : gsl::span(mDigits.data() + trg.getFirstEntry(), trg.getNumberOfObjects())) {
    auto tower = dig.getTower();
    if (tower != lasttower) {
      lasttower = tower;
      if (tower > 20000) {
        std::cout << "Wrong cell ID " << tower << std::endl;
      }
      auto onlineindices = mGeometry->getOnlineID(tower);
      int sruID = std::get<0>(onlineindices);
      auto towerdata = mSRUdata[sruID].mChannels.find(tower);
      if (towerdata == mSRUdata[sruID].mChannels.end()) {
        mSRUdata[sruID].mChannels[tower] = {std::get<1>(onlineindices), std::get<2>(onlineindices), std::vector<o2::emcal::Digit*>(mNADCSamples)};
        bunchDigits = &(mSRUdata[sruID].mChannels[tower].mDigits);
        memset(bunchDigits->data(), 0, sizeof(o2::emcal::Digit*) * mNADCSamples);
      } else {
        bunchDigits = &(towerdata->second.mDigits);
      }
    }

    // Get time sample of the digit:
    // Digitizer stores the time sample in ns, needs to be converted to time sample dividing
    // by the length of the time sample
    auto timesample = int(dig.getTimeStamp() / emcal::constants::EMCAL_TIMESAMPLE);
    if (timesample >= mNADCSamples) {
      LOG(error) << "Digit time sample " << timesample << " outside range [0," << mNADCSamples << "]";
      continue;
    }
    (*bunchDigits)[timesample] = &dig;
  }

  // Create and fill DMA pages for each channel
  LOG(debug) << "encode data";
  for (auto srucont : mSRUdata) {

    std::vector<char> payload; // this must be initialized per SRU, becuase pages are per SRU, therefore the payload was not reset.

    if (srucont.mSRUid == 21 || srucont.mSRUid == 22 || srucont.mSRUid == 36 || srucont.mSRUid == 39) {
      continue;
    }

    for (const auto& [tower, channel] : srucont.mChannels) {

      bool saturatedBunchHG = false;
      createPayload(channel, ChannelType_t::HIGH_GAIN, srucont.mSRUid, payload, saturatedBunchHG);
      if (saturatedBunchHG) {
        createPayload(channel, ChannelType_t::LOW_GAIN, srucont.mSRUid, payload, saturatedBunchHG);
      }
    }

    if (!payload.size()) {
      // [EMCAL-699] No payload found in SRU
      // Still the link is not completely ignored but a trailer with 0-payloadsize is added
      LOG(debug) << "Payload buffer has size 0 - only write empty trailer" << std::endl;
    }
    LOG(debug) << "Payload buffer has size " << payload.size();

    // Create RCU trailer
    auto trailerwords = createRCUTrailer(payload.size() / sizeof(uint32_t), 100., trg.getBCData().toLong(), srucont.mSRUid);
    for (auto word : trailerwords) {
      payload.emplace_back(word);
    }

    // register output data
    auto ddlid = srucont.mSRUid;
    auto [crorc, link] = mGeometry->getLinkAssignment(ddlid);
    LOG(debug1) << "Adding payload with size " << payload.size() << " (" << payload.size() / 4 << " ALTRO words)";
    mRawWriter->addData(ddlid, crorc, link, 0, trg.getBCData(), payload, false, trg.getTriggerBits());
  }
  LOG(debug) << "Done";
  return true;
}

void RawWriter::createPayload(o2::emcal::ChannelData channel, o2::emcal::ChannelType_t chanType, int ddlID, std::vector<char>& payload, bool& saturatedBunch)
{
  // Find out hardware address of the channel
  auto hwaddress = mMappingHandler->getMappingForDDL(ddlID).getHardwareAddress(channel.mRow, channel.mCol, chanType); // @TODO distinguish between high- and low-gain cells

  std::vector<int> rawbunches;
  int nbunches = 0;

  // Creating the high gain bunch
  for (auto& bunch : findBunches(channel.mDigits, chanType)) {
    if (!bunch.mADCs.size()) {
      LOG(error) << "Found bunch with without ADC entries - skipping ...";
      continue;
    }
    rawbunches.push_back(bunch.mADCs.size() + 2); // add 2 words for header information
    rawbunches.push_back(bunch.mStarttime);
    for (auto adc : bunch.mADCs) {
      rawbunches.push_back(adc);
      if (adc > o2::emcal::constants::LG_SUPPRESSION_CUT) {
        saturatedBunch = true;
      }
    }
    nbunches++;
  }

  if (!rawbunches.size()) {
    LOG(debug) << "No bunch selected";
    return;
  }
  LOG(debug) << "Selected " << nbunches << " bunches";

  auto encodedbunches = encodeBunchData(rawbunches);
  auto chanhead = createChannelHeader(hwaddress, rawbunches.size(), false); /// bad channel status eventually to be added later
  char* chanheadwords = reinterpret_cast<char*>(&chanhead);
  uint32_t* testheader = reinterpret_cast<uint32_t*>(chanheadwords);
  if ((*testheader >> 30) & 1) {
    // header pattern found, check that the payload size is properly reflecting the number of words
    uint32_t payloadsizeRead = ((*testheader >> 16) & 0x3FF);
    uint32_t nwordsRead = (payloadsizeRead + 2) / 3;
    if (encodedbunches.size() != nwordsRead) {
      LOG(error) << "Mismatch in number of 32-bit words, encoded " << encodedbunches.size() << ", recalculated " << nwordsRead << std::endl;
      LOG(error) << "Payload size: " << payloadsizeRead << ", number of words: " << rawbunches.size() << ", encodeed words " << encodedbunches.size() << ", calculated words " << nwordsRead << std::endl;
    } else {
      LOG(debug) << "Matching number of payload 32-bit words, encoded " << encodedbunches.size() << ", decoded " << nwordsRead;
    }
  } else {
    LOG(error) << "Header without header bit detected ..." << std::endl;
  }
  for (int iword = 0; iword < sizeof(ChannelHeader) / sizeof(char); iword++) {
    payload.emplace_back(chanheadwords[iword]);
  }
  char* channelwords = reinterpret_cast<char*>(encodedbunches.data());
  for (auto iword = 0; iword < encodedbunches.size() * sizeof(int) / sizeof(char); iword++) {
    payload.emplace_back(channelwords[iword]);
  }
}

std::vector<AltroBunch> RawWriter::findBunches(const std::vector<o2::emcal::Digit*>& channelDigits, ChannelType_t channelType)
{
  std::vector<AltroBunch> result;
  AltroBunch currentBunch;
  bool bunchStarted = false;
  // Digits in ALTRO bunch in time-reversed order
  int itime;
  for (itime = channelDigits.size() - 1; itime >= 0; itime--) {
    auto dig = channelDigits[itime];
    if (!dig) {
      if (bunchStarted) {
        // we have a bunch which is started and needs to be closed
        // check if the ALTRO bunch has a minimum amount of ADCs
        if (currentBunch.mADCs.size() >= mMinADCBunch) {
          // Bunch selected, set start time and push to bunches
          result.push_back(currentBunch);
          currentBunch = AltroBunch();
          bunchStarted = false;
        }
      }
      continue;
    }
    int adc = dig->getAmplitudeADC(channelType);
    if (adc < mPedestal) {
      // ADC value below threshold
      // in case we have an open bunch it needs to be stopped bunch
      // Set the start time to the time sample of previous (passing) digit
      if (bunchStarted) {
        // check if the ALTRO bunch has a minimum amount of ADCs
        if (currentBunch.mADCs.size() >= mMinADCBunch) {
          // Bunch selected, set start time and push to bunches
          result.push_back(currentBunch);
          currentBunch = AltroBunch();
          bunchStarted = false;
        }
      }
    }
    // Valid ADC value, if the bunch is closed we start a new bunch
    if (!bunchStarted) {
      bunchStarted = true;
      currentBunch.mStarttime = itime;
    }
    currentBunch.mADCs.emplace_back(adc);
  }
  // if we have a last bunch set time start time to the time bin of teh previous digit
  if (bunchStarted) {
    if (currentBunch.mADCs.size() >= mMinADCBunch) {
      result.push_back(currentBunch);
    }
  }
  return result;
}

std::vector<int> RawWriter::encodeBunchData(const std::vector<int>& data)
{
  std::vector<int> encoded;
  CaloBunchWord currentword;
  currentword.mDataWord = 0;
  int wordnumber = 0;
  for (auto adc : data) {
    if (adc > 0x3FF) {
      LOG(error) << "Exceeding max ADC count for 10 bit ALTRO word: " << adc << " (max: 1023)" << std::endl;
    }
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
    wordnumber++;
    if (wordnumber == 3) {
      // start new word;
      encoded.push_back(currentword.mDataWord);
      currentword.mDataWord = 0;
      wordnumber = 0;
    }
  }
  if (wordnumber) {
    encoded.push_back(currentword.mDataWord);
  }
  return encoded;
}

ChannelHeader RawWriter::createChannelHeader(int hardwareAddress, int payloadSize, bool isBadChannel)
{
  ChannelHeader header;
  header.mHardwareAddress = hardwareAddress;
  header.mPayloadSize = payloadSize;
  header.mBadChannel = isBadChannel ? 1 : 0;
  header.mHeaderBits = 1;
  return header;
}

std::vector<char> RawWriter::createRCUTrailer(int payloadsize, double timesample, uint64_t triggertime, int feeID)
{
  RCUTrailer trailer;
  trailer.setPayloadSize(payloadsize);
  trailer.setTimeSamplePhaseNS(triggertime, timesample);

  // You can find details about these settings here https://alice.its.cern.ch/jira/browse/EMCAL-650
  trailer.setRCUID(feeID);
  trailer.setFirmwareVersion(2);
  trailer.setActiveFECsA(0x0);
  trailer.setActiveFECsB(0x1);
  trailer.setBaselineCorrection(0);
  trailer.setPolarity(false);
  trailer.setNumberOfPresamples(0);
  trailer.setNumberOfPostsamples(0);
  trailer.setSecondBaselineCorrection(false);
  trailer.setGlitchFilter(0);
  trailer.setNumberOfNonZeroSuppressedPostsamples(1);
  trailer.setNumberOfNonZeroSuppressedPresamples(1);
  trailer.setNumberOfPretriggerSamples(0);
  trailer.setNumberOfSamplesPerChannel(15);
  // For MC we don't simulate pedestals. In order to prevent pedestal subtraction
  // in the raw fitter we set the zero suppression to true in the RCU trailer
  trailer.setZeroSuppression(true);
  trailer.setSparseReadout(true);
  trailer.setNumberOfAltroBuffers(RCUTrailer::BufferMode_t::NBUFFERS4);

  auto trailerwords = trailer.encode();
  std::vector<char> encoded(trailerwords.size() * sizeof(uint32_t));
  memcpy(encoded.data(), trailerwords.data(), trailerwords.size() * sizeof(uint32_t));
  return encoded;
}

int RawWriter::carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                               const char* ptr, int maxSize, int splitID,
                               std::vector<char>& trailer, std::vector<char>& header) const
{
  constexpr int TrailerSize = 9 * sizeof(uint32_t);
  int bytesLeft = data.data() + data.size() - ptr;
  int leftAfterSplit = bytesLeft - maxSize;

  if (leftAfterSplit < TrailerSize) {
    return std::max(0, bytesLeft - TrailerSize);
  }

  return maxSize;
}
