// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  mRawWriter->setApplyCarryOverToLastPage(true);
  for (auto iddl = 0; iddl < 40; iddl++) {
    // For EMCAL set
    // - FEE ID = DDL ID
    // - C-RORC and link increasing with DDL ID
    // @TODO replace with link assignment on production FLPs,
    // eventually storing in CCDB
    auto [crorc, link] = getLinkAssignment(iddl);
    std::string rawfilename = mOutputLocation;
    switch (mFileFor) {
      case FileFor_t::kFullDet:
        rawfilename += "/emcal.raw";
        break;
      case FileFor_t::kSubDet: {
        std::string detstring;
        if (iddl < 22) {
          detstring = "emcal";
        } else {
          detstring = "dcal";
        }
        rawfilename += fmt::format("/{:s}.raw", detstring.data());
        break;
      };
      case FileFor_t::kLink:
        rawfilename += fmt::format("/emcal_{:d}_{:d}.raw", crorc, link);
    }
    mRawWriter->registerLink(iddl, crorc, link, 0, rawfilename.data());
  }
  // initialize mappers
  if (!mMappingHandler) {
    mMappingHandler = std::make_unique<o2::emcal::MappingHandler>();
  }

  // initialize containers for SRU
  for (auto isru = 0; isru < 40; isru++) {
    SRUDigitContainer srucont;
    srucont.mSRUid = isru;
    mSRUdata.push_back(srucont);
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
  for (auto srucont : mSRUdata) {
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
    }

    // Get time sample of the digit:
    // Digitizer stores the time sample in ns, needs to be converted to time sample dividing
    // by the length of the time sample
    auto timesample = int(dig.getTimeStamp() / emcal::constants::EMCAL_TIMESAMPLE);
    if (timesample >= mNADCSamples) {
      LOG(ERROR) << "Digit time sample " << timesample << " outside range [0," << mNADCSamples << "]";
      continue;
    }
    (*bunchDigits)[timesample] = &dig;
  }

  // Create and fill DMA pages for each channel
  std::cout << "encode data" << std::endl;
  std::vector<char> payload;
  for (auto srucont : mSRUdata) {

    for (const auto& [tower, channel] : srucont.mChannels) {
      // Find out hardware address of the channel
      auto hwaddress = mMappingHandler->getMappingForDDL(srucont.mSRUid).getHardwareAddress(channel.mRow, channel.mCol, ChannelType_t::HIGH_GAIN); // @TODO distinguish between high- and low-gain cells

      std::vector<int> rawbunches;
      for (auto& bunch : findBunches(channel.mDigits)) {
        rawbunches.push_back(bunch.mADCs.size() + 2); // add 2 words for header information
        rawbunches.push_back(bunch.mStarttime);
        for (auto adc : bunch.mADCs) {
          rawbunches.push_back(adc);
        }
      }
      if (!rawbunches.size()) {
        continue;
      }
      auto encodedbunches = encodeBunchData(rawbunches);
      auto chanhead = createChannelHeader(hwaddress, rawbunches.size(), false); /// bad channel status eventually to be added later
      char* chanheadwords = reinterpret_cast<char*>(&chanhead);
      uint32_t* testheader = reinterpret_cast<uint32_t*>(chanheadwords);
      if ((*testheader >> 30) & 1) {
        // header pattern found, check that the payload size is properly reflecting the number of words
        uint32_t payloadsizeRead = ((*testheader >> 16) & 0x3FF);
        uint32_t nwordsRead = (payloadsizeRead + 2) / 3;
        if (encodedbunches.size() != nwordsRead) {
          LOG(ERROR) << "Mismatch in number of 32-bit words, encoded " << encodedbunches.size() << ", recalculated " << nwordsRead << std::endl;
          LOG(ERROR) << "Payload size: " << payloadsizeRead << ", number of words: " << rawbunches.size() << ", encodeed words " << encodedbunches.size() << ", calculated words " << nwordsRead << std::endl;
        }
      } else {
        LOG(ERROR) << "Header without header bit detected ..." << std::endl;
      }
      for (int iword = 0; iword < sizeof(ChannelHeader) / sizeof(char); iword++) {
        payload.emplace_back(chanheadwords[iword]);
      }
      char* channelwords = reinterpret_cast<char*>(encodedbunches.data());
      for (auto iword = 0; iword < encodedbunches.size() * sizeof(int) / sizeof(char); iword++) {
        payload.emplace_back(channelwords[iword]);
      }
    }

    if (!payload.size()) {
      continue;
    }

    // Create RCU trailer
    auto trailerwords = createRCUTrailer(payload.size() / 4, 16, 16, 100., 0.);
    for (auto word : trailerwords) {
      payload.emplace_back(word);
    }

    // register output data
    auto ddlid = srucont.mSRUid;
    auto [crorc, link] = getLinkAssignment(ddlid);
    LOG(DEBUG1) << "Adding payload with size " << payload.size() << " (" << payload.size() / 4 << " ALTRO words)";
    mRawWriter->addData(ddlid, crorc, link, 0, trg.getBCData(), payload, false, trg.getTriggerBits());
  }
  std::cout << "Done" << std::endl;
  return true;
}

std::vector<AltroBunch> RawWriter::findBunches(const std::vector<o2::emcal::Digit*>& channelDigits)
{
  std::vector<AltroBunch> result;
  AltroBunch* currentBunch = nullptr;
  // Digits in ALTRO bunch in time-reversed order
  int itime;
  for (itime = channelDigits.size() - 1; itime >= 0; itime--) {
    auto dig = channelDigits[itime];
    if (!dig) {
      continue;
    }
    int adc = dig->getAmplitudeADC();
    if (adc < mPedestal) {
      // Stop bunch
      // Set the start time to the time sample of previous (passing) digit
      currentBunch->mStarttime = itime + 1;
      currentBunch = nullptr;
      continue;
    }
    if (!currentBunch) {
      // start new bunch
      AltroBunch bunch;
      result.push_back(bunch);
      currentBunch = &(result.back());
    }
    currentBunch->mADCs.emplace_back(adc);
  }
  // if we have a last bunch set time start time to the time bin of teh previous digit
  if (currentBunch) {
    currentBunch->mStarttime = itime + 1;
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

  int ddlInSupermoudel = -1;
  if (0 <= row && row < 8) {
    ddlInSupermoudel = 0; // first cable row
  } else if (8 <= row && row < 16 && 0 <= col && col < 24) {
    ddlInSupermoudel = 0; // first half;
  } else if (8 <= row && row < 16 && 24 <= col && col < 48) {
    ddlInSupermoudel = 1; // second half;
  } else if (16 <= row && row < 24) {
    ddlInSupermoudel = 1; // third cable row
  }
  if (supermoduleID % 2 == 1) {
    ddlInSupermoudel = 1 - ddlInSupermoudel; // swap for odd=C side, to allow us to cable both sides the same
  }

  return std::make_tuple(supermoduleID * 2 + ddlInSupermoudel, row, col);
}

std::tuple<int, int> RawWriter::getLinkAssignment(int ddlID)
{
  // Temporary link assignment (till final link assignment is known -
  // eventually taken from CCDB)
  // - Link (0-5) and C-RORC ID linear with ddlID
  return std::make_tuple(ddlID / 6, ddlID % 6);
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

std::vector<char> RawWriter::createRCUTrailer(int payloadsize, int feca, int fecb, double timesample, double l1phase)
{
  RCUTrailer trailer;
  trailer.setActiveFECsA(feca);
  trailer.setActiveFECsB(fecb);
  trailer.setPayloadSize(payloadsize);
  trailer.setL1Phase(l1phase);
  trailer.setTimeSample(timesample);
  auto trailerwords = trailer.encode();
  std::vector<char> encoded(trailerwords.size() * sizeof(uint32_t));
  memcpy(encoded.data(), trailerwords.data(), trailerwords.size() * sizeof(uint32_t));
  return encoded;
}

int RawWriter::carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                               const char* ptr, int maxSize, int splitID,
                               std::vector<char>& trailer, std::vector<char>& header) const
{
  int offs = ptr - &data[0]; // offset wrt the head of the payload
  // make sure ptr and end of the suggested block are within the payload
  assert(offs >= 0 && size_t(offs + maxSize) <= data.size());

  // Read trailer template from the end of payload
  gsl::span<const uint32_t> payloadwords(reinterpret_cast<const uint32_t*>(data.data()), data.size() / sizeof(uint32_t));
  auto rcutrailer = RCUTrailer::constructFromPayloadWords(payloadwords);

  int sizeNoTrailer = maxSize - rcutrailer.getTrailerSize() * sizeof(uint32_t);
  // calculate payload size for RCU trailer:
  // assume actualsize is in byte
  // Payload size is defined as the number of 32-bit payload words
  // -> actualSize to be converted to size of 32 bit words
  auto payloadsize = sizeNoTrailer / sizeof(uint32_t);
  rcutrailer.setPayloadSize(payloadsize);
  auto trailerwords = rcutrailer.encode();
  trailer.resize(trailerwords.size() * sizeof(uint32_t));
  memcpy(trailer.data(), trailerwords.data(), trailer.size());
  // Size to return differs between intermediate pages and last page
  // - intermediate page: Size of the trailer needs to be removed as the trailer gets appended
  // - last page: Size of the trailer needs to be included as the trailer gets replaced
  int bytesLeft = data.size() - (ptr - &data[0]);
  bool lastPage = bytesLeft <= maxSize;
  int actualSize = maxSize;
  if (!lastPage) {
    actualSize = sizeNoTrailer;
  }
  return actualSize;
}
