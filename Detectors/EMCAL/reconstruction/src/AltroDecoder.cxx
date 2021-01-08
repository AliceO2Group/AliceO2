// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <cstring>
#include <boost/format.hpp>
#include "InfoLogger/InfoLogger.hxx"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/RawReaderMemory.h"

using namespace o2::emcal;

AltroDecoder::AltroDecoder(RawReaderMemory& reader) : mRawReader(reader),
                                                      mRCUTrailer(),
                                                      mChannels(),
                                                      mChannelsInitialized(false)
{
}

void AltroDecoder::decode()
{
  readRCUTrailer();
  checkRCUTrailer();
  readChannels();
}

void AltroDecoder::readRCUTrailer()
{
  try {
    auto payloadwordsOrig = mRawReader.getPayload().getPayloadWords();
    gsl::span<const uint32_t> payloadwords(payloadwordsOrig.data(), payloadwordsOrig.size());
    mRCUTrailer.constructFromRawPayload(payloadwords);
  } catch (RCUTrailer::Error& e) {
    AliceO2::InfoLogger::InfoLogger logger;
    logger << e.what();
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR, (boost::format("RCU trailer decoding error: %s") % e.what()).str().data());
  }
}

void AltroDecoder::checkRCUTrailer()
{
}

void AltroDecoder::readChannels()
{
  mChannelsInitialized = false;
  mChannels.clear();
  int currentpos = 0;
  auto& buffer = mRawReader.getPayload().getPayloadWords();
  while (currentpos < buffer.size() - mRCUTrailer.getTrailerSize()) {
    auto currentword = buffer[currentpos++];
    if (currentword >> 30 != 1) {
      continue;
    }
    // starting a new channel
    mChannels.emplace_back(currentword & 0xFFF, (currentword >> 16) & 0x3FF);
    auto& currentchannel = mChannels.back();
    currentchannel.setBadChannel((currentword >> 29) & 0x1);

    /// decode all words for channel
    int numberofwords = (currentchannel.getPayloadSize() + 2) / 3;
    std::vector<uint16_t> bunchwords;
    for (int iword = 0; iword < numberofwords; iword++) {
      currentword = buffer[currentpos++];
      if ((currentword >> 30) != 0) {
        // AliceO2::InfoLogger::InfoLogger logger;
        // logger << "Unexpected end of payload in altro channel payload! DDL=" << std::setw(3) << std::setfill(0) << mRawReader.getRawHeader().getLink()
        //       << ", Address=0x" << std::hex << current.getHardwareAddress() << ", word=0x" << currentword << std::dec;
        currentpos--;
        continue;
      }
      bunchwords.push_back((currentword >> 20) & 0x3FF);
      bunchwords.push_back((currentword >> 10) & 0x3FF);
      bunchwords.push_back(currentword & 0x3FF);
    }

    // decode bunches
    int currentsample = 0;
    while (currentsample < currentchannel.getPayloadSize() && bunchwords.size() > currentsample + 2) {
      int bunchlength = bunchwords[currentsample] - 2, // remove words for bunchlength and starttime
        starttime = bunchwords[currentsample + 1];
      auto& currentbunch = currentchannel.createBunch(bunchlength, starttime);
      currentbunch.initFromRange(gsl::span<uint16_t>(&bunchwords[currentsample + 2], std::min((unsigned long)bunchlength, bunchwords.size() - currentsample - 2)));
      currentsample += bunchlength + 2;
    }
  }
  mChannelsInitialized = true;
}

const RCUTrailer& AltroDecoder::getRCUTrailer() const
{
  if (!mRCUTrailer.isInitialized()) {
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR, "RCU trailer was not initialized");
  }
  return mRCUTrailer;
}

const std::vector<Channel>& AltroDecoder::getChannels() const
{
  if (!mChannelsInitialized) {
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::CHANNEL_ERROR, "Channels not initizalized");
  }
  return mChannels;
}