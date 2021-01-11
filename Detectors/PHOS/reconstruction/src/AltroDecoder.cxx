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
#include "PHOSReconstruction/AltroDecoder.h"
#include "PHOSReconstruction/RawReaderMemory.h"
#include "PHOSReconstruction/RawDecodingError.h"

#include "DetectorsRaw/RDHUtils.h"
#include <FairLogger.h>

using namespace o2::phos;

AltroDecoder::AltroDecoder(RawReaderMemory& reader) : mRawReader(reader),
                                                      mRCUTrailer(),
                                                      mChannels(),
                                                      mChannelsInitialized(false)
{
}

AltroDecoderError::ErrorType_t AltroDecoder::decode()
{
  try {
    readRCUTrailer();
  } catch (RCUTrailer::Error& e) {
    LOG(ERROR) << "RCU trailer error" << (int)e.getErrorType();
    return AltroDecoderError::RCU_TRAILER_ERROR;
  }
  //TODO  checkRCUTrailer();
  try {
    readChannels();
  } catch (AltroDecoderError::ErrorType_t e) {
    LOG(ERROR) << "Altro decoding error " << e;
    return e;
  }
  return AltroDecoderError::kOK;
}

void AltroDecoder::readRCUTrailer()
{
  try {
    auto payloadwords = mRawReader.getPayload().getPayloadWords();
    gsl::span<const uint32_t> tmp(payloadwords.data(), payloadwords.size());
    mRCUTrailer.constructFromRawPayload(tmp);
  } catch (RCUTrailer::Error& e) {
    throw e;
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

  int payloadend = mRCUTrailer.getPayloadSize();
  while (currentpos < payloadend) {
    auto currentword = buffer[currentpos++];
    ChannelHeader header = {currentword};

    if (header.mMark != 1) {
      LOG(ERROR) << "Channel header mark not found";
      continue;
    }
    // starting a new channel
    mChannels.emplace_back(int(header.mHardwareAddress), int(header.mPayloadSize));
    auto& currentchannel = mChannels.back();
    /// decode all words for channel
    int numberofwords = (currentchannel.getPayloadSize() + 2) / 3;
    if (numberofwords > payloadend - currentpos) {
      LOG(ERROR) << "Channel payload " << numberofwords << " larger than left in total " << payloadend - currentpos;
      continue;
    }
    std::vector<uint16_t> bunchwords;
    for (int iword = 0; iword < numberofwords; iword++) {
      currentword = buffer[currentpos++];
      if ((currentword >> 30) != 0) {
        LOG(ERROR) << "Unexpected end of payload in altro channel payload! FEE=" << o2::raw::RDHUtils::getFEEID(mRawReader.getRawHeader())
                   << ", Address=0x" << std::hex << currentchannel.getHardwareAddress() << ", word=0x" << currentword << std::dec;
        currentpos--;
        continue;
      }
      bunchwords.push_back((currentword >> 20) & 0x3FF);
      bunchwords.push_back((currentword >> 10) & 0x3FF);
      bunchwords.push_back(currentword & 0x3FF);
    }

    // decode bunches
    int currentsample = 0;
    while (currentsample < currentchannel.getPayloadSize()) {
      int bunchlength = bunchwords[currentsample] - 2, // remove words for bunchlength and starttime
        starttime = bunchwords[currentsample + 1];
      auto& currentbunch = currentchannel.createBunch(bunchlength, starttime);
      currentbunch.initFromRange(gsl::span<uint16_t>(&bunchwords[currentsample + 2], std::min((unsigned long)bunchlength, bunchwords.size() - currentsample - 2)));
      currentsample += bunchlength + 2;
    }
  }
  mChannelsInitialized = true;
}

const std::vector<Channel>& AltroDecoder::getChannels() const
{
  if (!mChannelsInitialized) {
    throw AltroDecoderError::ErrorType_t::CHANNEL_ERROR; // "Channels not initizalized");
  }
  return mChannels;
}
