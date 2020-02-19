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
#include <iomanip>
#include <iostream>
#include <boost/format.hpp>
#include "InfoLogger/InfoLogger.hxx"
#include "Headers/RAWDataHeader.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/RAWDataHeader.h"
#include "EMCALReconstruction/RawReaderFile.h"
#include "EMCALReconstruction/RawReaderMemory.h"

using namespace o2::emcal;

template <class RawReader>
AltroDecoder<RawReader>::AltroDecoder(RawReader& reader) : mRawReader(reader),
                                                           mRCUTrailer(),
                                                           mChannels(),
                                                           mChannelsInitialized(false)
{
}

template <class RawReader>
void AltroDecoder<RawReader>::decode()
{
  readRCUTrailer();
  checkRCUTrailer();
  readChannels();
}

template <class RawReader>
void AltroDecoder<RawReader>::readRCUTrailer()
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

template <class RawReader>
void AltroDecoder<RawReader>::checkRCUTrailer()
{
}

template <class RawReader>
void AltroDecoder<RawReader>::readChannels()
{
  mChannelsInitialized = false;
  mChannels.clear();
  int currentpos = 0;
  auto& buffer = mRawReader.getPayload().getPayloadWords();
  std::array<uint16_t, 1024> bunchwords;
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
    int numberofsamples = 0,
        numberofwords = (currentchannel.getPayloadSize() + 2) / 3;
    for (int iword = 0; iword < numberofwords; iword++) {
      currentword = buffer[currentpos++];
      if ((currentword >> 30) != 0) {
        // AliceO2::InfoLogger::InfoLogger logger;
        // logger << "Unexpected end of payload in altro channel payload! DDL=" << std::setw(3) << std::setfill(0) << mRawReader.getRawHeader().getLink()
        //       << ", Address=0x" << std::hex << current.getHardwareAddress() << ", word=0x" << currentword << std::dec;
        currentpos--;
        continue;
      }
      bunchwords[numberofsamples++] = (currentword >> 20) & 0x3FF;
      bunchwords[numberofsamples++] = (currentword >> 10) & 0x3FF;
      bunchwords[numberofsamples++] = currentword & 0x3FF;
    }

    // decode bunches
    int currentsample = 0;
    while (currentsample < currentchannel.getPayloadSize()) {
      int bunchlength = bunchwords[currentsample] - 2, // remove words for bunchlength and starttime
        starttime = bunchwords[currentsample + 1];
      auto& currentbunch = currentchannel.createBunch(bunchlength, starttime);
      currentbunch.initFromRange(gsl::span<uint16_t>(&bunchwords[currentsample + 2], std::min(bunchlength, numberofsamples - currentsample - 2)));
      currentsample += bunchlength + 2;
    }
  }
  mChannelsInitialized = true;
}

template <class RawReader>
const RCUTrailer& AltroDecoder<RawReader>::getRCUTrailer() const
{
  if (!mRCUTrailer.isInitialized())
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::RCU_TRAILER_ERROR, "RCU trailer was not initialized");
  return mRCUTrailer;
}

template <class RawReader>
const std::vector<Channel>& AltroDecoder<RawReader>::getChannels() const
{
  if (!mChannelsInitialized)
    throw AltroDecoderError(AltroDecoderError::ErrorType_t::CHANNEL_ERROR, "Channels not initizalized");
  return mChannels;
}

template class o2::emcal::AltroDecoder<o2::emcal::RawReaderFile<o2::emcal::RAWDataHeader>>;
template class o2::emcal::AltroDecoder<o2::emcal::RawReaderFile<o2::header::RAWDataHeaderV4>>;
template class o2::emcal::AltroDecoder<o2::emcal::RawReaderMemory<o2::emcal::RAWDataHeader>>;
template class o2::emcal::AltroDecoder<o2::emcal::RawReaderMemory<o2::header::RAWDataHeaderV4>>;