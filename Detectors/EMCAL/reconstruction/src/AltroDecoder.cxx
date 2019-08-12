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
#include "Rtypes.h"
#include "InfoLogger/InfoLogger.hxx"
#include "CommonConstants/LHCConstants.h"
#include "EMCALReconstruction/AltroDecoder.h"

using namespace o2::emcal;

AltroDecoder::AltroDecoder(RawReaderFile& reader) : mRawReader(reader),
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
    mRCUTrailer.constructFromRawBuffer(mRawReader.getRawBuffer());
  } catch (RCUTrailer::Error& e) {
    AliceO2::InfoLogger::InfoLogger logger;
    logger << e.what();
    throw Error(Error::ErrorType_t::RCU_TRAILER_ERROR, (boost::format("RCU trailer decoding error: %s") % e.what()).str().data());
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
  auto& buffer = mRawReader.getRawBuffer();
  std::array<uint16_t, 1024> bunchwords;
  while (currentpos < buffer.getNDataWords() - mRCUTrailer.getTrailerSize()) {
    auto currentword = buffer.getWord(currentpos++);
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
      currentword = buffer.getWord(currentpos++);
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
    while (currentsample < numberofsamples) {
      int bunchlength = bunchwords[currentsample] - 2, // remove words for bunchlength and starttime
        starttime = bunchwords[currentsample + 1];
      auto& currentbunch = currentchannel.createBunch(bunchlength, starttime);
      currentbunch.initFromRange(gsl::span<uint16_t>(&bunchwords[currentsample + 2], std::min(bunchlength, numberofsamples - currentsample - 2)));
      currentsample += bunchlength + 2;
    }
  }
  mChannelsInitialized = true;
}

const AltroDecoder::RCUTrailer& AltroDecoder::getRCUTrailer() const
{
  if (!mRCUTrailer.isInitialized())
    throw Error(Error::ErrorType_t::RCU_TRAILER_ERROR, "RCU trailer was not initialized");
  return mRCUTrailer;
}

const std::vector<AltroDecoder::Channel>& AltroDecoder::getChannels() const
{
  if (!mChannelsInitialized)
    throw Error(Error::ErrorType_t::CHANNEL_ERROR, "Channels not initizalized");
  return mChannels;
}

void AltroDecoder::RCUTrailer::reset()
{
  mRCUId = -1;
  mFirmwareVersion = 0;
  mTrailerSize = 0;
  mPayloadSize = 0;
  mFECERRA = 0;
  mFECERRB = 0;
  mERRREG2 = 0;
  mERRREG3 = 0;
  mActiveFECsA = 0;
  mActiveFECsB = 0;
  mAltroCFG1 = 0;
  mAltroCFG2 = 0;
  mIsInitialized = false;
}

void AltroDecoder::RCUTrailer::constructFromRawBuffer(const RawBuffer& buffer)
{
  reset();
  int index = buffer.getNDataWords() - 1;
  auto word = buffer.getWord(index);
  if ((word >> 30) != 3)
    throw Error(Error::ErrorType_t::DECODING_INVALID, "Last RCU trailer word not found!");
  mFirmwareVersion = (word >> 16) & 0xFF;

  mRCUId = (int)((word >> 7) & 0x1FF);
  int trailerSize = (word & 0x7F);

  if (trailerSize < 2)
    throw Error(Error::ErrorType_t::SIZE_INVALID, (boost::format("Invalid trailer size found (%d bytes) !") % (trailerSize * 4)).str().data());
  mTrailerSize = trailerSize;

  for (; trailerSize > 0; trailerSize--) {
    word = buffer.getWord(--index);
    if ((word >> 30) != 2) {
      std::cerr << "Missing RCU trailer identifier pattern!\n";
      continue;
    }
    Int_t parCode = (word >> 26) & 0xF;
    Int_t parData = word & 0x3FFFFFF;
    switch (parCode) {
      case 1:
        // ERR_REG1
        mFECERRA = ((parData >> 13) & 0x1FFF) << 7;
        mFECERRB = ((parData & 0x1FFF)) << 7;
        break;
      case 2:
        // ERR_REG2
        mERRREG2 = parData & 0x1FF;
        break;
      case 3:
        // ERR_REG3
        mERRREG3 = parData & 0x1FFFFFF;
        break;
      case 4:
        // FEC_RO_A
        mActiveFECsA = parData & 0xFFFF;
        break;
      case 5:
        // FEC_RO_B
        mActiveFECsB = parData & 0xFFFF;
        break;
      case 6:
        // RDO_CFG1
        mAltroCFG1 = parData & 0xFFFFF;
        break;
      case 7:
        // RDO_CFG2
        mAltroCFG2 = parData & 0x1FFFFFF;
        break;
      default:
        std::cerr << "Undefined parameter code " << parCode << ", ignore it !\n";
        break;
    }
  }
  mPayloadSize = buffer.getWord(--index) & 0x3FFFFFF;
  mIsInitialized = true;
}

double AltroDecoder::RCUTrailer::getTimeSample() const
{
  unsigned char fq = (mAltroCFG2 >> 5) & 0xF;
  double tSample;
  switch (fq) {
    case 0:
      // 20 MHz
      tSample = 2.0;
      break;
    case 1:
      // 10 Mhz
      tSample = 4.0;
      break;
    case 2:
      // 5 MHz
      tSample = 8.;
      break;
    default:
      throw Error(Error::ErrorType_t::SAMPLINGFREQ_INVALID, (boost::format("Invalid sampling frequency value %d !") % int(fq)).str().data());
  }

  return tSample * o2::constants::lhc::LHCBunchSpacingNS * 1.e-9;
}

double AltroDecoder::RCUTrailer::getL1Phase() const
{
  double tSample = getTimeSample(),
         phase = ((double)(mAltroCFG2 & 0x1F)) * o2::constants::lhc::LHCBunchSpacingNS * 1.e-9;
  if (phase >= tSample) {
    throw Error(Error::ErrorType_t::L1PHASE_INVALID, (boost::format("Invalid L1 trigger phase (%f >= %f) !") % phase % tSample).str().data());
  }
  return phase;
}

void AltroDecoder::RCUTrailer::printStream(std::ostream& stream) const
{
  stream << "RCU trailer (Format version 2):\n"
         << "==================================================\n"
         << "RCU ID:                                    " << mRCUId << "\n"
         << "Firmware version:                          " << int(mFirmwareVersion) << "\n"
         << "Trailer size:                              " << mTrailerSize << "\n"
         << "Payload size:                              " << mPayloadSize << "\n"
         << "FECERRA:                                   0x" << std::hex << mFECERRA << "\n"
         << "FECERRB:                                   0x" << std::hex << mFECERRB << "\n"
         << "ERRREG2:                                   0x" << std::hex << mERRREG2 << "\n"
         << "#channels skipped due to address mismatch: " << std::dec << getNumberOfChannelAddressMismatch() << "\n"
         << "#channels skipped due to bad block length: " << std::dec << getNumberOfChannelLengthMismatch() << "\n"
         << "Active FECs (branch A):                    0x" << std::hex << mActiveFECsA << "\n"
         << "Active FECs (branch B):                    0x" << std::hex << mActiveFECsB << "\n"
         << "Baseline corr:                             0x" << std::hex << int(getBaselineCorrection()) << "\n"
         << "Number of presamples:                      " << std::dec << int(getNumberOfPresamples()) << "\n"
         << "Number of postsamples:                     " << std::dec << int(getNumberOfPostsamples()) << "\n"
         << "Second baseline corr:                      " << (hasSecondBaselineCorr() ? "yes" : "no") << "\n"
         << "GlitchFilter:                              " << std::dec << int(getGlitchFilter()) << "\n"
         << "Number of non-ZS postsamples:              " << std::dec << int(getNumberOfNonZeroSuppressedPostsamples()) << "\n"
         << "Number of non-ZS presamples:               " << std::dec << int(getNumberOfNonZeroSuppressedPresamples()) << "\n"
         << "Number of ALTRO buffers:                   " << std::dec << getNumberOfAltroBuffers() << "\n"
         << "Number of pretrigger samples:              " << std::dec << int(getNumberOfPretriggerSamples()) << "\n"
         << "Number of samples per channel:             " << std::dec << getNumberOfSamplesPerChannel() << "\n"
         << "Sparse readout:                            " << (isSparseReadout() ? "yes" : "no") << "\n"
         << "Sampling time:                             " << std::scientific << getTimeSample() << " s\n"
         << "L1 Phase:                                  " << std::scientific << getL1Phase() << " s\n"
         << "AltroCFG1:                                 0x" << std::hex << mAltroCFG1 << "\n"
         << "AltroCFG2:                                 0x" << std::hex << mAltroCFG2 << "\n"
         << "==================================================\n"
         << std::dec << std::fixed;
}

int AltroDecoder::Channel::getBranchIndex() const
{
  if (mHardwareAddress == -1)
    throw HardwareAddressError();
  return ((mHardwareAddress >> 11) & 0x1);
}

int AltroDecoder::Channel::getFECIndex() const
{
  if (mHardwareAddress == -1)
    throw HardwareAddressError();
  return ((mHardwareAddress >> 7) & 0xF);
}

Int_t AltroDecoder::Channel::getAltroIndex() const
{
  if (mHardwareAddress == -1)
    throw HardwareAddressError();
  return ((mHardwareAddress >> 4) & 0x7);
}

Int_t AltroDecoder::Channel::getChannelIndex() const
{
  if (mHardwareAddress == -1)
    throw HardwareAddressError();
  return (mHardwareAddress & 0xF);
}

AltroDecoder::Bunch& AltroDecoder::Channel::createBunch(uint8_t bunchlength, uint8_t starttime)
{
  mBunches.emplace_back(bunchlength, starttime);
  return mBunches.back();
}

void AltroDecoder::Bunch::initFromRange(gsl::span<uint16_t> adcs)
{
  for (auto adcval : adcs)
    mADC.emplace_back(adcval);
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const AltroDecoder::RCUTrailer& trailer)
{
  trailer.printStream(stream);
  return stream;
}
