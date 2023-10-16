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
#include <cfloat>
#include <cmath>
#include <iostream>
#include <fmt/format.h>
#include "CommonConstants/LHCConstants.h"
#include "EMCALBase/RCUTrailer.h"
#include <fairlogger/Logger.h>

using namespace o2::emcal;

void RCUTrailer::reset()
{
  mRCUId = -1;
  mFirmwareVersion = 0;
  mTrailerSize = 0;
  mPayloadSize = 0;
  mFECERRA = 0;
  mFECERRB = 0;
  mErrorCounter.mErrorRegister2 = 0;
  mErrorCounter.mErrorRegister3 = 0;
  mActiveFECsA = 0;
  mActiveFECsB = 0;
  mAltroConfig.mWord1 = 0;
  mAltroConfig.mWord2 = 0;
  mIsInitialized = false;
}

bool RCUTrailer::checkLastTrailerWord(uint32_t trailerword)
{
  const int MIN_FWVERSION = 2;
  const int MAX_FWVERSION = 2;
  if ((trailerword >> 30) != 3) {
    return false;
  }
  auto firmwarevesion = (trailerword >> 16) & 0xFF;
  auto trailerSize = (trailerword & 0x7F);
  if (firmwarevesion < MIN_FWVERSION || firmwarevesion > MAX_FWVERSION) {
    return false;
  }
  if (trailerSize < 2) {
    return false;
  }
  if (firmwarevesion == 2) {
    if (trailerSize < 9) {
      return false;
    }
  }
  return true;
}

void RCUTrailer::constructFromRawPayload(const gsl::span<const uint32_t> payloadwords)
{
  reset();
  int index = payloadwords.size();
  auto word = payloadwords[--index];
  if ((word >> 30) != 3) {
    throw Error(Error::ErrorType_t::DECODING_INVALID, "Last RCU trailer word not found!");
  }
  mFirmwareVersion = (word >> 16) & 0xFF;

  mRCUId = (int)((word >> 7) & 0x1FF);
  int trailerSize = (word & 0x7F);

  if (trailerSize < 2) {
    throw Error(Error::ErrorType_t::SIZE_INVALID, fmt::format("Invalid trailer size found (%d bytes) !", trailerSize * 4).data());
  }
  mTrailerSize = trailerSize;

  trailerSize -= 2; // Cut first and last trailer words as they are handled separately
  int foundTrailerWords = 0;
  for (; trailerSize > 0; trailerSize--) {
    word = payloadwords[--index];
    if ((word >> 30) != 2) {
      continue;
    }
    foundTrailerWords++;
    int parCode = (word >> 26) & 0xF;
    int parData = word & 0x3FFFFFF;
    // std::cout << "Found trailer word 0x" << std::hex << word << "(Par code: " << std::dec << parCode << ", Par data: 0x" << std::hex << parData << std::dec << ")" << std::endl;
    switch (parCode) {
      case 1:
        // ERR_REG1
        mFECERRA = ((parData >> 13) & 0x1FFF) << 7;
        mFECERRB = ((parData & 0x1FFF)) << 7;
        break;
      case 2:
        // ERR_REG2
        mErrorCounter.mErrorRegister2 = parData & 0x1FF;
        break;
      case 3:
        // ERR_REG3
        mErrorCounter.mErrorRegister3 = parData & 0x1FFFFFF;
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
        mAltroConfig.mWord1 = parData & 0xFFFFF;
        break;
      case 7:
        // RDO_CFG2
        mAltroConfig.mWord2 = parData & 0x1FFFFFF;
        break;
      default:
        LOG(warning) << "RCU trailer: Undefined parameter code " << parCode << " in word " << index << " (0x" << std::hex << word << std::dec << "), ignoring word";
        mWordCorruptions++;
        break;
    }
  }
  auto lastword = payloadwords[--index];
  if (lastword >> 30 == 2) {
    mPayloadSize = lastword & 0x3FFFFFF;
    foundTrailerWords++;
  }
  if (foundTrailerWords + 1 < mTrailerSize) { // Must account for the first word which was chopped
    throw Error(Error::ErrorType_t::DECODING_INVALID, fmt::format("Corrupted trailer words: {:d} word(s) not having trailer marker", mTrailerSize - foundTrailerWords).data());
  }
  mIsInitialized = true;
}

double RCUTrailer::getTimeSampleNS() const
{
  uint8_t fq = mAltroConfig.mSampleTime;
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
      throw Error(Error::ErrorType_t::SAMPLINGFREQ_INVALID, fmt::format("Invalid sampling frequency value {:d} !", int(fq)).data());
  }

  return tSample * o2::constants::lhc::LHCBunchSpacingNS;
}

void RCUTrailer::setTimeSamplePhaseNS(uint64_t triggertime, uint64_t timesample)
{
  int sample = 0;
  switch (timesample) {
    case 50:
      sample = 0;
      break;
    case 100:
      sample = 1;
      break;
    case 200:
      sample = 2;
      break;
    default:
      throw Error(Error::ErrorType_t::SAMPLINGFREQ_INVALID, fmt::format("invalid time sample: {:f}", timesample).data());
  };
  mAltroConfig.mSampleTime = sample;
  // calculate L1 phase
  mAltroConfig.mL1Phase = (triggertime % timesample) / 25;
}

double RCUTrailer::getL1PhaseNS() const
{
  double tSample = getTimeSampleNS(),
         phase = static_cast<double>(mAltroConfig.mL1Phase) * o2::constants::lhc::LHCBunchSpacingNS;
  if (phase >= tSample) {
    throw Error(Error::ErrorType_t::L1PHASE_INVALID, fmt::format("Invalid L1 trigger phase ({:e} ns (phase) >= {:e} ns (sampling time)) !", phase, tSample).data());
  }
  return phase;
}

std::vector<uint32_t> RCUTrailer::encode() const
{
  std::vector<uint32_t> encoded;
  encoded.emplace_back(mPayloadSize | 2 << 30);
  encoded.emplace_back(mFECERRB >> 7 | (mFECERRA >> 7) << 13 | 1 << 26 | 2 << 30);
  encoded.emplace_back(mErrorCounter.mErrorRegister2 | 2 << 26 | 2 << 30);
  encoded.emplace_back(mErrorCounter.mErrorRegister3 | 3 << 26 | 2 << 30);
  encoded.emplace_back(mActiveFECsA | 4 << 26 | 2 << 30);
  encoded.emplace_back(mActiveFECsB | 5 << 26 | 2 << 30);
  encoded.emplace_back(mAltroConfig.mWord1 | 6 << 26 | 2 << 30);
  encoded.emplace_back(mAltroConfig.mWord2 | 7 << 26 | 2 << 30);

  uint32_t lasttrailerword = 3 << 30 | mFirmwareVersion << 16 | mRCUId << 7 | (encoded.size() + 1);
  encoded.emplace_back(lasttrailerword);

  return encoded;
}

void RCUTrailer::printStream(std::ostream& stream) const
{
  std::vector<std::string> errors;
  double timesample = -1., l1phase = -1.;
  try {
    timesample = getTimeSampleNS();
  } catch (Error& e) {
    errors.push_back(e.what());
  }
  try {
    l1phase = getL1PhaseNS();
  } catch (Error& e) {
    errors.push_back(e.what());
  }

  stream << "RCU trailer (Format version 2):\n"
         << "==================================================\n"
         << "RCU ID:                                    " << mRCUId << "\n"
         << "Firmware version:                          " << int(mFirmwareVersion) << "\n"
         << "Trailer size:                              " << mTrailerSize << "\n"
         << "Payload size:                              " << mPayloadSize << "\n"
         << "FECERRA:                                   0x" << std::hex << mFECERRA << "\n"
         << "FECERRB:                                   0x" << std::hex << mFECERRB << "\n"
         << "ERRREG2:                                   0x" << std::hex << mErrorCounter.mErrorRegister2 << "\n"
         << "ERRREG3:                                   0x" << std::hex << mErrorCounter.mErrorRegister3 << "\n"
         << "#channels skipped due to address mismatch: " << std::dec << getNumberOfChannelAddressMismatch() << "\n"
         << "#channels skipped due to bad block length: " << std::dec << getNumberOfChannelLengthMismatch() << "\n"
         << "Active FECs (branch A):                    0x" << std::hex << mActiveFECsA << "\n"
         << "Active FECs (branch B):                    0x" << std::hex << mActiveFECsB << "\n"
         << "Baseline corr:                             " << std::hex << getBaselineCorrection() << "\n"
         << "Polarity:                                  " << (getPolarity() ? "yes" : "no") << "\n"
         << "Number of presamples:                      " << std::dec << getNumberOfPresamples() << "\n"
         << "Number of postsamples:                     " << std::dec << getNumberOfPostsamples() << "\n"
         << "Second baseline corr:                      " << (hasSecondBaselineCorr() ? "yes" : "no") << "\n"
         << "Glitch filter:                             " << std::dec << getGlitchFilter() << "\n"
         << "Number of non-ZS postsamples:              " << std::dec << getNumberOfNonZeroSuppressedPostsamples() << "\n"
         << "Number of non-ZS presamples:               " << std::dec << getNumberOfNonZeroSuppressedPresamples() << "\n"
         << "Zero suppression:                          " << (hasZeroSuppression() ? "yes" : "no") << "\n"
         << "Number of ALTRO buffers:                   " << std::dec << getNumberOfAltroBuffers() << "\n"
         << "Number of pretrigger samples:              " << std::dec << getNumberOfPretriggerSamples() << "\n"
         << "Number of samples per channel:             " << std::dec << getNumberOfSamplesPerChannel() << "\n"
         << "Sparse readout:                            " << (isSparseReadout() ? "yes" : "no") << "\n"
         << "AltroCFG1:                                 0x" << std::hex << mAltroConfig.mWord1 << "\n"
         << "AltroCFG2:                                 0x" << std::hex << mAltroConfig.mWord2 << "\n"
         << "Sampling time:                             " << std::dec << timesample << " ns\n"
         << "L1 Phase:                                  " << std::dec << l1phase << " ns (" << mAltroConfig.mL1Phase << ")\n"
         << std::dec << std::fixed;
  if (errors.size()) {
    stream << "Errors: \n"
           << "-------------------------------------------------\n";
    for (const auto& e : errors) {
      stream << e << "\n";
    }
  }
  stream << "==================================================\n";
}

RCUTrailer RCUTrailer::constructFromPayloadWords(const gsl::span<const uint32_t> payloadwords)
{
  RCUTrailer result;
  result.constructFromRawPayload(payloadwords);
  return result;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::RCUTrailer& trailer)
{
  trailer.printStream(stream);
  return stream;
}