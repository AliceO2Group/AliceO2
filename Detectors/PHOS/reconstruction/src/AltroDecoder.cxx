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
#include <cstring>
#include <cstdint>
#include <boost/format.hpp>
#include "InfoLogger/InfoLogger.hxx"
#include "PHOSBase/PHOSSimParams.h"
#include "PHOSBase/Geometry.h"
#include "PHOSReconstruction/AltroDecoder.h"
#include "PHOSReconstruction/RawReaderMemory.h"
#include "PHOSReconstruction/RawDecodingError.h"
#include "DetectorsRaw/RDHUtils.h"
#include <fairlogger/Logger.h>

using namespace o2::phos;

AltroDecoderError::ErrorType_t AltroDecoder::decode(RawReaderMemory& rawreader, CaloRawFitter* rawFitter,
                                                    std::vector<o2::phos::Cell>& currentCellContainer, std::vector<o2::phos::Cell>& currentTRUContainer)
{
  mOutputHWErrors.clear();
  mOutputFitChi.clear();

  try {
    auto& header = rawreader.getRawHeader();
    mddl = o2::raw::RDHUtils::getFEEID(header);
  } catch (...) {
    return AltroDecoderError::RCU_TRAILER_ERROR;
  }
  const std::vector<uint32_t>& payloadwords = rawreader.getPayload().getPayloadWords();

  if (payloadwords.size() == 0) {
    return AltroDecoderError::kOK;
  }

  try {
    gsl::span<const uint32_t> tmp(payloadwords.data(), payloadwords.size());
    mRCUTrailer.constructFromRawPayload(tmp);
  } catch (RCUTrailer::Error& e) {
    mOutputHWErrors.emplace_back(mddl, kGeneralSRUErr, static_cast<char>(e.getErrorType())); // assign general SRU header errors to non-existing FEE 15
    return AltroDecoderError::RCU_TRAILER_ERROR;
  }

  try {
    readChannels(payloadwords, rawFitter, currentCellContainer, currentTRUContainer);
  } catch (AltroDecoderError::ErrorType_t e) {
    mOutputHWErrors.emplace_back(mddl, kGeneralTRUErr, static_cast<char>(e)); // assign general SRU header errors to non-existing FEE 16
    return e;
  }
  return AltroDecoderError::kOK;
}

void AltroDecoder::readChannels(const std::vector<uint32_t>& buffer, CaloRawFitter* rawFitter,
                                std::vector<o2::phos::Cell>& currentCellContainer, std::vector<o2::phos::Cell>& currentTRUContainer)
{
  int currentpos = 0;
  mTRUFlags.clear();
  mTRUDigits.fill(0);
  mFlag4x4Bitset.reset();
  mFlag2x2Bitset.reset();

  int payloadend = buffer.size() - mRCUTrailer.getTrailerSize(); // mRCUTrailer.getPayloadSize() was not updated in case of merged pages.
  // Extract offset from fee configuration
  short value = mRCUTrailer.getAltroCFGReg1();
  short offset = (value >> 10) & 0xf;
  while (currentpos < payloadend) {
    auto currentword = buffer[currentpos++];
    ChannelHeader header = {currentword};
    if (header.mMark != 1) {
      if (currentword != 0) {
        short fec = header.mHardwareAddress >> 7 & 0xf; // try to extract FEE number from header
        short branch = header.mHardwareAddress >> 11 & 0x1;
        if (fec > 14) {
          fec = kGeneralSRUErr;
        }
        fec += kGeneralTRUErr * branch;
        mOutputHWErrors.emplace_back(mddl, fec, 5); // 5: channel header error
      }
      continue;
    }
    /// decode all words for channel
    int numberofwords = (header.mPayloadSize + 2) / 3;
    if (numberofwords > payloadend - currentpos) {
      short fec = header.mHardwareAddress >> 7 & 0xf; // try to extract FEE number from header
      short branch = header.mHardwareAddress >> 11 & 0x1;
      if (fec > 14) {
        fec = kGeneralSRUErr;
      }
      fec += kGeneralTRUErr * branch;
      mOutputHWErrors.emplace_back(mddl, fec, 6); // 6: channel payload error
      continue;
    }
    mBunchwords.clear();
    int isample = 0;
    while (isample < header.mPayloadSize) {
      currentword = buffer[currentpos++];
      if ((currentword >> 30) != 0) {
        currentpos--;
        short fec = header.mHardwareAddress >> 7 & 0xf; // try to extract FEE number from header
        short branch = header.mHardwareAddress >> 11 & 0x1;
        if (fec > 14) {
          fec = kGeneralSRUErr;
        }
        fec += kGeneralTRUErr * branch;
        mOutputHWErrors.emplace_back(mddl, fec, 6); // 6: channel payload error
        break;
      }
      mBunchwords.push_back((currentword >> 20) & 0x3FF);
      isample++;
      if (isample < header.mPayloadSize) {
        mBunchwords.push_back((currentword >> 10) & 0x3FF);
        isample++;
        if (isample < header.mPayloadSize) {
          mBunchwords.push_back(currentword & 0x3FF);
          isample++;
        } else {
          break;
        }
      } else {
        break;
      }
    }
    short absId;
    Mapping::CaloFlag caloFlag;
    if (!hwToAbsAddress(header.mHardwareAddress, absId, caloFlag)) {
      // do not decode, skip to hext channel
      short fec = header.mHardwareAddress >> 7 & 0xf; // try to extract FEE number from header
      short branch = header.mHardwareAddress >> 11 & 0x1;
      if (fec > 14) {
        fec = kGeneralSRUErr;
      }
      fec += kGeneralTRUErr * branch;
      mOutputHWErrors.emplace_back(mddl, fec, 7); // 7: wrong hw address
      continue;
    }

    // Get time and amplitude
    if (caloFlag != Mapping::kTRU) { // HighGain or LowGain
      // decode bunches
      int currentsample = 0;
      while (currentsample < header.mPayloadSize) {
        int bunchlength = mBunchwords[currentsample] - 2, // remove words for bunchlength and starttime
          starttime = mBunchwords[currentsample + 1];
        if (bunchlength < 0) {                            // corrupted data,
          short fec = header.mHardwareAddress >> 7 & 0xf; // try to extract FEE number from header
          short branch = header.mHardwareAddress >> 11 & 0x1;
          fec += kGeneralTRUErr * branch;
          mOutputHWErrors.emplace_back(mddl, fec, 6); // 6: channel payload error
          break;
        }
        // extract sample properties
        CaloRawFitter::FitStatus fitResult = rawFitter->evaluate(gsl::span<uint16_t>(&mBunchwords[currentsample + 2], std::min((unsigned long)bunchlength, mBunchwords.size() - currentsample - 2)));
        currentsample += bunchlength + 2;
        if (!rawFitter->isOverflow() && rawFitter->getChi2() > 0) { // Overflow is will show wrong chi2
          short chiAddr = absId;
          chiAddr |= caloFlag << 14;
          mOutputFitChi.emplace_back(chiAddr);
          mOutputFitChi.emplace_back(short(std::min(5.f * rawFitter->getChi2(), float(SHRT_MAX - 1)))); // 0.2 accuracy
        }
        if (fitResult == CaloRawFitter::FitStatus::kOK || fitResult == CaloRawFitter::FitStatus::kNoTime) {
          if (!mPedestalRun) {
            if (caloFlag == Mapping::kHighGain && !rawFitter->isOverflow()) {
              currentCellContainer.emplace_back(absId, std::max(rawFitter->getAmp() - offset, float(0)),
                                                (rawFitter->getTime() + starttime - bunchlength - mPreSamples) * o2::phos::PHOSSimParams::Instance().mTimeTick * 1.e-9, (ChannelType_t)caloFlag);
            }
            if (caloFlag == Mapping::kLowGain) {
              currentCellContainer.emplace_back(absId, std::max(rawFitter->getAmp() - offset, float(0)),
                                                (rawFitter->getTime() + starttime - bunchlength - mPreSamples) * o2::phos::PHOSSimParams::Instance().mTimeTick * 1.e-9, (ChannelType_t)caloFlag);
            }
          } else { // pedestal, to store RMS, scale in by 1.e-7 to fit range
            currentCellContainer.emplace_back(absId, std::max(rawFitter->getAmp() - offset, float(0)), 1.e-7 * rawFitter->getTime(), (ChannelType_t)caloFlag);
          }
        }  // Successful fit
      }    // Bunched of a channel
    }      // HG or LG channel
    else { // TRU channel
      // Channels in TRU:
      // There are 112 readout channels and 12 channels reserved for production flags:
      //  Channels 0-111: channel data readout
      //  Channels 112-123: production flags
      if (Mapping::isTRUReadoutchannel(header.mHardwareAddress)) {
        Mapping::Instance()->hwToAbsId(mddl, header.mHardwareAddress, absId, caloFlag);
        readTRUDigits(absId, header.mPayloadSize);
      } else {
        readTRUFlags(header.mHardwareAddress, header.mPayloadSize);
      }
    } // TRU channel
  }

  if (mKeepTruNoise) { // copy all TRU digits and TRU flags for noise scan
    // TRU flags are copied with 4x4 mark
    for (const Cell cFlag : mTRUFlags) {
      currentTRUContainer.emplace_back(cFlag);
      currentTRUContainer.back().setType(TRU4x4);
    }
    // Copy digits with 2x2 mark
    for (int itru = 0; itru < 224; itru++) {
      if (mTRUDigits[itru] > 0) {
        short absId = Mapping::NCHANNELS + 224 * mddl + itru + 1;
        truDigitPack dp = {mTRUDigits[itru]};
        float a = dp.mAmp, t = dp.mTime;
        currentTRUContainer.emplace_back(absId, a, t, TRU2x2);
      }
    }
  } else {
    // Find matching of Flags and truDigits and create output
    // if trigger cell exists and  the trigger flag true -add it
    // Normally we have few ~2-4 digits and flags per event
    // no need for clever algoritm here
    // One 2x2 tru digit can contribute several 4x4 TRU flags
    for (const Cell cFlag : mTRUFlags) {
      float sum = 0;
      if (matchTruDigits(cFlag, sum)) {
        currentTRUContainer.emplace_back(cFlag);
        currentTRUContainer.back().setEnergy(sum);
      }
    }
  }
}

bool AltroDecoder::hwToAbsAddress(short hwAddr, short& absId, Mapping::CaloFlag& caloFlag)
{
  // check hardware address and convert to absId and caloFlag

  if (mddl < 0 || mddl > o2::phos::Mapping::NDDL) {
    return (char)4;
  }
  //  short chan = hwAddr & 0xf;
  short chip = hwAddr >> 4 & 0x7;
  short fec = hwAddr >> 7 & 0xf;
  short branch = hwAddr >> 11 & 0x1;

  short e2 = 0;
  if (fec > 14) {
    e2 = 2;
    fec = kGeneralSRUErr;
    mOutputHWErrors.emplace_back(mddl, fec + branch * kGeneralTRUErr, 2);
  } else {
    if (fec != 0 && (chip < 0 || chip > 4 || chip == 1)) { // Do not check for TRU (fec=0)
      e2 = 3;
      mOutputHWErrors.emplace_back(mddl, fec + branch * kGeneralTRUErr, 3);
    }
  }

  if (e2) {
    return false;
  }
  // correct hw address, try to convert
  Mapping::ErrorStatus s = Mapping::Instance()->hwToAbsId(mddl, hwAddr, absId, caloFlag);
  if (s != Mapping::ErrorStatus::kOK) {
    mOutputHWErrors.emplace_back(mddl, branch * kGeneralTRUErr + kGeneralSRUErr, 4); // 4: error in mapping
    return false;
  }
  return true;
}

void AltroDecoder::readTRUDigits(short absId, int payloadSize)
{
  int currentsample = 0;
  short maxAmp = 0;
  int timeBin = 0;
  while (currentsample < payloadSize) {
    int bunchlength = mBunchwords[currentsample] - 2; // remove words for bunchlength and starttime
    if (bunchlength < 0) {                            // corrupted sample: add error and ignore the reast of bunchwords
      // 1: wrong TRU header
      mOutputHWErrors.emplace_back(mddl, kGeneralTRUErr, static_cast<char>(1));
      return;
    }
    timeBin = mBunchwords[currentsample + 1] - bunchlength;
    int istart = currentsample + 2;
    int iend = std::min(istart + bunchlength - 2, static_cast<int>(mBunchwords.size()));
    for (int i = istart; i < iend; i++) {
      if (maxAmp < mBunchwords[i]) {
        maxAmp = mBunchwords[i];
      }
    }
    currentsample += bunchlength + 2;
  }
  truDigitPack dp = {0};
  dp.mHeader = -1;
  dp.mAmp = maxAmp;
  dp.mTime = timeBin;
  int chId = (absId - Mapping::NCHANNELS - 1) % 224;
  mTRUDigits[chId] = dp.mDataWord;
}
void AltroDecoder::readTRUFlags(short hwAddress, int payloadSize)
{
  // Production flags:
  // Production flags are supplied in channels 112 - 123
  // Each of the channels is 10 bit wide
  // The bits inside the channel (indexing starting from the first bit of channel 112) is as follows:
  //  Bits 0-111: Trigger flags for corresponding channel index
  //    If using 4x4 algorithm, only 91 first bits are used of these
  //  information about used trigger is stored in channel 123
  //  Bit 112: Marker for 4x4 algorithm (1 active, 0 not active)
  //  Bit 113: Marker for 2x2 algorithm (1 active, 0 not active)
  //  Bit 114: Global L0 OR of all patches in the TRU
  const int kWordLength = 10; // Length of one data word in TRU raw data
  int currentsample = 0;
  while (currentsample < payloadSize) {
    int bunchlength = mBunchwords[currentsample] - 2; // remove words for bunchlength and starttime
    if (bunchlength < 1) {                            // corrupted sample: add error and ignore the reast of bunchwords
      // 1: wrong TRU header
      mOutputHWErrors.emplace_back(mddl, kGeneralTRUErr, static_cast<char>(1));
      return;
    }
    int timeBin = mBunchwords[currentsample + 1] + 1; // +1 for further convenience
    int istart = currentsample + 2;
    int iend = istart + std::min(bunchlength, static_cast<int>(mBunchwords.size()) - currentsample - 2);
    currentsample += bunchlength + 2;

    for (int i = iend - 1; i >= istart; i--) {
      --timeBin;
      short a = mBunchwords[i];

      // Assign the bits in the words to corresponding channels
      for (Int_t bitIndex = 0; bitIndex < kWordLength; bitIndex++) {
        // Find the correct channel number assuming that
        // hwAddress 112 = bits 0-9 corresponding trigger flags in channels 0-9
        // hwAddress 113 = bits 10-19 corresponding trigger flags in channels 10-19
        // and so on
        short channel;
        if (hwAddress < 128) {
          channel = (hwAddress - Mapping::NTRUBranchReadoutChannels) * kWordLength + bitIndex;
        } else {
          channel = 112 + (hwAddress - 2048 - Mapping::NTRUBranchReadoutChannels) * kWordLength + bitIndex; // branch 0
        }
        if (hwAddress == Mapping::TRUFinalProductionChannel || hwAddress == Mapping::TRUFinalProductionChannel + 2048) {
          // fill 4x4 or 2x2 flags
          if ((a & (1 << 2)) > 0) {
            mFlag4x4Bitset[timeBin] = 1;
          }
          if ((a & (1 << 3)) > 0) { // bit 113 2x2 trigger
            mFlag2x2Bitset[timeBin] = 1;
          }
        } else {
          short absId;
          o2::phos::Mapping::CaloFlag fl;
          if (a & (1 << bitIndex)) {
            ChannelType_t trFlag = TRU4x4;
            if (mFlag4x4Bitset[timeBin]) {
              trFlag = TRU4x4;
              if ((channel > 90 && channel < 112) || channel > 202) { // no such channels in 4x4 trigger
                continue;
              }
            } else {
              if (mFlag2x2Bitset[timeBin]) {
                trFlag = TRU2x2;
              } else { // trigger was not fired at all at this time bin
                continue;
              }
            }
            Mapping::Instance()->hwToAbsId(mddl, channel, absId, fl);
            // Prepare TRU cell with zero yet amplitude
            if (mTRUFlags.size() > 0 && mTRUFlags.back().getTRUId() == absId) { // Just added, set earliest time
              mTRUFlags.back().setTime(timeBin * o2::phos::PHOSSimParams::Instance().mTRUTimeTick * 1.e-9);
            } else {
              mTRUFlags.emplace_back(absId, timeBin * o2::phos::PHOSSimParams::Instance().mTRUTimeTick * 1.e-9, 0., trFlag);
            }
          }
        }
      } // Bits in one word
    }   // Length of signal
  }
}
bool AltroDecoder::matchTruDigits(const Cell& cTruFlag, float& sumAmp)
{
  // Check if TRU digit matches with TRU flag
  //  return true if at least one matched
  //  and sum of amplitudes
  // TODO Should we check time as well? So far keep time of summary table mark
  if (cTruFlag.getType() == TRU2x2) { // direct match of channel ID
    short ch = (cTruFlag.getTRUId() - Mapping::NCHANNELS - 1) % 224;
    if (mTRUDigits[ch] > 0) {
      truDigitPack dp = {mTRUDigits[ch]};
      sumAmp = dp.mAmp;
      return true;
    } else {
      sumAmp = 0.;
      return false;
    }
  }
  if (cTruFlag.getType() == TRU4x4) { // direct match of channel ID
    char relid[3];
    Geometry::truAbsToRelNumbering(cTruFlag.getTRUId(), 1, relid); // 1 for 4x4 trigger
    bool found = false;
    sumAmp = 0.;
    short ch = Geometry::truRelToAbsNumbering(relid, 0); // first 2x2 tile
    ch = (ch - Mapping::NCHANNELS - 1) % 224;
    if (mTRUDigits[ch] != 0) {
      truDigitPack dp = {mTRUDigits[ch]};
      sumAmp += dp.mAmp;
      found = true;
    }
    relid[1] += 2;
    ch = Geometry::truRelToAbsNumbering(relid, 0); // another 2x2 tile
    ch = (ch - Mapping::NCHANNELS - 1) % 224;
    if (mTRUDigits[ch] != 0) {
      truDigitPack dp = {mTRUDigits[ch]};
      sumAmp += dp.mAmp;
      found = true;
    }
    relid[2] += 2;
    ch = Geometry::truRelToAbsNumbering(relid, 0); // another 2x2 tile
    ch = (ch - Mapping::NCHANNELS - 1) % 224;
    if (mTRUDigits[ch] != 0) {
      truDigitPack dp = {mTRUDigits[ch]};
      sumAmp += dp.mAmp;
      found = true;
    }
    relid[1] -= 2;
    ch = Geometry::truRelToAbsNumbering(relid, 0); // another 2x2 tile
    ch = (ch - Mapping::NCHANNELS - 1) % 224;
    if (mTRUDigits[ch] != 0) {
      truDigitPack dp = {mTRUDigits[ch]};
      sumAmp += dp.mAmp;
      found = true;
    }
    return found;
  }
  // Not applicable for non-TRU cells
  return false;
}
