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
#include <boost/format.hpp>
#include "InfoLogger/InfoLogger.hxx"
#include "PHOSBase/PHOSSimParams.h"
#include "PHOSReconstruction/AltroDecoder.h"
#include "PHOSReconstruction/RawReaderMemory.h"
#include "PHOSReconstruction/RawDecodingError.h"

#include "DetectorsRaw/RDHUtils.h"
#include <FairLogger.h>

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
    LOG(error) << "RCU trailer error " << (int)e.getErrorType();
    mOutputHWErrors.emplace_back(mddl, kGeneralSRUErr, static_cast<char>(e.getErrorType())); // assign general SRU header errors to non-existing FEE 15
    return AltroDecoderError::RCU_TRAILER_ERROR;
  }

  // TODO  checkRCUTrailer();
  try {
    readChannels(payloadwords, rawFitter, currentCellContainer, currentTRUContainer);
  } catch (AltroDecoderError::ErrorType_t e) {
    LOG(error) << "Altro decoding error " << e;
    mOutputHWErrors.emplace_back(mddl, kGeneralTRUErr, static_cast<char>(e)); // assign general SRU header errors to non-existing FEE 16
    return e;
  }
  return AltroDecoderError::kOK;
}

void AltroDecoder::readChannels(const std::vector<uint32_t>& buffer, CaloRawFitter* rawFitter,
                                std::vector<o2::phos::Cell>& currentCellContainer, std::vector<o2::phos::Cell>& currentTRUContainer)
{
  int currentpos = 0;

  int payloadend = buffer.size() - mRCUTrailer.getTrailerSize(); // mRCUTrailer.getPayloadSize() was not updated in case of merged pages.
  // Extract offset from fee configuration
  short value = mRCUTrailer.getAltroCFGReg1();
  short offset = (value >> 10) & 0xf;
  while (currentpos < payloadend) {
    auto currentword = buffer[currentpos++];
    ChannelHeader header = {currentword};
    if (header.mMark != 1) {
      if (currentword != 0) {
        LOG(error) << "Channel header mark not found, header word " << currentword;
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
      LOG(error) << "Channel payload " << numberofwords << " larger than left in total " << payloadend - currentpos;
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
        LOG(error) << "Unexpected end of payload in altro channel payload! FEE=" << mddl
                   << ", Address=0x" << std::hex << header.mHardwareAddress << ", word=0x" << currentword << std::dec;
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
        // set output cell
        //  if (fitResult == CaloRawFitter::FitStatus::kNoTime) { //Time evaluation error occured: should we add this err to list?
        //    short fec = header.mHardwareAddress >> 7 & 0xf;     //try to extract FEE number from header
        //    short branch = header.mHardwareAddress >> 11 & 0x1;
        //    if (fec > 14) {
        //      fec = kGeneralSRUErr;
        //    }
        //    fec += kGeneralTRUErr * branch;
        //    mOutputHWErrors.emplace_back(mddl, fec, 8); //8: time calculation failed
        //  }
        if (!rawFitter->isOverflow() && rawFitter->getChi2() > 0) { // Overflow is will show wrong chi2
          short chiAddr = absId;
          chiAddr |= caloFlag << 14;
          mOutputFitChi.emplace_back(chiAddr);
          mOutputFitChi.emplace_back(short(5 * rawFitter->getChi2())); // 0.2 accuracy
        }
        if (fitResult == CaloRawFitter::FitStatus::kOK || fitResult == CaloRawFitter::FitStatus::kNoTime) {
          if (!mPedestalRun) {
            if (caloFlag == Mapping::kHighGain && !rawFitter->isOverflow()) {
              currentCellContainer.emplace_back(absId, rawFitter->getAmp() - offset,
                                                (rawFitter->getTime() + starttime - bunchlength - mPreSamples) * o2::phos::PHOSSimParams::Instance().mTimeTick * 1.e-9, (ChannelType_t)caloFlag);
            }
            if (caloFlag == Mapping::kLowGain) {
              currentCellContainer.emplace_back(absId, rawFitter->getAmp() - offset,
                                                (rawFitter->getTime() + starttime - bunchlength - mPreSamples) * o2::phos::PHOSSimParams::Instance().mTimeTick * 1.e-9, (ChannelType_t)caloFlag);
            }
          } else { // pedestal, to store RMS, scale in by 1.e-7 to fit range
            currentCellContainer.emplace_back(absId, rawFitter->getAmp() - offset, 1.e-7 * rawFitter->getTime(), (ChannelType_t)caloFlag);
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
        readTRUDigits(absId, header.mPayloadSize, currentTRUContainer);
      } else {
        readTRUFlags(header.mHardwareAddress, header.mPayloadSize);
      }
    } // TRU channel
  }

  // Scan Flags and trigger cells and left only good
  // if trigger cell exists and  the trigger flag true -add it
  bool is4x4Trigger = mTRUFlags[Mapping::NTRUReadoutChannels];
  for (auto rit = currentTRUContainer.rbegin(); rit != currentTRUContainer.rend(); rit++) {
    if (mTRUFlags[rit->getTRUId()]) { // there is corresponding flag
      if (is4x4Trigger) {
        rit->setType(ChannelType_t::TRU4x4);
      } else {
        rit->setType(ChannelType_t::TRU2x2);
      }
    } else { // will be removed later
      rit->setEnergy(0);
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

void AltroDecoder::readTRUDigits(short absId, int payloadSize, std::vector<o2::phos::Cell>& truContainer)
{
  int currentsample = 0;
  while (currentsample < payloadSize) {
    int bunchlength = mBunchwords[currentsample] - 2;                           // remove words for bunchlength and starttime
    if (bunchlength < 0) {                                                      // corrupted sample: add error and ignore the reast of bunchwords
      mOutputHWErrors.emplace_back(mddl, kGeneralTRUErr, static_cast<char>(1)); // 1: wrong TRU header
      return;
    }
    int timeBin = mBunchwords[currentsample + 1];
    int istart = currentsample + 2;
    int iend = std::min((unsigned long)bunchlength, mBunchwords.size() - currentsample - 2);
    currentsample += bunchlength + 2;
    int smax = 0, tmax = 0;
    // Loop over all the time steps in the signal
    for (int i = iend - 1; i >= istart; i--) {
      if (mBunchwords[i] > smax) {
        smax = mBunchwords[i];
        tmax = timeBin;
      }
      timeBin++;
    }
    truContainer.emplace_back(absId + 14337 + 1, smax, tmax * 1.e-9, TRU2x2); // add TRU cells
  }
}
void AltroDecoder::readTRUFlags(short hwAddress, int payloadSize)
{
  // Production flags:
  // Production flags are supplied in channels 112 - 123
  // Each of the channels is 10 bit wide
  // The bits inside the channel (indexing starting from the first bit of channel 112) is as follows:
  //  Bits 0-111: Trigger flags for corresponding channel index
  //    If using 4x4 algorithm, only 91 first bits are used of these
  //  Bit 112: Marker for 4x4 algorithm (1 active, 0 not active)
  //  Bit 113: Marker for 2x2 algorithm (1 active, 0 not active)
  //  Bit 114: Global L0 OR of all patches in the TRU

  int currentsample = 0;
  while (currentsample < payloadSize) {
    int bunchlength = mBunchwords[currentsample] - 2;                           // remove words for bunchlength and starttime
                                                                                //     int timeBin = mBunchwords[currentsample + 1];
    if (bunchlength < 0) {                                                      // corrupted sample: add error and ignore the reast of bunchwords
      mOutputHWErrors.emplace_back(mddl, kGeneralTRUErr, static_cast<char>(1)); // 1: wrong TRU header
      return;
    }
    int istart = currentsample + 2;
    int iend = std::min((unsigned long)bunchlength, mBunchwords.size() - currentsample - 2);
    currentsample += bunchlength + 2;

    for (int i = iend - 1; i >= istart; i--) {
      short a = mBunchwords[i];
      // If bit 112 is 1, we are considering 4x4 algorithm
      if (hwAddress == Mapping::TRUFinalProductionChannel) {
        mTRUFlags[Mapping::NTRUReadoutChannels] = (a & (1 << 2)); // Check the bit number 112
      }
      const int kWordLength = 10; // Length of one data word in TRU raw data

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
        mTRUFlags[channel] = mTRUFlags[channel] | (a & (1 << bitIndex));
      } // Bits in one word
    }   // Length of signal
  }
}
