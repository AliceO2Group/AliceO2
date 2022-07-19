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

/// @file   DigitsParser.h
/// @brief  TRD raw data parser for digits

#include "TRDReconstruction/DigitsParser.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/RawDataStats.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/CompressedDigit.h"
#include "DataFormatsTRD/Digit.h"
#include "TRDReconstruction/EventRecord.h"
#include "fairlogger/Logger.h"

//TODO come back and figure which of below headers I actually need.
#include <cstring> //memcpy
#include <string>
#include <vector>
#include <array>
#include <iterator>
#include <bitset>
#include <iomanip>
#include <iostream>

namespace o2::trd
{

int DigitsParser::Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start,
                        std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, int detector, int stack, int layer, int side, DigitHCHeader& hcheader, uint16_t timebins,
                        TRDFeeID& feeid, unsigned int linkindex, EventRecord* eventrecord, EventStorage* eventrecords, std::bitset<16> options)
{
  setData(data);
  mStartParse = start;
  mEndParse = end;
  mDetector = detector;
  mStack = stack;
  mLayer = layer;
  mHalfChamberSide = side;
  mStackLayer = mStack * constants::NLAYER + mLayer;
  mDigitHCHeader = hcheader;
  mFEEID = feeid;
  mReturnVectorPos = 0;
  mEventRecord = eventrecord;
  mEventRecords = eventrecords;
  mTimeBins = timebins;
  if (mTimeBins > constants::TIMEBINS) {
    mTimeBins = constants::TIMEBINS;
    LOG(warn) << "Time bins DigitHC Header is too large:" << timebins << " > " << constants::TIMEBINS;
  }
  return Parse();
};

void DigitsParser::incParsingError(int error, std::string message)
{
  int sector = mFEEID.supermodule;
  int stack = mStack;
  int layer = mLayer;
  int side = mHalfChamberSide;
  int hcid = sector * constants::NCHAMBER * constants::NSTACK * constants::NLAYER + stack * constants::NSTACK * constants::NLAYER + layer + side;

  if (mOptions[TRDGenerateStats]) {
    mEventRecords->incParsingError(error, hcid);
  }
  if (mOptions[TRDVerboseErrorsBit]) {
    LOG(info) << "PE Parsing Error : " << o2::trd::ParsingErrorsString[error] << " sector:stack:layer:side :: " << mFEEID.supermodule << ":" << mFEEID.side << ":" << mStack << ":" << mLayer;
    LOG(info) << message;
  }
  if (mOptions[TRDVerboseLinkBit]) {
    mDumpLink = true;
  }
}

void DigitsParser::OutputIncomingData()
{

  LOG(info) << "Data buffer to parse for Digits begin " << mStartParse << ":" << mEndParse;
  int wordcount = 0;
  std::stringstream outputstring;
  auto word = mStartParse;
  outputstring << "digit 0x" << std::hex << std::setfill('0') << std::setw(6) << 0 << " :: ";
  while (word <= mEndParse) { // loop over the entire data buffer (a complete link of tracklets and digits)

    if (wordcount != 0 && (wordcount % 8 == 0 || word == mEndParse)) {
      LOG(info) << outputstring.str();
      outputstring.str("");
      outputstring << "digit 0x" << std::hex << std::setfill('0') << std::setw(6) << wordcount;
    }
    if (wordcount == 0) {
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << *word;
    } else {
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << *word;
    }
    word++;
    wordcount++;
  }
  LOG(info) << "Data buffer to parse for Digits end";
}

int DigitsParser::Parse(bool verbose)
{

  //we are handed the buffer payload of an rdh and need to parse its contents.
  //producing a vector of digits.
  mOptions[TRDVerboseBit] = verbose;

  mState = StateDigitMCMHeader;
  mDataWordsParsed = 0; // count of data wordsin data that have been parsed in current call to parse.
  mWordsDumped = 0;
  mDigitsFound = 0;     // tracklets found in the data block, mostly used for debugging.
  mBufferLocation = 0;
  mPaddingWordsCounter = 0;
  int bitsinmask = 0;
  int overchannelcount = 0;
  int lastmcmread = 0;
  int lastrobread = 0;
  int lasteventcounterread = 0;
  int mcmdatacount = 0;
  int digittimebinoffset = 0;
  //mData holds a buffer containing digits parse placing the read digits where they need to be
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  // data starts with a DigitHCHeader, so pull that off first to simplify looping

  for (auto word = mStartParse; word < mEndParse; ++word) { // loop over the entire data buffer (a complete link of digits)
    //loop over all the words
    //check for digit end marker
    if (mOptions[TRDByteSwapBit]) {
      // byte swap if needed.
      HelperMethods::swapByteOrder(*word);
    }

    if (mOptions[TRDVerboseWordBit]) {
      LOGF(info, "parsing word:0x%08x\n", *word);
    }
    auto nextword = std::next(word, 1);
    if ((*word) == 0x0 && (*nextword == 0x0)) { // no need to byte swap nextword
      // end of digits marker.
      //state *should* be StateDigitMCMData check that it is
      if (mState == StateDigitMCMData || mState == StateDigitEndMarker || mState == StateDigitHCHeader || mState == StateDigitMCMHeader) {
      } else {
        incParsingError(DigitEndMarkerWrongState, fmt::format("Wrong state word : {:#x}", *word));
      }
      //only thing that can remain is the padding.
      //now read padding words till end.
      //no need to byteswap its uniform.
      mBufferLocation += 2; // 2 words forward.
      mDataWordsParsed += 2;
      mState = StateDigitEndMarker;
      std::advance(word, 1); // advance over the second word of the endmarker
      if (word < mEndParse) {
        mWordsDumped += std::distance(word, mEndParse) - 1;
      }
      word = mEndParse;
    } else {
      if ((*word & 0xf) == 0xc && (mState == StateDigitMCMHeader || mState == StateDigitMCMData)) { //marker for DigitMCMHeader.
        //read the header
        mcmdatacount = 0;
        mCurrentADCChannel = 0;
        mDigitMCMHeader = (DigitMCMHeader*)(word);
        if (mOptions[TRDVerboseWordBit]) {
          printDigitMCMHeader(*mDigitMCMHeader);
        }
        if (!sanityCheckDigitMCMHeader(mDigitMCMHeader)) {
          incParsingError(DigitMCMHeaderSanityCheckFailure, fmt::format("sanity Check failure on Digit MCM Header"));
          if (mOptions[TRDVerboseErrorsBit]) {
            printDigitMCMHeader(*mDigitMCMHeader);
          }
          // we dump the remainig data pending better options.
          mWordsDumped = std::distance(word, mEndParse) - 1;
          word = mEndParse;
          continue;
        }
        // now check mcm/rob are read in correct order
        if (mDigitMCMHeader->rob > lastrobread) {
          lastmcmread = 0;
        } else {
          if (mDigitMCMHeader->rob < lastrobread) {
            incParsingError(DigitROBDecreasing, fmt::format("current rob less than previous one : {} last one was: {}", (int)mDigitMCMHeader->rob, lastrobread));
            if (mOptions[TRDVerboseErrorsBit]) {
              DigitMCMHeader a;
              a.word = *word;
              printDigitMCMHeader(a);
            }
            // we dump the remainig data pending better options.
            mWordsDumped += std::distance(word, mEndParse) - 1;
            word = mEndParse;
          }
          //the case of rob not changing we ignore, error condition handled by mcm # increasing.
        }
        if (mDigitMCMHeader->mcm < lastmcmread && mDigitMCMHeader->rob == lastrobread) {
          incParsingError(DigitMCMNotIncreasing, fmt::format("current mcm less than previous one : {} last one was {}", (int)mDigitMCMHeader->mcm, lastmcmread));
          if (mOptions[TRDVerboseErrorsBit]) {
            DigitMCMHeader a;
            a.word = *word;
            printDigitMCMHeader(a);
          }
        }
        lastmcmread = mDigitMCMHeader->mcm;
        lastrobread = mDigitMCMHeader->rob;
        lasteventcounterread = mDigitMCMHeader->eventcount;
        if (mDigitHCHeader.major & 0x20) {
          //zero suppressed
          //so we have an adcmask next
          std::advance(word, 1);
          if (mOptions[TRDByteSwapBit]) {
            // byte swap if needed.
            HelperMethods::swapByteOrder(*word);
          }
          mDigitMCMADCMask = (DigitMCMADCMask*)(word);
          mADCMask = mDigitMCMADCMask->adcmask;
          if (mOptions[TRDVerboseWordBit]) {
            DigitMCMADCMask a;
            a.word = *word;
            printDigitMCMADCMask(a);
          }
          if (word == mEndParse) {
            incParsingError(DigitADCMaskAdvanceToEnd, fmt::format("we seem to be at the end of the buffer to parse with word of : {:#08x}", *word));
          }
          std::bitset<21> adcmask(mADCMask);
          bitsinmask = adcmask.count();
          //check ADCMask:
          if (!sanityCheckDigitMCMADCMask(*mDigitMCMADCMask, bitsinmask)) {
            incParsingError(DigitADCMaskMismatch, fmt::format(" adcmask of {:#08x} does not match bitsinmask: {}", (uint32_t)mDigitMCMADCMask->word, bitsinmask));
            if (mOptions[TRDVerboseErrorsBit]) {
              DigitMCMADCMask a;
              a.word = *word;
              printDigitMCMADCMask(a);
            }
            mWordsDumped += std::distance(word, mEndParse) - 1;
            word = mEndParse;
          }
          overchannelcount = 0;
          //output all the adc data for the described adc mask, i.e 10 32 bit words per bit in mask.
        }
        mBufferLocation++;
        //new header so digit word count becomes zero
        mDigitWordCount = 0;
        mState = StateDigitMCMData;
        mMCM = mDigitMCMHeader->mcm;
        mROB = mDigitMCMHeader->rob;
        //TOOD does it match the feeid which ncodes this information as well.
        //
        mEventCounter = mDigitMCMHeader->eventcount;
        mDataWordsParsed++; // header
        if (mDigitHCHeader.major & 0x20) {
          //zero suppressed digits
          mDataWordsParsed++; // adc mask
        }
        mCurrentADCChannel = 0;
        mADCValues.fill(0);
        digittimebinoffset = 0;
        if (!mReturnVector) {
          //returning the raw "compressed" data stream.
          //build the digit header and add to outgoing buffer;
          uint32_t* header = &(*mData)[mReturnVectorPos];
          //build header.
        }
        // we dont care about the year flag, we are >2007 already.
      } else {
        if (mState == StateDigitMCMHeader) { //safety check for some weird data occurances
          unsigned int lastbit = (*word) & 0xf;
          incParsingError(DigitMCMHeaderBypassButStateMCMHeader, fmt::format(" we bypassed the mcmheader block but the state is MCMHeader ... {:#08x} lastbit should be 0xc and is {:#x}", *word, lastbit));
          if (mOptions[TRDVerboseErrorsBit]) {
            DigitMCMADCMask a;
            a.word = *word;
            printDigitMCMADCMask(a);
          }
          mWordsDumped += std::distance(word, mEndParse) - 1;
          //dump data
          mBufferLocation++;
          mWordsDumped++;
          // carry on parsing we might find an mcm header again
        } else if (*word == o2::trd::constants::CRUPADDING32) {
          if (mOptions[TRDVerboseBit]) {
            LOG(info) << "***CRUPADDING32 word : 0x" << std::hex << *word << "  state is:" << mState;
          }
          //another pointer with padding.
          mBufferLocation++;
          mPaddingWordsCounter++;
          mState = StatePadding;
          // mDataWordsParsed++;
          mDataWordsParsed += std::distance(word, mEndParse);
          //dump the rest of the received buffer its the rdh with e's
          word = mEndParse;
          mcmdatacount = 0;
          // this is supposed to carry on till the end of the buffer, hence the term padding.
        } else { // all we are left with is digitmcmdata words.
          if (mState == StateDigitEndMarker) {

            if (mOptions[TRDVerboseBit]) {
              LOG(info) << "***DigitEndMarker State : " << std::hex << *word << " at offset " << std::distance(mStartParse, word);
              //we are at the end
              // do nothing.
            }
            incParsingError(DigitEndMarkerStateButReadingMCMADCData, fmt::format("State is endmarker but we are reading mcm adc data : {:#08x}", *word));
            if (mOptions[TRDVerboseErrorsBit]) {
              DigitMCMData a;
              a.word = *word;
              printDigitMCMData(a);
            }
            mDataWordsParsed += std::distance(word, mEndParse) - 1;
            word = mEndParse;
          } else {
            //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
            //for dpl build a vector and connect it with a triggerrecord.
            if (mOptions[TRDVerboseBit]) {
              LOG(info) << "***DigitMCMWord : " << std::hex << *word << " channel:" << std::dec << mCurrentADCChannel << " wordcount:" << mDigitWordCount << " at offset " << std::hex << std::distance(mStartParse, word);
            }
            if (mDigitWordCount == 0 || mDigitWordCount == mTimeBins / 3) {
              //new adc expected so set channel accord to bitpattern or sequential depending on zero suppressed or not.
              if (mDigitHCHeader.major & 0x20) { // zero suppressed
                //zero suppressed, so channel must be extracted from next available bit in adcmask
                mCurrentADCChannel = getNextMCMADCfromBP(mADCMask, mCurrentADCChannel);

                if (mCurrentADCChannel == 21) {
                  incParsingError(DigitADCChannel21);
                  if (mOptions[TRDVerboseErrorsBit]) {
                    DigitMCMData a;
                    a.word = *word;
                    printDigitMCMData(a);
                  }
                }
                if (mCurrentADCChannel > 22) {
                  incParsingError(DigitADCChannelGT22, fmt::format("invalid bitpattern (read a zero) for this mcm {:#08x} at offset {}", mADCMask, std::distance(mStartParse, word)));
                  if (mOptions[TRDVerboseErrorsBit]) {
                    DigitMCMData a;
                    a.word = *word;
                    printDigitMCMData(a);
                  }
                  mCurrentADCChannel = 100 * bitsinmask + overchannelcount++;
                  if (mOptions[TRDVerboseBit]) {
                    LOG(info) << "EEE " << mDetector << ":" << mROB << ":" << mMCM << ":" << mCurrentADCChannel
                              << " supermodule:stack:layer:side : " << mDigitHCHeader.supermodule << ":" << mDigitHCHeader.stack << ":" << mDigitHCHeader.layer << ":" << mDigitHCHeader.side;
                  }
                }
                if (mADCMask == 0) {
                  //no more adc for zero suppression.
                  //now we should either have another MCMHeader, or End marker
                  if (*word != 0 && *(std::next(word)) != 0) { // end marker is a sequence of 32 bit 2 zeros.
                    mState = StateDigitMCMHeader;
                  } else {
                    mState = StateDigitEndMarker;
                  }
                }
              }
            }
            if (mDigitWordCount > mTimeBins / 3) {
              incParsingError(DigitGT10ADCs);
              if (mOptions[TRDVerboseErrorsBit]) {
                DigitMCMData a;
                a.word = *word;
                printDigitMCMData(a);
              }
            }
            mDigitMCMData = (DigitMCMData*)word;
            if (mOptions[TRDVerboseWordBit]) {
              DigitMCMData a;
              a.word = *word;
              printDigitMCMData(a);
            }
            mBufferLocation++;
            mDataWordsParsed++;
            mcmdatacount++;
            // digit sanity check
            if (!sanityCheckDigitMCMWord(mDigitMCMData, mCurrentADCChannel)) {
              incParsingError(DigitSanityCheck);
              if (mOptions[TRDVerboseErrorsBit]) {
                DigitMCMData a;
                a.word = *word;
                printDigitMCMData(a);
              }
              mWordsDumped++;
              mDataWordsParsed--; // we incremented it above the if loop so now transfer the count to dumped instead of parsed;
            } else {
              mState = StateDigitMCMData;
              mDigitWordCount++;
              mADCValues[digittimebinoffset++] = mDigitMCMData->z;
              mADCValues[digittimebinoffset++] = mDigitMCMData->y;
              mADCValues[digittimebinoffset++] = mDigitMCMData->x;

              if (digittimebinoffset > mTimeBins) {
                incParsingError(DigitSanityCheck);
                if (mOptions[TRDVerboseErrorsBit]) {
                  LOG(info) << "Parsing Digit Sanity Check digittimebinoffset> mTimeBins : "
                            << digittimebinoffset << " > " << mTimeBins << " reading word 0x" << std::hex << *word;
                }
                //bale out TODO
                mWordsDumped += std::distance(word, mEndParse) - 1;
                word = mEndParse;
              }
              if (mOptions[TRDVerboseBit]) {
                LOG(info) << "digit word count is : " << mDigitWordCount << " digittimebinoffset = " << digittimebinoffset;
              }
              if (mDigitWordCount == mTimeBins / 3) {
                //write out adc value to vector
                //zero digittimebinoffset
                mEventRecord->getDigits().emplace_back(mDetector, mROB, mMCM, mCurrentADCChannel, mADCValues); // outgoing parsed digits
                mEventRecord->incDigitsFound(1);
                //mEventRecords->incMCMDigitCount(mDetector, mROB, mMCM, 1);
                if (mOptions[TRDVerboseBit]) {
                  LOG(info) << "DDD " << mDetector << ":" << mROB << ":" << mMCM << ":" << mCurrentADCChannel
                            << " supermodule:stack:layer:side : " << mDigitHCHeader.supermodule << ":" << mDigitHCHeader.stack << ":" << mDigitHCHeader.layer << ":" << mDigitHCHeader.side;
                }
                mDigitsFound++;
                digittimebinoffset = 0;
                mDigitWordCount = 0; // end of the digit.
                if (!(mDigitHCHeader.major & 0x2)) {
                  mCurrentADCChannel++; // we count channels as all 21 channels are present, no way to check this.
                }
              } // mDigitWordCount == timebins/3
            }
          }   //else digitendmarker if statement
        }     // else of if crupadding word
      }       // else of if digitMCMheader
    }         // else of if digitendmarker
    //accounting
    // mCurrentLinkDataPosition256++;
    // mCurrentHalfCRUDataPosition256++;
    // mTotalHalfCRUDataLength++;
    //end of data so
  } // for loop over word
  if (!(mState == StateDigitMCMHeader || mState == StatePadding || mState == StateDigitEndMarker)) {
    incParsingError(DigitParsingExitInWrongState);
  }
  if (std::distance(mStartParse, mEndParse) != mDataWordsParsed && mOptions[TRDVerboseBit]) {
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << " we rejected " << mWordsDumped << " word and parse " << mDataWordsParsed << " % loss rate of " << (double)mWordsDumped / (double)mDataWordsParsed * 100.0;
    }
  }
  return mDataWordsParsed;
}


} // namespace o2::trd
