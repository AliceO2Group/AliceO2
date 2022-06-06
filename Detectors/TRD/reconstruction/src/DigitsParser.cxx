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

inline void DigitsParser::swapByteOrder(unsigned int& word)
{
  word = (word >> 24) |
         ((word << 8) & 0x00FF0000) |
         ((word >> 8) & 0x0000FF00) |
         (word << 24);
}
int DigitsParser::Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start,
                        std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, int detector, int stack, int layer, int side, DigitHCHeader& hcheader, uint16_t timebins,
                        TRDFeeID& feeid, unsigned int linkindex, EventRecord* eventrecord, EventStorage* eventrecords, std::bitset<16> options, bool cleardigits)
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
  setVerbose(options[TRDVerboseBit], options[TRDHeaderVerboseBit], options[TRDDataVerboseBit]);
  if (cleardigits) {
    clearDigits();
  }
  setByteSwap(options[TRDByteSwapBit]);
  mReturnVectorPos = 0;
  mEventRecord = eventrecord;
  mEventRecords = eventrecords;
  mTimeBins = timebins;
  if (mTimeBins > constants::TIMEBINS) {
    mTimeBins = constants::TIMEBINS;
    if (mMaxErrsPrinted > 0) {
      LOG(alarm) << "Time bins DigitHC Header is too large:" << timebins << " > " << constants::TIMEBINS;
      checkNoErr();
    }
  }
  return Parse();
};

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
      outputstring << "digit 0x" << std::hex << std::setfill('0') << std::setw(6) << wordcount << " :: ";
    }
    if (wordcount == 0) {
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << HelperMethods::swapByteOrderreturn(*word);
    } else {
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << HelperMethods::swapByteOrderreturn(*word);
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
  mVerbose = verbose;

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
  if (mHeaderVerbose) {
    OutputIncomingData();
  }
  int mcmdatacount = 0;
  int digittimebinoffset = 0;
  //mData holds a buffer containing digits parse placing the read digits where they need to be
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  // data starts with a DigitHCHeader, so pull that off first to simplify looping

  for (auto word = mStartParse; word < mEndParse; ++word) { // loop over the entire data buffer (a complete link of digits)
    //loop over all the words
    //check for digit end marker
    if (mByteOrderFix) {
      // byte swap if needed.
      swapByteOrder(*word);
    }
    auto nextword = std::next(word, 1);
    if ((*word) == 0x0 && (*nextword == 0x0)) { // no need to byte swap nextword
      // end of digits marker.
      if (mHeaderVerbose) {
        LOG(info) << "*** DigitEndMarker :0x" << std::hex << *word << "::0x" << *nextword << " at offset " << std::distance(mStartParse, word);
      }
      //state *should* be StateDigitMCMData check that it is
      if (mState == StateDigitMCMData || mState == StateDigitEndMarker || mState == StateDigitHCHeader || mState == StateDigitMCMHeader) {
      } else {
        incParsingError(TRDParsingDigitEndMarkerWrongState);
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
        if (mHeaderVerbose) {
          printDigitMCMHeader(*mDigitMCMHeader);
        }
        if (!sanityCheckDigitMCMHeader(mDigitMCMHeader)) {
          incParsingError(TRDParsingDigitMCMHeaderSanityCheckFailure);
          // we dump the remainig data pending better options.
          // we can try a 16 bit bitshift...
          mWordsDumped = std::distance(word, mEndParse) - 1;
          word = mEndParse;
          continue;
        }
        // now check mcm/rob are read in correct order
        if (mDigitMCMHeader->rob > lastrobread) {
          lastmcmread = 0;
        } else {
          if (mDigitMCMHeader->rob < lastrobread) {
            incParsingError(TRDParsingDigitROBDecreasing);
            // we dump the remainig data pending better options.
            // we can try a 16 bit bitshift...
            mWordsDumped += std::distance(word, mEndParse) - 1;
            word = mEndParse;
          }
          //the case of rob not changing we ignore, error condition handled by mcm # increasing.
        }
        if (mDigitMCMHeader->mcm < lastmcmread && mDigitMCMHeader->rob == lastrobread) {
          incParsingError(TRDParsingDigitMCMNotIncreasing);
        }
        lastmcmread = mDigitMCMHeader->mcm;
        lastrobread = mDigitMCMHeader->rob;
        lasteventcounterread = mDigitMCMHeader->eventcount;
        if (mDigitHCHeader.major & 0x20) {
          //zero suppressed
          //so we have an adcmask next
          std::advance(word, 1);
          if (mByteOrderFix) {
            // byte swap if needed.
            swapByteOrder(*word);
          }
          mDigitMCMADCMask = (DigitMCMADCMask*)(word);
          mADCMask = mDigitMCMADCMask->adcmask;
          if (mHeaderVerbose) {
            LOG(info) << "**DigitADCMask is " << std::hex << mDigitMCMADCMask->adcmask << " raw form : 0x" << std::hex << mDigitMCMADCMask->word << " at offset " << std::distance(mStartParse, word);
          }
          if (word == mEndParse) {
            incParsingError(TRDParsingDigitADCMaskAdvanceToEnd);
          }
          std::bitset<21> adcmask(mADCMask);
          bitsinmask = adcmask.count();
          //check ADCMask:
          if (!sanityCheckDigitMCMADCMask(*mDigitMCMADCMask, bitsinmask)) {
            incParsingError(TRDParsingDigitADCMaskMismatch);
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
          if (mHeaderVerbose) {
            LOG(info) << " we bypassed the mcmheader block but the state is MCMHeader ... 0x" << std::hex << *word << " " << lastbit << " should == 0xc";
          }
          incParsingError(TRDParsingDigitMCMHeaderBypassButStateMCMHeader);
          //dump data
          mBufferLocation++;
          mWordsDumped++;
          // carry on parsing we might find an mcm header again
        } else if (*word == o2::trd::constants::CRUPADDING32) {
          if (mHeaderVerbose) {
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

            if (mHeaderVerbose) {
              LOG(info) << "***DigitEndMarker State : " << std::hex << *word << " at offset " << std::distance(mStartParse, word);
              //we are at the end
              // do nothing.
            }
            incParsingError(TRDParsingDigitEndMarkerStateButReadingMCMADCData);
            mDataWordsParsed += std::distance(word, mEndParse) - 1;
            word = mEndParse;
          } else {
            //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
            //for dpl build a vector and connect it with a triggerrecord.
            if (mHeaderVerbose) {
              LOG(info) << "***DigitMCMWord : " << std::hex << *word << " channel:" << std::dec << mCurrentADCChannel << " wordcount:" << mDigitWordCount << " at offset " << std::hex << std::distance(mStartParse, word);
            }
            if (mDigitWordCount == 0 || mDigitWordCount == mTimeBins / 3) {
              //new adc expected so set channel accord to bitpattern or sequential depending on zero suppressed or not.
              if (mDigitHCHeader.major & 0x20) { // zero suppressed
                //zero suppressed, so channel must be extracted from next available bit in adcmask
                mCurrentADCChannel = getNextMCMADCfromBP(mADCMask, mCurrentADCChannel);

                if (mCurrentADCChannel == 21) {
                  incParsingError(TRDParsingDigitADCChannel21);
                }
                if (mCurrentADCChannel > 22) {
                  LOG(error) << "invalid bitpattern (read a zero) for this mcm 0x" << std::hex << mADCMask << " at offset " << std::distance(mStartParse, word);
                  incParsingError(TRDParsingDigitADCChannelGT22);
                  mCurrentADCChannel = 100 * bitsinmask + overchannelcount++;
                  if (mHeaderVerbose) {
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
              incParsingError(TRDParsingDigitGT10ADCs);
            }
            mDigitMCMData = (DigitMCMData*)word;
            mBufferLocation++;
            mDataWordsParsed++;
            mcmdatacount++;
            // digit sanity check
            if (!sanityCheckDigitMCMWord(mDigitMCMData, mCurrentADCChannel)) {
              incParsingError(TRDParsingDigitSanityCheck);
              mWordsDumped++;
              mDataWordsParsed--; // we incremented it above the if loop so now transfer the count to dumped instead of parsed;
            } else {
              mState = StateDigitMCMData;
              mDigitWordCount++;
              mADCValues[digittimebinoffset++] = mDigitMCMData->z;
              mADCValues[digittimebinoffset++] = mDigitMCMData->y;
              mADCValues[digittimebinoffset++] = mDigitMCMData->x;

              if (digittimebinoffset > mTimeBins) {
                incParsingError(TRDParsingDigitSanityCheck);
                //bale out TODO
                mWordsDumped += std::distance(word, mEndParse) - 1;
                word = mEndParse;
              }
              if (mVerbose || mDataVerbose) {
                LOG(info) << "digit word count is : " << mDigitWordCount << " digittimebinoffset = " << digittimebinoffset;
              }
              if (mDigitWordCount == mTimeBins / 3) {
                //write out adc value to vector
                //zero digittimebinoffset
                mEventRecord->getDigits().emplace_back(mDetector, mROB, mMCM, mCurrentADCChannel, mADCValues); // outgoing parsed digits
                mEventRecord->incDigitsFound(1);
                //mEventRecords->incMCMDigitCount(mDetector, mROB, mMCM, 1);
                if (mDataVerbose) {
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
    incParsingError(TRDParsingDigitParsingExitInWrongState);
  }
  if (std::distance(mStartParse, mEndParse) != mDataWordsParsed && mHeaderVerbose) {
    if (mHeaderVerbose) {
      LOG(info) << " we rejected " << mWordsDumped << " word and parse " << mDataWordsParsed << " % loss rate of " << (double)mWordsDumped / (double)mDataWordsParsed * 100.0;
    }
  }
  return mDataWordsParsed;
}

void DigitsParser::checkNoErr()
{
  if (!mVerbose && --mMaxErrsPrinted == 0) {
    LOG(error) << "Errors limit reached, the following ones will be suppressed";
  }
}

} // namespace o2::trd
