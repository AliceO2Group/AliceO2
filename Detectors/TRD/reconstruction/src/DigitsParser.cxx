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
                        std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, int detector, int stack, int layer, int side, DigitHCHeader& hcheader,
                        TRDFeeID& feeid, unsigned int linkindex, EventRecord* eventrecord, std::bitset<16> options, bool cleardigits)
{
  setData(data);
  mStartParse = start;
  mEndParse = end;
  mDetector = detector;
  mStack = stack;
  mLayer = layer;
  mSide = side;
  mDigitHCHeader = hcheader;
  mFEEID = feeid;
  setVerbose(options[TRDVerboseBit], options[TRDHeaderVerboseBit], options[TRDDataVerboseBit]);
  if (cleardigits) {
    clearDigits();
  }
  setByteSwap(options[TRDByteSwapBit]);
  mReturnVectorPos = 0;
  mEventRecord = eventrecord;
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

  auto timedigitparsestart = std::chrono::high_resolution_clock::now(); // measure total processing time
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
    auto looptime = std::chrono::high_resolution_clock::now() - timedigitparsestart;
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
        //LOG(warn) << "Digit end marker found but state is not StateDigitMCMData(" << StateDigitMCMData << ") or StateDigit but rather " << mState;
        // mEventRecord.ErrorStats[TRDParsingDigitEndMarkerWrongState]++;
        if (mOptions[TRDEnableRootOutputBit]) {
          mParsingErrors->Fill(TRDParsingDigitEndMarkerWrongState);
          increment2dHist(TRDParsingDigitEndMarkerWrongState);
        }
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
        //checkDigitMCMHeader(mDigitMCMHeader,lastmcmreader,lastrobread,lastevencounterread):
        if (!digitMCMHeaderSanityCheck(mDigitMCMHeader)) {
          //LOG(warn) << "***DigitMCMHeader Sanity Check Failure 0x" << std::hex << *word << " at offset " << std::distance(mStartParse, word);
          // mEventRecord.ErrorStats[TRDParsingDigitMCMHeaderSanityCheckFailure]++;
          if (mOptions[TRDEnableRootOutputBit]) {
            mParsingErrors->Fill(TRDParsingDigitMCMHeaderSanityCheckFailure);
            increment2dHist(TRDParsingDigitMCMHeaderSanityCheckFailure);
          }
          // if (mDumpUnknownData) {
          // we dump the remainig data pending better options.
          // we can try a 16 bit bitshift...
          mWordsDumped = std::distance(word, mEndParse) - 1;
          //LOG(error) << " dumping the rest of this digitparsing buffer of " << mWordsDumped;
          word = mEndParse;
          continue;
        }
        // now check mcm/rob are read in correct order
        if (mDigitMCMHeader->rob > lastrobread) {
          lastmcmread = 0;
        } else {
          if (mDigitMCMHeader->rob < lastrobread) {
            //LOG(warn) << "**DigitMCMHeader ROB number is not increasing was:" << lastrobread << " now:" << mDigitMCMHeader->rob << " 0x" << std::hex << *word << " at offset " << std::distance(mStartParse, word);
            // mEventRecord.ErrorStats[TRDParsingDigitROBDecreasing]++;
            if (mOptions[TRDEnableRootOutputBit]) {
              mParsingErrors->Fill(TRDParsingDigitROBDecreasing);
              increment2dHist(TRDParsingDigitROBDecreasing);
            }
              // we dump the remainig data pending better options.
              // we can try a 16 bit bitshift...
              mWordsDumped += std::distance(word, mEndParse) - 1;
              word = mEndParse;
          }
          //the case of rob not changing we ignore, error condition handled by mcm # increasing.
        }
        if (mDigitMCMHeader->mcm < lastmcmread && mDigitMCMHeader->rob == lastrobread) {
          //LOG(warn) << "**DigitMCMHeader MCM number is not increasing 0x" << std::hex << *word << " at offset " << std::distance(mStartParse, word);
          // mEventRecord.ErrorStats[TRDParsingDigitMCMNotIncreasing]++;
          if (mOptions[TRDEnableRootOutputBit]) {
            mParsingErrors->Fill(TRDParsingDigitMCMNotIncreasing);
            increment2dHist(TRDParsingDigitMCMNotIncreasing);
          }
          printDigitMCMHeader(*mDigitMCMHeader);
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
            //LOG(warn) << "we have a problem we have advanced from MCMHeader to the adcmask but are now at the end of the loop";
            // mEventRecord.ErrorStats[TRDParsingDigitADCMaskAdvanceToEnd]++;
            if (mOptions[TRDEnableRootOutputBit]) {
              mParsingErrors->Fill(TRDParsingDigitADCMaskAdvanceToEnd);
              increment2dHist(TRDParsingDigitADCMaskAdvanceToEnd);
            }
          }
          std::bitset<21> adcmask(mADCMask);
          bitsinmask = adcmask.count();
          //check ADCMask:
          if (!digitMCMADCMaskSanityCheck(*mDigitMCMADCMask, bitsinmask)) {
            //LOG(info) << "**DigitADCMask SANITY CHECK FAILURE " << std::hex << mDigitMCMADCMask->adcmask << " raw form : 0x" << std::hex << mDigitMCMADCMask->word << " at offset " << std::distance(mStartParse, word);
            // mEventRecord.ErrorStats[TRDParsingDigitADCMaskMismatch]++;
            if (mOptions[TRDEnableRootOutputBit]) {
              mParsingErrors->Fill(TRDParsingDigitADCMaskMismatch);
              increment2dHist(TRDParsingDigitADCMaskMismatch);
            }
            mWordsDumped += std::distance(word, mEndParse) - 1;
            // tryFindMCMHeaderAndDisplay(word);
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
          //mEventRecord.ErrorStats[TRDParsingDigitMCMHeaderBypassButStateMCMHeader]++;
          if (mOptions[TRDEnableRootOutputBit]) {
            mParsingErrors->Fill(TRDParsingDigitMCMHeaderBypassButStateMCMHeader);
            increment2dHist(TRDParsingDigitMCMHeaderBypassButStateMCMHeader);
          }
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
          //TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
        } else { // all we are left with is digitmcmdata words.
          if (mState == StateDigitEndMarker) {

            if (mHeaderVerbose) {
              LOG(info) << "***DigitEndMarker State : " << std::hex << *word << " at offset " << std::distance(mStartParse, word);
              //we are at the end
              // do nothing.
            }
            //mEventRecord.ErrorStats[TRDParsingDigitEndMarkerStateButReadingMCMADCData]++;
            if (mOptions[TRDEnableRootOutputBit]) {
              mParsingErrors->Fill(TRDParsingDigitEndMarkerStateButReadingMCMADCData);
              increment2dHist(TRDParsingDigitEndMarkerStateButReadingMCMADCData);
            }
            mDataWordsParsed += std::distance(word, mEndParse) - 1;
            word = mEndParse;
          } else {
            //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
            //for dpl build a vector and connect it with a triggerrecord.
            if (mHeaderVerbose) {
              LOG(info) << "***DigitMCMWord : " << std::hex << *word << " channel:" << std::dec << mCurrentADCChannel << " wordcount:" << mDigitWordCount << " at offset " << std::hex << std::distance(mStartParse, word);
            }
            if (mDigitWordCount == 0 || mDigitWordCount == constants::TIMEBINS / 3) {
              //new adc expected so set channel accord to bitpattern or sequential depending on zero suppressed or not.
              if (mDigitHCHeader.major & 0x20) { // zero suppressed
                //zero suppressed, so channel must be extracted from next available bit in adcmask
                mCurrentADCChannel = nextmcmadc(mADCMask, mCurrentADCChannel);
                if (mCurrentADCChannel == 21) {
                  //   LOG(warn) << "ADCMask is zero but we seem to have a digit";
                  //mEventRecord.ErrorStats[TRDParsingDigitADCChannel21]++;
                  if (mOptions[TRDEnableRootOutputBit]) {
                    mParsingErrors->Fill(TRDParsingDigitADCChannel21);
                    increment2dHist(TRDParsingDigitADCChannel21);
                  }
                }
                if (mCurrentADCChannel > 22) {
                  LOG(error) << "invalid bitpattern (read a zero) for this mcm 0x" << std::hex << mADCMask << " at offset " << std::distance(mStartParse, word);
                  if (mOptions[TRDEnableRootOutputBit]) {
                    mParsingErrors->Fill(TRDParsingDigitADCChannelGT22);
                    increment2dHist(TRDParsingDigitADCChannelGT22);
                  }
                  //mEventRecord.ErrorStats[TRDParsingDigitADCChannelGT22]++;
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
              } else { // non zero suppressed to simply increment to the next expected channel
                mCurrentADCChannel++;
              }
            }
            if (mDigitWordCount > constants::TIMEBINS / 3) {
              // LOG(error) << "***DigitMCMData with more than 10 adc's! currently on 0x" << std::hex << *word << " at offset " << std::distance(mStartParse, word);
              //mEventRecord.ErrorStats[TRDParsingDigitGT10ADCs]++;
              if (mOptions[TRDEnableRootOutputBit]) {
                mParsingErrors->Fill(TRDParsingDigitGT10ADCs);
                increment2dHist(TRDParsingDigitGT10ADCs);
              }
            }
            mDigitMCMData = (DigitMCMData*)word;
            mBufferLocation++;
            mDataWordsParsed++;
            mcmdatacount++;
            // digit sanity check
            if (!digitMCMWordSanityCheck(mDigitMCMData, mCurrentADCChannel)) {
              //mEventRecord.ErrorStats[TRDParsingDigitSanityCheck]++;
              if (mOptions[TRDEnableRootOutputBit]) {
                mParsingErrors->Fill(TRDParsingDigitSanityCheck);
                increment2dHist(TRDParsingDigitSanityCheck);
              }
              mWordsDumped++;
              mDataWordsParsed--; // we incremented it above the if loop so now transfer the count to dumped instead of parsed;
            } else {
              mState = StateDigitMCMData;
              mDigitWordCount++;
              mADCValues[digittimebinoffset++] = mDigitMCMData->z;
              mADCValues[digittimebinoffset++] = mDigitMCMData->y;
              mADCValues[digittimebinoffset++] = mDigitMCMData->x;

              if (digittimebinoffset > constants::TIMEBINS) {
                //LOG(error) << "too many timebins to insert into mADCValues digittimebinoffset:" << digittimebinoffset;
                //mEventRecord.ErrorStats[TRDParsingDigitExcessTimeBins]++;
                if (mOptions[TRDEnableRootOutputBit]) {
                  mParsingErrors->Fill(TRDParsingDigitExcessTimeBins);
                  increment2dHist(TRDParsingDigitExcessTimeBins);
                }
                //bale out TODO
                mWordsDumped += std::distance(word, mEndParse) - 1;
                //LOG(error) << " dumping the rest of this digitparsing buffer of " << mWordsDumped;
                word = mEndParse;
              }
              if (mVerbose || mDataVerbose) {
                LOG(info) << "digit word count is : " << mDigitWordCount << " digittimebinoffset = " << digittimebinoffset;
              }
              if (mDigitWordCount == constants::TIMEBINS / 3) {
                //write out adc value to vector
                //zero digittimebinoffset
                mEventRecord->getDigits().emplace_back(mDetector, mROB, mMCM, mCurrentADCChannel, mADCValues); // outgoing parsed digits
                if (mDataVerbose) {
                  LOG(info) << "DDD " << mDetector << ":" << mROB << ":" << mMCM << ":" << mCurrentADCChannel
                            << " supermodule:stack:layer:side : " << mDigitHCHeader.supermodule << ":" << mDigitHCHeader.stack << ":" << mDigitHCHeader.layer << ":" << mDigitHCHeader.side;
                }
                mDigitsFound++;
                digittimebinoffset = 0;
                mDigitWordCount = 0; // end of the digit.
                if (mDigitHCHeader.major & 0x3) {
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
    // LOG(warn) << "Exiting parsing but the state is wrong ... mState= " << mState;
    //mEventRecord.ErrorStats[TRDParsingDigitParsingExitInWrongState]++;
    if (mOptions[TRDEnableRootOutputBit]) {
      mParsingErrors->Fill(TRDParsingDigitParsingExitInWrongState);
      increment2dHist(TRDParsingDigitParsingExitInWrongState);
    }
  }
  if (std::distance(mStartParse, mEndParse) != mDataWordsParsed && mHeaderVerbose) {
    if (mHeaderVerbose) {
      LOG(info) << " we rejected " << mWordsDumped << " word and parse " << mDataWordsParsed << " % loss rate of " << (double)mWordsDumped / (double)mDataWordsParsed * 100.0;
    }
  }
  return mDataWordsParsed;
}

void DigitsParser::tryFindMCMHeaderAndDisplay(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator word)
{
  // given something has gone wrong, assume its a 16 bit shift and see if we can find a valid mcmheader,
  // note the mcm and rob and output to log
  // merely for debugging to see if we can find a pattern in where the 16 bit shifts/losses occur.
  // maybe more recoverly logic later.
  uint32_t current = *word;
  uint32_t next = *(std::next(word, 1));
  uint32_t previous = *(std::prev(word, 1));
  DigitMCMHeader firstguess; // 16 bits somewhere before the mcmheader got dropped, manifesting as directly before.
  DigitMCMHeader secondguess;
  DigitMCMHeader thirdguess;
  //first  last 16 bits of previous and first 16 bits of current
  uint32_t a = previous & 0xffff << 16;
  uint32_t b = current & 0xffff;
  firstguess.word = a + b;
  if (digitMCMHeaderSanityCheck(&firstguess)) {
    //sanity check passed to prossibly correct.
    //LOG(warn) << "***DigitMCMHeader GUESS ??? to follow";
    printDigitMCMHeader(firstguess);
  } else {
    // LOG(warn) << "***DigitMCMHeader GUESS failed " << std::hex << firstguess.word << " words were 0x" << previous << " 0x" << current << " 0x" << next;
  }
}

} // namespace o2::trd
