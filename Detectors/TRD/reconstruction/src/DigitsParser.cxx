// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitsParser.h
/// @brief  TRD raw data parser for digits

#include "TRDReconstruction/DigitsParser.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/CompressedDigit.h"

#include "fairlogger/Logger.h"

//TODO come back and figure which of below headers I actually need.
#include <cstring> //memcpy
#include <string>
#include <vector>
#include <array>
#include <iterator>

namespace o2::trd
{

inline void DigitsParser::swapByteOrder(unsigned int& word)
{
  word = (word >> 24) |
         ((word << 8) & 0x00FF0000) |
         ((word >> 8) & 0x0000FF00) |
         (word << 24);
}

int DigitsParser::Parse(bool verbose)
{

  //we are handed the buffer payload of an rdh and need to parse its contents.
  //producing a vector of digits.
  mVerbose = verbose;
  //  mDataVerbose = true;
  //  mHeaderVerbose = true;

  mState = StateDigitHCHeader;
  mDataWordsParsed = 0; // count of data wordsin data that have been parsed in current call to parse.
  mDigitsFound = 0;     // tracklets found in the data block, mostly used for debugging.
  mBufferLocation = 0;
  mPaddingWordsCounter = 0;
  if (mVerbose) {
    LOG(info) << "Digit Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at pos " << mStartParse;
    if (mDisableByteOrderFix) {

      LOG(info) << " we will not be byte swapping";
    } else {

      LOG(info) << " we will be byte swapping";
    }
  }
  if (mDataVerbose) {
    LOG(info) << "trackletdata to parse begin";
    std::vector<uint32_t> datacopy(mStartParse, mEndParse);
    if (!mDisableByteOrderFix) {
      for (auto a : datacopy) {
        swapByteOrder(a);
      }
    }

    LOG(info) << "digitdata to parse with size of " << datacopy.size();
    int loopsize = 0;
    if (datacopy.size() > 1024) {
      loopsize = 64;
    }
    for (int i = 0; i < loopsize; i += 8) {
      LOG(info) << std::hex << "0x" << datacopy[i] << " " << std::hex << "0x" << datacopy[i + 1] << " " << std::hex << "0x" << datacopy[i + 2] << " " << std::hex << "0x" << datacopy[i + 3] << " " << std::hex << "0x" << datacopy[i + 4] << " " << std::hex << "0x" << datacopy[i + 5] << " " << std::hex << "0x" << datacopy[i + 6] << " " << std::hex << "0x" << datacopy[i + 7];
    }
    LOG(info) << "digitdata to parse end";
    if (datacopy.size() > 1024) {
      LOG(fatal) << "some very wrong with digit parsing >1024";
    }
  }
  int mcmdatacount = 0;
  int mcmadccount = 0;
  int digitwordcount = 9;
  int digittimebinoffset = 0;
  if (mVerbose) {
    LOG(info) << "Digits Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at " << std::distance(mData->begin(), mStartParse) << " ending at " << std::distance(mData->begin(), mEndParse);
    LOG(info) << "Digits Parser parse " << std::distance(mStartParse, mEndParse) << " items for digits should be 21*11+header";
    LOG(info) << "word to parse : " << std::hex << *mStartParse << "and " << *(mStartParse + 1) << " in state :" << mState;
  }
  //mData holds a buffer containing digits parse placing the read digits where they need to be
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  // data starts with a DigitHCHeader, so pull that off first to simplify looping
  if (mState == StateDigitHCHeader) {
    if (mVerbose) {
      LOG(info) << "at start of data";
    }
    mDigitHCHeader = (DigitHCHeader*)mStartParse;
    if (!mDisableByteOrderFix) {
      // byte swap if needed.
      swapByteOrder(mDigitHCHeader->word0);
      swapByteOrder(mDigitHCHeader->word1);
    }
    if (mVerbose) {
      LOG(info) << mDigitHCHeader->bunchcrossing << " was bunchcrossing and " << mDigitHCHeader->supermodule << " " << mDigitHCHeader->layer;
    }
    if (mHeaderVerbose) {
      printDigitHCHeader(*mDigitHCHeader);
    }
    mState = StateDigitMCMHeader;
    mBufferLocation += 2;
    mDataWordsParsed += 2;
    std::advance(mStartParse, 2);
    //move over the DigitHCHeader;
  }
  mState = StateDigitHCHeader;
  for (auto word = mStartParse; word != mEndParse; word++) { // loop over the entire data buffer (a complete link of digits)
    //loop over all the words
    if (mDataVerbose || mVerbose) {
      LOG(info) << "parsing word : " << std::hex << *word;
    }
    //check for digit end marker
    if (!mDisableByteOrderFix) {
      // byte swap if needed.
      swapByteOrder(*word);
    }
    auto nextword = std::next(word, 1);
    if ((*word) == 0x0 && (*nextword == 0x0)) { // no need to byte swap nextword
      // end of digits marker.
      if (mVerbose || mHeaderVerbose || mDataVerbose) {
        LOG(info) << "Found digits end marker :" << std::hex << *word << "::" << *nextword;
      }
      //state *should* be StateDigitMCMData check that it is
      if (mState == StateDigitMCMData || mState == StateDigitEndMarker || mState == StateDigitHCHeader) {
      } else {

        LOG(fatal) << "Digit end marker found but state is not StateDigitMCMData(" << StateDigitMCMData << ") or StateDigitbut rather " << mState;
      }
      //only thing that can remain is the padding.
      //now read padding words till end.
      //no need to byteswap its uniform.
      mBufferLocation += 2; // 2 words forward.
      mDataWordsParsed += 2;
      mState = StateDigitEndMarker;
    } else {
      if ((*word & 0xf) == 0xc && mState == StateDigitMCMHeader) { //marker for DigitMCMHeader.
        if (mVerbose) {
          LOG(info) << " **** mDigitMCMHeader has value " << std::hex << *word;
        }
        //read the header OR padding of 0xeeee;
        //we actually have an header word.
        mcmadccount = 0;
        mcmdatacount = 0;
        mDigitMCMHeader = (DigitMCMHeader*)(word);
        //if (mVerbose)
        if (mVerbose || mHeaderVerbose) {
          LOG(info) << "state mcmheader and word : 0x" << std::hex << *word;
          printDigitMCMHeader(*mDigitMCMHeader);
        }
        //sanity check of digitheader ??  Still to implement.
        if (mHeaderVerbose) {
          std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator tmpword;

          if (!digitMCMHeaderSanityCheck(mDigitMCMHeader)) {
            LOG(warn) << "Sanity check Failure Digit MCMHeader : " << std::hex << mDigitMCMHeader->word;
            LOG(warn) << "Sanity check Failure Digit MCMHeader word: " << std::hex << *word;
            DigitMCMHeader* t;
            tmpword = std::prev(word, 3);
            printDigitMCMHeader(*(DigitMCMHeader*)(tmpword));
            tmpword = std::prev(word, 2);
            printDigitMCMHeader(*(DigitMCMHeader*)(tmpword));
            tmpword = std::prev(word, 1);
            printDigitMCMHeader(*(DigitMCMHeader*)(tmpword));
            printDigitMCMHeader(*mDigitMCMHeader);
            tmpword = std::next(word, 1);
            printDigitMCMHeader(*(DigitMCMHeader*)(tmpword));
            tmpword = std::next(word, 2);
            printDigitMCMHeader(*(DigitMCMHeader*)(tmpword));
            tmpword = std::next(word, 3);
            printDigitMCMHeader(*(DigitMCMHeader*)(tmpword));
            LOG(warn) << "Sanity check Failure Digit MCMHeader print out finished";
          } else {
            LOG(info) << "Sanity check passed for digitmcmheader";
            printDigitMCMHeader(*mDigitMCMHeader);
          }
        }
        mBufferLocation++;
        mState = StateDigitHCHeader;
        //new header so digit word count becomes zero
        digitwordcount = 0;
        mMCM = mDigitMCMHeader->mcm;
        mROB = mDigitMCMHeader->rob;
        mEventCounter = mDigitMCMHeader->eventcount;
        mDataWordsParsed++;
        mChannel = 0;
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
        if (*word == o2::trd::constants::CRUPADDING32) {
          if (mVerbose) {
            LOG(info) << "state padding and word : 0x" << std::hex << *word;
          }
          //another pointer with padding.
          mBufferLocation++;
          mPaddingWordsCounter++;
          mState = StatePadding;
          mDataWordsParsed++;
          mcmdatacount = 0;

          // this is supposed to carry on till the end of the buffer, hence the term padding.
          //TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
        } else { // all we are left with is digitmcmdata words.
          if (mState == StateDigitEndMarker) {

            //we are at the end
            // do nothing.
            if (mDataVerbose) {
              LOG(info) << " digit end marker state ...";
            }
          } else {
            if (mVerbose || mDataVerbose) {
              LOG(info) << "mDigitMCMData with state=" << mState << " is at " << mBufferLocation << " had value 0x" << std::hex << *word << " mcmdatacount of : " << mcmdatacount << " adc#" << mcmadccount;
            }
            //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
            //for dpl build a vector and connect it with a triggerrecord.
            mDataWordsParsed++;
            mcmdatacount++;
            if (mReturnVector) { // we will generate a vector
              mDigitMCMData = (DigitMCMData*)word;
              mBufferLocation++;
              mState = StateDigitMCMData;
              digitwordcount++;
              if (mVerbose || mDataVerbose) {
                LOG(info) << "adc values : " << mDigitMCMData->x << "::" << mDigitMCMData->y << "::" << mDigitMCMData->z;
                LOG(info) << "digittimebinoffset = " << digittimebinoffset;
              }
              mADCValues[digittimebinoffset] = mDigitMCMData->x;
              mADCValues[digittimebinoffset++] = mDigitMCMData->y;
              mADCValues[digittimebinoffset++] = mDigitMCMData->z;
              if (mVerbose || mDataVerbose) {
                LOG(info) << "digit word count is : " << digitwordcount;
              }
              if (digitwordcount == constants::TIMEBINS / 3) {
                //sanity check, next word shouldbe either a. end of digit marker, digitMCMHeader,or padding.
                if (mSanityCheck) {
                  uint32_t* tmp = std::next(word);
                  if (mDataVerbose) {
                    LOG(info) << "digitwordcount = " << digitwordcount << " hopefully the next data is digitendmarker, didgitMCMHeader or padding 0x" << std::hex << *tmp;
                  }
                }
                if (mDataVerbose) {
                  LOG(info) << "change of adc";
                }
                mcmadccount++;
                //write out adc value to vector
                //zero digittimebinoffset
                mDigits.emplace_back(mDetector, mROB, mMCM, mChannel, mADCValues); // outgoing parsed digits
                digittimebinoffset = 0;
                digitwordcount = 0; // end of the digit.
                mChannel++;
              }

            } else { //version 2 will have below, it will be quicker not to have the intermediary step, but is it really needed?
                     //returning digits raw, as its pretty much the most compressed you are going to get in anycase.
                     // we will send the raw stream back. "compressed"
                     // memcpy((char*)&(*mData)[mReturnVectorPos], (void*)word, sizeof(uint32_t));
                     //TODO or should we just copy all timebins at the same time?
                     //
            }
          } //end state endmarker
        }
      }
    }

    //accounting ....
    // mCurrentLinkDataPosition256++;
    // mCurrentHalfCRUDataPosition256++;
    // mTotalHalfCRUDataLength++;
    //end of data so
    if (word == mEndParse) {
      if (mVerbose) {
        LOG(info) << "word is mEndParse";
      }
    }
    if (std::distance(word, mEndParse) < 0) {
      if (mVerbose || mDataVerbose || mHeaderVerbose) {
        LOG(info) << std::dec << "word to mEndParse is :" << std::distance(word, mEndParse);
      }
    }
  }
  if (mVerbose) {
    LOG(info) << "***** parsing loop finished for this link";
  }

  if (!(mState == StateDigitMCMHeader || mState == StatePadding || mState == StateDigitEndMarker)) {
    LOG(warn) << "Exiting parsing but the state is wrong ... mState= " << mState;
    if (mVerbose) {
      LOG(info) << "Exiting parsing but the state is wrong ... mState= " << mState;
    }
  }
  return mDataWordsParsed;
}

} // namespace o2::trd
