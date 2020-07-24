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

namespace o2
{
namespace trd
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
  mState = StateDigitHCHeader;
  mDataWordsParsed = 0; // count of data wordsin data that have been parsed in current call to parse.
  mDigitsFound = 0;     // tracklets found in the data block, mostly used for debugging.
  mBufferLocation = 0;
  mPaddingWordsCounter = 0;
  int mcmdatacount=0;
  int mcmadccount=0;
  int digitwordcount = 9;
  int digittimebinoffset = 0;
  if (mVerbose) LOG(info) << "Digits Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at " << std::distance(mData->begin(), mStartParse) << " ending at " << std::distance(mData->begin(), mEndParse);
  if (mVerbose) LOG(info) << "Digits Parser parse " << std::distance(mStartParse,mEndParse) << " items for digits should be 21*11+header";
  //mData holds a buffer containing digits parse placing the read digits where they need to be
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  // data starts with a DigitHCHeader, so pull that off first to simplify looping
  if (mVerbose) LOG(info) << "word to parse : " << std::hex << *mStartParse << "and " << *(mStartParse + 1) << " in state :" << mState;
  if (mState == StateDigitHCHeader) {
    if (mVerbose) LOG(info) << "at start of data";
    mDigitHCHeader = (DigitHCHeader*)mStartParse;
    if (!mDisableByteOrderFix) {
      // byte swap if needed.
      swapByteOrder(mDigitHCHeader->word0);
      swapByteOrder(mDigitHCHeader->word1);
    }
    if (mVerbose) LOG(info) << mDigitHCHeader->bunchcrossing << " was bunchcrossing and " << mDigitHCHeader->supermodule << " " << mDigitHCHeader->layer;
    //    printDigitHCHeader(*mDigitHCHeader);
    mState = StateDigitMCMHeader;
    mBufferLocation += 2;
    mDataWordsParsed+=2;
    std::advance(mStartParse,2);
    //move over the DigitHCHeader;
  }
  mState=StateDigitHCHeader;
  for (auto word = mStartParse; word != mEndParse; word++) { // loop over the entire data buffer (a complete link of digits)
    //if(mVerbose) LOG(info) << "word to parse : " << std::hex << *word << " in state :" << mState;
    //if(mVerbose) LOG(info) <<std::dec <<  "word to mEndParse is :"<< std::distance(word,mEndParse);
    //if(mVerbose) LOG(info) <<std::dec <<  "mStartParse to mEndParse is :"<< std::distance(mStartParse,mEndParse);
   // LOG(info) <<std::dec <<  "word to mEndParse is :"<< std::distance(word,mEndParse);
    //loop over all the words
    if (mVerbose) LOG(info) << "parsing word : " << std::hex << *word;
    //check for digit end marker
    if (!mDisableByteOrderFix) {
      // byte swap if needed.
      swapByteOrder(*word);
    }
    auto nextword = std::next(word, 1);
    if ((*word) == 0x0 && (*nextword == 0x0)) { // no need to byte swap nextword as zero does not change.
      // end of digits marker.
      if (mVerbose) LOG(info) << "Found digits end marker :" << std::hex << *word << "::" << *nextword;
      //state *should* be StateDigitMCMData check that it is
      if (mState != StateDigitMCMData || mState != StateDigitEndMarker) {
        LOG(fatal) << "Digit end marker found but state is not StateDigitMCMData(" << StateDigitMCMData << ") or StateDigitbut rather " << mState;
      }
      //only thing that can remain is the padding.
      //now read padding words till end.
      //no need to byteswap its uniform.
      mBufferLocation+=2; // 2 words forward.
      mDataWordsParsed+=2;
      mState = StateDigitEndMarker;
    } else {
      if ((*word & 0xf) == 0xc) { //marker for DigitMCMHeader.
        if (mVerbose) LOG(info) << " **** mDigitMCMHeader has value " << std::hex << *word;
        //read the header OR padding of 0xeeee;
        //we actually have an header word.
        mcmadccount=0;
        mcmdatacount=0;
        mDigitMCMHeader = (DigitMCMHeader*)(word);
        //if (mVerbose)
        if(mvVerbose) LOG(info) << "state mcmheader and word : 0x" << std::hex << *word;
        printDigitMCMHeader(*mDigitMCMHeader);
        //sanity check of digitheader ??  Still to implement.
        if (!digitMCMHeaderSanityCheck(mDigitMCMHeader)) {
          LOG(warn) << "Sanity check Failure MCMHeader : " << std::hex << mDigitMCMHeader->word;
          printDigitMCMHeader(*mDigitMCMHeader);
        };
        mBufferLocation++;
        mState = StateDigitHCHeader;
        //new header so digit word count becomes zero
        digitwordcount = 0;
        mMCM = mDigitMCMHeader->mcm;
        mROB = mDigitMCMHeader->rob;
        mEventCounter = mDigitMCMHeader->eventcount;
        mDataWordsParsed++;
        mChannel=0;
        if (!mReturnVector) {
          //returning the raw "compressed" data stream.
          //build the digit header and add to outgoing buffer;
          uint32_t* header = &(*mData)[mReturnVectorPos];
          //build header.
        }
        // we dont care about the year flag, we are >2007 already.
      } else {
        if (*word == o2::trd::constants::CRUPADDING32) {
          if (mVerbose) LOG(info) << "state padding and word : 0x" << std::hex << *word;
          //another pointer with padding.
          mBufferLocation++;
          mPaddingWordsCounter++;
          mState = StatePadding;
          mDataWordsParsed++;
          mcmdatacount=0;
          
          // this is supposed to carry on till the end of the buffer, hence the term padding.
          //TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
        } else { // all we are left with is digitmcmdata words.
            if (mVerbose) LOG(info) << "mDigitMCMData is at " << mBufferLocation << " had value 0x" << std::hex << *word << " mcmdatacount of : " << mcmdatacount<< " adc#" << mcmadccount;
            //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
            //for dpl build a vector and connect it with a triggerrecord.
            mDataWordsParsed++;
            mcmdatacount++;
            if (mReturnVector) { // we will generate a vector
                mDigitMCMData = (DigitMCMData*)word;
                mBufferLocation++;
                mState = StateDigitMCMData;
                digitwordcount++;
                if (mVerbose) LOG(info) << "adc values : " << mDigitMCMData->x << "::" << mDigitMCMData->y << "::" << mDigitMCMData->z;
                mADCValues[digittimebinoffset] = mDigitMCMData->x;
                mADCValues[digittimebinoffset++] = mDigitMCMData->y;
                mADCValues[digittimebinoffset++] = mDigitMCMData->z;
                if (mVerbose) LOG(info) << "digit word count is : " << digitwordcount;
                if (digitwordcount == constants::TIMEBINS/3) {
                    //sanity check, next word shouldbe either a. end of digit marker, digitMCMHeader,or padding.
                    if(mSanityCheck){
                        uint32_t *tmp = std::next(word);
                        LOG(info) << "digitwordcount = " << digitwordcount << " hopefully the next data is digitendmarker, didgitMCMHeader or padding 0x" << std::hex << *tmp;
                    }
                    if (mVerbose) LOG(info) << "change of adc";
                    mcmadccount++;
                    //write out adc value to vector
                    //zero digittimebinoffset
                    mDigits.emplace_back(mDetector, mROB, mMCM, mChannel, mADCValues); // outgoing parsed digits
                    digittimebinoffset = 0;
                    digitwordcount = 0; // end of the digit.
                    mChannel++;
                }

            } else {//version 2 will have below, it will be quicker not to have the intermediary step, but is it really needed?
                //returning digits raw, as its pretty much the most compressed you are going to get in anycase.
                // we will send the raw stream back. "compressed"
                // memcpy((char*)&(*mData)[mReturnVectorPos], (void*)word, sizeof(uint32_t));
                //TODO or should we just copy all timebins at the same time?
                //
            }
        }
      }
    }

    //accounting ....
    // mCurrentLinkDataPosition256++;
    // mCurrentHalfCRUDataPosition256++;
    // mTotalHalfCRUDataLength++;
    //end of data so
    if (word == mEndParse)
        if (mVerbose) LOG(info) << "word is mEndParse";
    if (std::distance(word, mEndParse) < 0)
        if (mVerbose) LOG(info) << std::dec << "word to mEndParse is :" << std::distance(word, mEndParse);
  }
  if (mVerbose) LOG(info) << "***** parsing loop finished for this link";

  if (mState != StateDigitMCMHeader || mState != StatePadding || mState != StateDigitEndMarker) {
      LOG(warn) << "Exiting parsing but the state is wrong ... mState= " << mState;
  }
  return mDataWordsParsed;
}

} // namespace trd
} // namespace o2
