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

#include "TRDRaw/DigitsParser.h"
#include "DataFormatsTRD/RawData.h"
#include "TRDBase/Digit.h"
#include "DataFormatsTRD/Constants.h"

#include "fairlogger/Logger.h"

//TODO come back and figure which of below headers I actually need.
#include <cstring>
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
            ((word<<8) & 0x00FF0000) |
            ((word>>8) & 0x0000FF00) |
            (word << 24);
    }

int DigitsParser::Parse()
{

  //we are handed the buffer payload of an rdh and need to parse its contents.
  //producing a vector of digits.
  LOG(info) << "Digits Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at " << std::distance(mData->begin(), mStartParse) << " ending at " << std::distance(mData->begin(), mEndParse);
  //mData holds a buffer containing tracklets parse placing tracklets in the output vector.
  //mData holds 2048 digits.
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  // data starts with a DigitHCHeader, so pull that off first to simplify looping
  LOG(info) << "word to parse : " << std::hex << *mStartParse << "and " << *(mStartParse + 1) << " in state :" << mState;
  if (mState == StateDigitHCHeader) {
    LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% at start of data";
    mDigitHCHeader = (DigitHCHeader*)mStartParse;
    LOG(info) << mDigitHCHeader->bunchcrossing << " was bunchcrossing and " << mDigitHCHeader->supermodule << " " << mDigitHCHeader->layer;
    mState = StateDigitMCMHeader;
  }
  for (auto word = mStartParse; word != mEndParse; word++) { // loop over the entire data buffer (a complete link)
    //LOG(info) << "word to parse : " << std::hex << *word << " in state :" << mState;
    //LOG(info) <<std::dec <<  "word to mEndParse is :"<< std::distance(word,mEndParse);
    //LOG(info) <<std::dec <<  "mStartParse to mEndParse is :"<< std::distance(mStartParse,mEndParse);

    //loop over all the words
    //check for digit end marker
    swapByteOrder(*word);
    auto nextword=std::next(word,1);
    swapByteOrder(*nextword);
    if((*word)==0x0 && (*nextword)==0x0){
        // end of digits marker.
        LOG(info) << "Found digits end marker :" << std::hex << *word << "::" << *nextword;
        //state *should* be StateDigitMCMData check that it is
        if(mState != StateDigitMCMData){
            LOG(warn) << "Digit end marker found but state is not StateDigitMCMData("<<StateDigitMCMData << ") but rather " << mState;
        }
        //only thing that can remain is the padding.
        //now read padding words till end.
        //no need to byteswap its uniform.
        mBufferLocation+=2;
        int paddingoffsetcount=2;
        auto paddingword=std::next(word,paddingoffsetcount);
        while(*paddingword == 0xeeeeeeee){
            //read the remainder of the padding words.
            mPaddingWordCounter++;
            paddingword=std::next(word,paddingoffsetcount);
        }
    }
    if (mState == StateDigitMCMHeader) {
      LOG(info) << "mDigitMCMHeader has value " << std::hex << *word;
      //read the header OR padding of 0xeeee;
      if (*word != 0xeeeeeeee) {
        //we actually have an header word.
        mDigitMCMHeader = (DigitMCMHeader*)(word);
        LOG(info) << "state mcmheader and word : 0x" << std::hex << *word;
        //sanity check of digitheader ??  Still to implement.
        if (!digitMCMHeaderSanityCheck(mDigitMCMHeader)) {
          LOG(warn) << "Sanity check Failure MCMHeader : " << std::hex << mDigitMCMHeader->word;
          printDigitMCMHeader(*mDigitMCMHeader);
        };
        mBufferLocation++;
        mState = StateDigitMCMData;
      } else { // this is the case of a first padding word for a "noncomplete" tracklet i.e. not all 3 tracklets.
               //        LOG(info) << "C";
        mState = StatePadding;
        mBufferLocation++;
        //    TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
      }
    } else {
      if (mState == StatePadding) {
        LOG(info) << "state padding and word : 0x" << std::hex << *word;
        if (*word == 0xeeeeeeee) {
          //another pointer with padding.
          mBufferLocation++;
          //TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
          if (*word & 0x1) {
            //mcmheader
            //        LOG(info) << "changing state from padding to mcmheader as next datais 0x" << std::hex << mDataPointer[0];
            mState = StateDigitMCMHeader;
          } else if (*word != 0xeeeeeeee) {
            //        LOG(info) << "changing statefrom padding to mcmdata as next datais 0x" << std::hex << mDataPointer[0];
            mState = StateDigitMCMData;
          }
        } else {
          LOG(info) << "some went wrong we are in state padding, but not a pad word. 0x" << std::hex << *word;
        }
      }
      if (mState == StateDigitMCMData) {
        LOG(info) << "mDigitMCMData is at " << mBufferLocation << " had value 0x" << std::hex << *word;
        //tracklet data;
        // build tracklet.
        //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
        //for dpl build a vector and connect it with a triggerrecord.
        mDigitMCMData = (DigitMCMData*)word;
        mBufferLocation++;
        if (*word == 0xeeeeeeee) {
          mState = StatePadding;
          LOG(info) <<"changing state to padding from mcmdata" ;
        } else {
          if (*word & 0x1) {
            mState = StateDigitMCMHeader; // we have more tracklet data;
            LOG(info) << "changing from digitMCMData to digitMCMHeader";
          } else {
            mState = StateDigitMCMData;
            LOG(info) << "continuing with digitmcmdata";
          }
        }
        // Digit64 trackletsetQ0(o2::trd::getDigitQ0());
      }

      //accounting ....
      // mCurrentLinkDataPosition256++;
      // mCurrentHalfCRUDataPosition256++;
      // mTotalHalfCRUDataLength++;
      //end of data so
    }
    if (word == mEndParse) LOG(info) << "word is mEndParse";
    if (std::distance(word,mEndParse)> 0) LOG(info) <<std::dec <<  "word to mEndParse is :"<< std::distance(word,mEndParse);
  }
  LOG(info) << "***** parsing loop finished for this link";
  return 1;
}

} // namespace trd
} // namespace o2
