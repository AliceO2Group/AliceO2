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
  if (mVerbose)
    LOG(info) << "Digits Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at " << std::distance(mData->begin(), mStartParse) << " ending at " << std::distance(mData->begin(), mEndParse);
  //mData holds a buffer containing tracklets parse placing tracklets in the output vector.
  //mData holds 2048 digits.
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  // data starts with a DigitHCHeader, so pull that off first to simplify looping
  if (mVerbose)
    LOG(info) << "word to parse : " << std::hex << *mStartParse << "and " << *(mStartParse + 1) << " in state :" << mState;
  if (mState == StateDigitHCHeader) {
    if (mVerbose)
      LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% at start of data";
    mDigitHCHeader = (DigitHCHeader*)mStartParse;
    swapByteOrder(mDigitHCHeader->word0);
    swapByteOrder(mDigitHCHeader->word1);
    if (mVerbose)
      LOG(info) << mDigitHCHeader->bunchcrossing << " was bunchcrossing and " << mDigitHCHeader->supermodule << " " << mDigitHCHeader->layer;
    printDigitHCHeader(*mDigitHCHeader);
    mState = StateDigitMCMHeader;
    mBufferLocation += 2;
  }
  for (auto word = mStartParse + 2; word != mEndParse; word++) { // loop over the entire data buffer (a complete link of digits)
    //if(mVerbose) LOG(info) << "word to parse : " << std::hex << *word << " in state :" << mState;
    //if(mVerbose) LOG(info) <<std::dec <<  "word to mEndParse is :"<< std::distance(word,mEndParse);
    //if(mVerbose) LOG(info) <<std::dec <<  "mStartParse to mEndParse is :"<< std::distance(mStartParse,mEndParse);
    //loop over all the words
    if (mVerbose)
      LOG(info) << "parsing word : " << std::hex << *word;
    //check for digit end marker
    swapByteOrder(*word);
    auto nextword = std::next(word, 1);
    if ((*word) == 0x0 && (*nextword == 0x0)) { // no need to byte swap nextword as zero does not change.
      // end of digits marker.
      if (mVerbose)
        LOG(info) << "Found digits end marker :" << std::hex << *word << "::" << *nextword;
      //state *should* be StateDigitMCMData check that it is
      if (mState != StateDigitMCMData) {
        LOG(warn) << "Digit end marker found but state is not StateDigitMCMData(" << StateDigitMCMData << ") but rather " << mState;
      }
      //only thing that can remain is the padding.
      //now read padding words till end.
      //no need to byteswap its uniform.
      mBufferLocation++;
      mState = StateDigitEndMarker;
    } else {
      if ((*word & 0xf) == 0xc) { //marker for DigitMCMHeader.
        if (mVerbose)
          LOG(info) << "mDigitMCMHeader has value " << std::hex << *word;
        //read the header OR padding of 0xeeee;
        //we actually have an header word.
        mDigitMCMHeader = (DigitMCMHeader*)(word);
        if (mVerbose)
          LOG(info) << "state mcmheader and word : 0x" << std::hex << *word;
        //sanity check of digitheader ??  Still to implement.
        if (!digitMCMHeaderSanityCheck(mDigitMCMHeader)) {
          LOG(warn) << "Sanity check Failure MCMHeader : " << std::hex << mDigitMCMHeader->word;
          printDigitMCMHeader(*mDigitMCMHeader);
        };
        mBufferLocation++;
        mState = StateDigitHCHeader;

      } else {
        if (*word == o2::trd::constants::CRUPADDING32) {
          if (mVerbose)
            LOG(info) << "state padding and word : 0x" << std::hex << *word;
          //another pointer with padding.
          mBufferLocation++;
          mPaddingWordsCounter++;
          mState = StatePadding;

          // this is supposed to carry on till the end of the buffer, hence the term padding.
          //TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
        } else { // all we are left with is digitmcmdata words.
          if (mVerbose)
            LOG(info) << "assumed mDigitMCMData is at " << mBufferLocation << " had value 0x" << std::hex << *word;
          //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
          //for dpl build a vector and connect it with a triggerrecord.
          mDigitMCMData = (DigitMCMData*)word;
          mBufferLocation++;
          mState = StateDigitMCMData;
        }
        }
      }

      //accounting ....
      // mCurrentLinkDataPosition256++;
      // mCurrentHalfCRUDataPosition256++;
      // mTotalHalfCRUDataLength++;
      //end of data so
      if (word == mEndParse)
        if (mVerbose)
          LOG(info) << "word is mEndParse";
      if (std::distance(word, mEndParse) > 0)
        if (mVerbose)
          LOG(info) << std::dec << "word to mEndParse is :" << std::distance(word, mEndParse);
    }
    if (mVerbose)
      LOG(info) << "***** parsing loop finished for this link";
    return 1;
  }

} // namespace trd
} // namespace o2
