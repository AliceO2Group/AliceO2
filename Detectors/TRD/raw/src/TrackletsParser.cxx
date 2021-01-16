// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackletParser.h
/// @brief  TRD raw data parser for Tracklet data format

#include "TRDRaw/TrackletsParser.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"

#include "fairlogger/Logger.h"

//TODO come back and figure which of below headers I actually need.
#include <cstring>
#include <string>
#include <vector>
#include <array>

namespace o2::trd
{

inline void swapByteOrder(unsigned int& ui)
{
  ui = (ui >> 24) |
       ((ui << 8) & 0x00FF0000) |
       ((ui >> 8) & 0x0000FF00) |
       (ui << 24);
}

int TrackletsParser::Parse()
{
  //we are handed the buffer payload of an rdh and need to parse its contents.
  //producing a vector of digits.
  LOG(debug) << "Tracklet Parser parse of data sitting at :" << std::hex << (void*)mData;
  //mData holds a buffer containing tracklets parse placing tracklets in the output vector.
  //mData holds 2048 digits.
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  mCurrentLink = 0;
  mWordsRead = 0;
  mState = StateTrackletHCHeader;
  int currentLinkStart = 0;
  for (uint32_t index = 0; index < (mData)->size(); index++) { // loop over the entire cru payload.
    //loop over all the words ... duh
    //check for tracklet end marker 0x1000 0x1000
    uint32_t word = (*mData)[index];
    uint32_t nextword = (*mData)[index + 1];
    swapByteOrder(word);
    swapByteOrder(nextword);
    LOG(info) << "tracklet parsing 0x " << std::hex << word << " at pos : " << mWordsRead;
    LOGF(info, "tracklet parsing 0x%08x at pos %d", word, mWordsRead);
    LOGF(info, "tracklet parsing 0x%08x at pos %d", nextword, mWordsRead);
    if (word == 0x10001000 && nextword == 0x10001000) {
      LOG(info) << "found tracklet end marker bailing out of trackletparsing";
      mWordsRead += 2;
      return mWordsRead;
    }
    if (mState == StateTrackletHCHeader && (mWordsRead != 0)) {
      LOG(warn) << " Parsing state is StateTrackletHCHeader, yet according to the lengths we are not at the beginning of a half chamber. " << mWordsRead << " != 0 ";
    }
    // we are changing a link.
    if (mState == StateTrackletMCMHeader) {
      LOG(debug) << "mTrackletMCMHeader is has value 0x" << std::hex << word;
      //read the header OR padding of 0xeeee;
      if (word != 0xeeeeeeee) {
        //we actually have an header word.
        mTrackletHCHeader = (TrackletHCHeader*)&word;
        LOG(debug) << "state mcmheader and word : 0x" << std::hex << word;
        //sanity check of trackletheader ??
        if (!trackletMCMHeaderSanityCheck(*mTrackletMCMHeader)) {
          LOG(warn) << "Sanity check Failure MCMHeader : " << mTrackletMCMHeader;
        };
        mWordsRead++;
        mState = StateTrackletMCMData;
      } else { // this is the case of a first padding word for a "noncomplete" tracklet i.e. not all 3 tracklets.
        //        LOG(debug) << "C";
        mState = StatePadding;
        mWordsRead++;
        //    TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
      }
    } else {
      if (mState == StatePadding) {
        LOG(debug) << "state padding and word : 0x" << std::hex << word;
        if (word == 0xeeeeeeee) {
          //another pointer with padding.
          mWordsRead++;
          //TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
          if (word & 0x1) {
            //mcmheader
            //        LOG(debug) << "changing state from padding to mcmheader as next datais 0x" << std::hex << mDataPointer[0];
            mState = StateTrackletMCMHeader;
          } else if (word != 0xeeeeeeee) {
            //        LOG(debug) << "changing statefrom padding to mcmdata as next datais 0x" << std::hex << mDataPointer[0];
            mState = StateTrackletMCMData;
          }
        } else {
          LOG(debug) << "some went wrong we are in state padding, but not a pad word. 0x" << word;
        }
      }
      if (mState == StateTrackletMCMData) {
        LOG(debug) << "mTrackletMCMData is at " << mWordsRead << " had value 0x" << std::hex << word;
        //tracklet data;
        // build tracklet.
        //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
        //for dpl build a vector and connect it with a triggerrecord.
        mTrackletMCMData = (TrackletMCMData*)&word;
        mWordsRead++;
        if (word == 0xeeeeeeee) {
          mState = StatePadding;
          //  LOG(debug) <<"changing to padding from mcmdata" ;
        } else {
          if (word & 0x1) {
            mState = StateTrackletMCMHeader; // we have more tracklet data;
            LOG(debug) << "changing from MCMData to MCMHeader";
          } else {
            mState = StateTrackletMCMData;
            LOG(debug) << "continuing with mcmdata";
          }
        }
        // Tracklet64 trackletsetQ0(o2::trd::getTrackletQ0());
      }

      //accounting ....
      // mCurrentLinkDataPosition256++;
      // mCurrentHalfCRUDataPosition256++;
      // mTotalHalfCRUDataLength++;
      //end of data so
    }
  }
  return -1; // if we get here it means that we did not find a tracklet end marker ergo its an error condition;
}

} // namespace o2::trd
