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

#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"

#include "TRDReconstruction/TrackletsParser.h"
#include "fairlogger/Logger.h"

//TODO come back and figure which of below headers I actually need.
#include <cstring>
#include <string>
#include <vector>
#include <array>

namespace o2::trd
{

inline void TrackletsParser::swapByteOrder(unsigned int& ui)
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
  if (mVerbose) {
    LOG(info) << "Tracklet Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at pos " << mStartParse;
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

    LOG(info) << "trackletdata to parse with size of " << datacopy.size();
    int loopsize = 0;
    if (datacopy.size() > 1024) {
      loopsize = 64;
    }
    for (int i = 0; i < loopsize; i += 8) {
      LOG(info) << std::hex << "0x" << datacopy[i] << " " << std::hex << "0x" << datacopy[i + 1] << " " << std::hex << "0x" << datacopy[i + 2] << " " << std::hex << "0x" << datacopy[i + 3] << " " << std::hex << "0x" << datacopy[i + 4] << " " << std::hex << "0x" << datacopy[i + 5] << " " << std::hex << "0x" << datacopy[i + 6] << " " << std::hex << "0x" << datacopy[i + 7];
    }
    LOG(info) << "trackletdata to parse end";
    if (datacopy.size() > 1024) {
      LOG(fatal) << "something very wrong with tracklet parsing >1024";
    }
  }
  //mData holds a buffer containing tracklets parse placing tracklets in the output vector.
  //mData holds 2048 digits.
  mCurrentLink = 0;
  mWordsRead = 0;
  mState = StateTrackletMCMHeader; // we start with a trackletMCMHeader

  int currentLinkStart = 0;
  int mcmtrackletcount = 0;
  int trackletloopcount = 0;
  if (mDataVerbose) {
    LOG(info) << "distance to parse over is " << std::distance(mStartParse, mEndParse);
  }
  for (auto word = mStartParse; word != mEndParse; word++) { // loop over the entire data buffer (a complete link of tracklets and digits)
    if (mState == StateFinished) {
      return mWordsRead;
    }
    //  for (uint32_t index = start; index < end; index++) { // loop over the entire cru payload.
    //loop over all the words ... duh
    //check for tracklet end marker 0x1000 0x1000
    int index = std::distance(mStartParse, word);  //mData->begin());
    int indexend = std::distance(word, mEndParse); //mData->begin());
    if (mVerbose) {
      LOG(info) << "====== Tracklet Loop WordsRead = " << mWordsRead << " count : " << trackletloopcount << " and index " << index << " words till end : " << indexend << " word is :" << word << "  start is : " << mStartParse << " endis : " << mEndParse;
    }

    std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator nextword = word;
    std::advance(nextword, 1);
    uint32_t nextwordcopy = *nextword;

    if (mDataVerbose) {
      LOG(info) << "Before byteswapping " << index << " word is : " << std::hex << word << " next word is : " << nextwordcopy << " and raw nextword is :" << std::hex << (*mData)[index + 1];
    }
    if (!mDisableByteOrderFix) {
      swapByteOrder(*word);
      swapByteOrder(nextwordcopy);
    }
    if (mDataVerbose) {
      if (!mDisableByteOrderFix) {
        LOG(info) << "After byteswapping " << index << " word is : " << std::hex << word << " next word is : " << nextwordcopy << " and raw nextword is :" << std::hex << (*mData)[index + 1];
      } else {
        LOG(info) << "After byteswapping " << index << " word is : " << std::hex << word << " next word is : " << nextwordcopy << " and raw nextword is :" << std::hex << (*mData)[index + 1];
      }
      LOG(info) << "tracklet parsing " << std::hex << word << " at pos : " << mWordsRead;
    }

    if (*word == 0x10001000 && nextwordcopy == 0x10001000) {
      if (mVerbose) {
        LOG(info) << "found tracklet end marker bailing out of trackletparsing index is : " << index << " data size is : " << (mData)->size();
        LOG(info) << "=!=!=!=!=!=! loop distance to end of loop defined end  : " << std::distance(mEndParse, word) << " should be 2 for the currently read end marker still to account for";
        LOG(info) << "=!=!=!=!=!=! loop around for  Tracklet Loop WordsRead = " << mWordsRead + 2 << " count : " << trackletloopcount << " and index " << index << " words till end :" << indexend << " word is :" << word << "  start is : " << mStartParse << " endis : " << mEndParse;
      }
      mWordsRead += 2;
      //we should now have a tracklet half chamber header.
      mState = StateTrackletHCHeader;
      std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator hchword = word;
      std::advance(hchword, 2);
      uint32_t halfchamberheaderint = *hchword;
      if ((((*hchword) & (0x1 << 11)) != 0) && !mIgnoreTrackletHCHeader) { //TrackletHCHeader has bit 11 set to 1 always. Check for state because raw data can have bit 11 set!
        if (mHeaderVerbose) {
          LOG(info) << "after tracklet end marker mTrackletHCHeader is has value 0x" << std::hex << *word;
        }
        //read the header
        //we actually have an header word.
        mTrackletHCHeader = (TrackletHCHeader*)&word;
        if (mHeaderVerbose) {
          LOG(info) << "state mcmheader and word : 0x" << std::hex << *word << " sanity check : " << trackletHCHeaderSanityCheck(*mTrackletHCHeader);
          //printTrackletMCMHeader(*word);
        }
        mWordsRead++;
        mState = StateFinished;
      }

      //
      return mWordsRead;
    }
    //if (mState == StateTrackletHCHeader && (mWordsRead != 0)) {
    //    LOG(warn) << " Parsing state is StateTrackletHCHeader, yet according to the lengths we are not at the beginning of a half chamber. " << mWordsRead << " != 0 ";
    //}
    if (*word == o2::trd::constants::CRUPADDING32) {
      //padding word first as it clashes with the hcheader.
      mState = StatePadding;
      mWordsRead++;
    } else {
      //now for Tracklet hc header
      if ((((*word) & (0x1 << 11)) != 0) && !mIgnoreTrackletHCHeader && mState == StateTrackletHCHeader) { //TrackletHCHeader has bit 11 set to 1 always. Check for state because raw data can have bit 11 set!
        if (mHeaderVerbose) {
          LOG(info) << "mTrackletHCHeader is has value 0x" << std::hex << *word;
        }
        if (mState != StateTrackletHCHeader) {
          LOG(warn) << "Something wrong with TrackletHCHeader bit 11 is set but state is not " << StateTrackletMCMHeader << " its :" << mState;
        }
        //read the header
        //we actually have an header word.
        mTrackletHCHeader = (TrackletHCHeader*)&word;
        if (mHeaderVerbose) {
          LOG(info) << "state mcmheader and word : 0x" << std::hex << *word << " sanity check : " << trackletHCHeaderSanityCheck(*mTrackletHCHeader);
          //printTrackletMCMHeader(*word);
        }
        //sanity check of trackletheader ??
        if (!trackletHCHeaderSanityCheck(*mTrackletHCHeader)) {
          LOG(warn) << "Sanity check Failure HCHeader : " << mTrackletHCHeader;
        }

        mWordsRead++;
        mState = StateTrackletMCMHeader; // now we should read a MCMHeader next time through loop
                                         //    TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
      } else {                           //not TrackletMCMHeader
        if ((*word) & 0x80000001) {      //TrackletMCMHeader has the bits on either end always 1
          //mcmheader
          //        LOG(debug) << "changing state from padding to mcmheader as next datais 0x" << std::hex << mDataPointer[0];
          mTrackletMCMHeader = (TrackletMCMHeader*)&(*word);
          if (mHeaderVerbose) {
            LOG(info) << "state mcmheader and word : 0x" << std::hex << *word;
          }
          mState = StateTrackletMCMData; // afrter reading a header we should then have data for next round through the loop
          mcmtrackletcount = 0;
          mWordsRead++;
        } else {
          //        LOG(debug) << "changing statefrom padding to mcmdata as next datais 0x" << std::hex << mDataPointer[0];
          mState = StateTrackletMCMData;
          if (mDataVerbose) {
            LOG(info) << "mTrackletMCMData is at " << mWordsRead << " had value 0x" << std::hex << *word;
          }
          //tracklet data;
          // build tracklet.
          //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
          //for dpl build a vector and connect it with a triggerrecord.
          mTrackletMCMData = (TrackletMCMData*)&(*word);
          mWordsRead++;
          // take the header and this data word and build the underlying 64bit tracklet.
          int q0, q1, q2;
          int qa, qb;
          switch (mcmtrackletcount) {
            case 0:
              qa = mTrackletMCMHeader->pid0;
            case 1:
              qa = mTrackletMCMHeader->pid1;
            case 2:
              qa = mTrackletMCMHeader->pid2;
            default:
              LOG(warn) << "mcmtrackletcount is not in [0:2] something very wrong parsing the TrackletMCMData fields with data of : 0x" << std::hex << mTrackletMCMData->word;
          }
          q0 = getQFromRaw(mTrackletMCMHeader, mTrackletMCMData, 0, mcmtrackletcount);
          q1 = getQFromRaw(mTrackletMCMHeader, mTrackletMCMData, 1, mcmtrackletcount);
          q2 = getQFromRaw(mTrackletMCMHeader, mTrackletMCMData, 2, mcmtrackletcount);
          int padrow = mTrackletMCMHeader->padrow;
          int col = mTrackletMCMHeader->col;
          int pos = mTrackletMCMData->pos;
          int slope = mTrackletMCMData->slope;
          mTracklets.emplace_back(4, 21, padrow, col, pos, slope, q0, q1, q2);
          mcmtrackletcount++;
          if (mcmtrackletcount == 3) {
            LOG(warn) << "Tracklet count is >3 in parsing the TrackletMCMData attached to a single TrackletMCMHeader";
          }
        }
      }

      //accounting ....
      // mCurrentLinkDataPosition256++;
      // mCurrentHalfCRUDataPosition256++;
      // mTotalHalfCRUDataLength++;
    } // else
    if (mVerbose) {
      LOG(info) << "=!=!=!=!=!=! loop around for  Tracklet Loop count : " << trackletloopcount << " and index " << index << " word is :" << word << "  start is : " << mStartParse << " endis : " << mEndParse;
    }
    trackletloopcount++;
  } //end of for loop
  //sanity check, we should now see a digit Half Chamber Header in the following 2 32bit words.
  LOG(warn) << " end of Trackelt parsing but we are exiting with out a tracklet end marker with " << mWordsRead << " 32bit words read";
  return mWordsRead;
}

} // namespace o2::trd
