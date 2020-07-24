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
  LOG(info) << "Tracklet Parser parse of data sitting at :" << std::hex << (void*)mData << " starting at pos " << mStartParse;
  //mData holds a buffer containing tracklets parse placing tracklets in the output vector.
  //mData holds 2048 digits.
  // due to the nature of the incoming data, there will *never* straggling digits or for that matter trap outputs spanning a boundary.
  mCurrentLink = 0;
  mWordsRead = 0;
  mState = StateTrackletHCHeader;

  int currentLinkStart = 0;
  int mcmtrackletcount=0;
  for (auto word = mStartParse; word != mEndParse; word++) { // loop over the entire data buffer (a complete link of tracklets and digits)
//  for (uint32_t index = start; index < end; index++) { // loop over the entire cru payload.
    //loop over all the words ... duh
    //check for tracklet end marker 0x1000 0x1000
    int index=std::distance(word,mData->begin());
    LOG(info) << " index is :  "<< std::hex << index << " word is :" << word<< "  start is : " << mStartParse << " endis : " << mEndParse;;
  // uint32_t word = (*mData)[index];
    
    std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>::iterator nextword=word;
    std::advance(nextword,1);
//    uint32_t nextword = (*mData)[index + 1]; std::
    uint32_t nextwordcopy=*nextword;

    LOG(info) << "Before byteswapping " << index << " word is : " << std::hex << word << " next word is : " <<nextwordcopy << " and raw nextword is :" << std::hex << (*mData)[index+1];
    if (!mDisableByteOrderFix) {
      swapByteOrder(*word);
      swapByteOrder(nextwordcopy);
    }
    LOG(info) << "Before byteswapping " << index << " word is : " << std::hex << *word << " next word is : " <<*nextword << " and raw nextword is :" << std::hex << (*mData)[index+1];
    LOG(info) << "Before byteswapping " << index << " word is : " << std::hex << word << " next word is : " <<nextwordcopy << " and raw nextword is :" << std::hex << (*mData)[index+1];
    LOG(info) << "tracklet parsing 0x " << std::hex << word << " at pos : " << mWordsRead;
    if (*word == 0x10001000 && nextwordcopy == 0x10001000) {
      LOG(info) << "found tracklet end marker bailing out of trackletparsing index is : " << index << " data size is : " << (mData)->size();
      mWordsRead += 2;
      return mWordsRead;
    }
    if (mState == StateTrackletHCHeader && (mWordsRead != 0)) {
      LOG(warn) << " Parsing state is StateTrackletHCHeader, yet according to the lengths we are not at the beginning of a half chamber. " << mWordsRead << " != 0 ";
    }
    if (*word == o2::trd::constants::CRUPADDING32) {
      //padding word first as it clashes with the hcheader.
      mState = StatePadding;
      mWordsRead++;
    } else {
      //now for Tracklet hc header
      if ((((*word) & 0x800) != 0)) { //TrackletHCHeader has bit 11 set to 1 always.
        LOG(debug) << "mTrackletHCHeader is has value 0x" << std::hex << *word;
        if (mState != StateTrackletMCMHeader) {
          LOG(warn) << "Something wrong with TrackletMCMHeader bit 11 is set but state is not " << StateTrackletMCMHeader << " its :" << mState;
        }
        //read the header
        //we actually have an header word.
        mTrackletHCHeader = (TrackletHCHeader*)&word;
        LOG(debug) << "state mcmheader and word : 0x" << std::hex << *word;
        //sanity check of trackletheader ??
        if (!trackletHCHeaderSanityCheck(*mTrackletHCHeader)) {
          LOG(warn) << "Sanity check Failure HCHeader : " << mTrackletHCHeader;
        }
        mWordsRead++;
        mState = StateTrackletMCMHeader; // now we should read a MCMHeader next time through loop
                                         //    TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
      } else {                           //not TrackletMCMHeader
        if ((*word) & 0x80000001) {         //TrackletHCHeader has the bits no either end always 1
          //mcmheader
          //        LOG(debug) << "changing state from padding to mcmheader as next datais 0x" << std::hex << mDataPointer[0];
          mTrackletMCMHeader = (TrackletMCMHeader*)&(*word);
          LOG(debug) << "state mcmheader and word : 0x" << std::hex << *word;
          mState = StateTrackletMCMData; // afrter reading a header we should then have data for next round through the loop
          mcmtrackletcount=0;
        } else {
          //        LOG(debug) << "changing statefrom padding to mcmdata as next datais 0x" << std::hex << mDataPointer[0];
          mState = StateTrackletMCMData;
          LOG(debug) << "mTrackletMCMData is at " << mWordsRead << " had value 0x" << std::hex << *word;
          //tracklet data;
          // build tracklet.
          //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
          //for dpl build a vector and connect it with a triggerrecord.
          mTrackletMCMData = (TrackletMCMData*)&(*word);
          mWordsRead++;
          // take the header and this data word and build the underlying 64bit tracklet.
          int q0,q1,q2;
          int qa,qb;
          switch(mcmtrackletcount){
                case 0 : qa= mTrackletMCMHeader->pid0;
                case 1 : qa= mTrackletMCMHeader->pid1;
                case 2 : qa= mTrackletMCMHeader->pid2;
               default : LOG(warn) << "mcmtrackletcount is not in [0:2] something very wrong parsing the TrackletMCMData fields with data of : 0x" << std::hex << mTrackletMCMData->word;
          }
          q0=getQFromRaw(mTrackletMCMHeader,mTrackletMCMData,0,mcmtrackletcount);
          q1=getQFromRaw(mTrackletMCMHeader,mTrackletMCMData,1,mcmtrackletcount);
          q2=getQFromRaw(mTrackletMCMHeader,mTrackletMCMData,2,mcmtrackletcount);
          int padrow=mTrackletMCMHeader->padrow;
          int col=mTrackletMCMHeader->col;
          int pos=mTrackletMCMData->pos;
          int slope=mTrackletMCMData->slope;
          mTracklets.emplace_back(4, 21, padrow,  col, pos, slope,q0,q1,q2 );
         mcmtrackletcount++; 
         if(mcmtrackletcount==3)LOG(warn) << "Tracklet count is >3 in parsing the TrackletMCMData attached to a single TrackletMCMHeader";
        }
      }

      //accounting ....
      // mCurrentLinkDataPosition256++;
      // mCurrentHalfCRUDataPosition256++;
      // mTotalHalfCRUDataLength++;
    }        // else
  }          //end of for loop
  //sanity check, we should now see a digit Half Chamber Header in the following 2 32bit words.
  return mWordsRead; // if we get here it means that we did not find a tracklet end marker ergo its an error condition;
}

} // namespace o2::trd
