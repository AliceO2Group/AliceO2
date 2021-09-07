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

/// @file   TrackletParser.h
/// @brief  TRD raw data parser for Tracklet data format

#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/RawDataStats.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/HelperMethods.h"

#include "TRDReconstruction/TrackletsParser.h"
#include "TRDReconstruction/EventRecord.h"
#include "fairlogger/Logger.h"

//TODO come back and figure which of below headers I actually need.
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iomanip>
#include <iostream>
#include "TH1F.h"

namespace o2::trd
{

inline void TrackletsParser::swapByteOrder(unsigned int& ui)
{
  ui = (ui >> 24) |
       ((ui << 8) & 0x00FF0000) |
       ((ui >> 8) & 0x0000FF00) |
       (ui << 24);
}

int TrackletsParser::Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data,
                           std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start,
                           std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end,
                           TRDFeeID feeid, int robside, int detector, int stack, int layer,
                           EventRecord* eventrecord, std::bitset<16> options, bool cleardigits, int usetracklethcheader)
{
  mStartParse = start;
  mEndParse = end;
  mDetector = detector;
  mFEEID = feeid;
  mRobSide = robside;
  mStack = stack;
  mLayer = layer;
  mOptions = options;
  setData(data);
  setVerbose(options[TRDVerboseBit], options[TRDHeaderVerboseBit], options[TRDDataVerboseBit]);
  setByteSwap(options[TRDByteSwapBit]);
  mWordsRead = 0;
  mWordsDumped = 0;
  mTrackletsFound = 0;
  mPaddingWordsCounter = 0;
  mTrackletHCHeaderState = usetracklethcheader; //what to with the tracklet half chamber header 0,1,2
  mIgnoreTrackletHCHeader = options[TRDIgnoreTrackletHCHeaderBit];
  mEventRecord = eventrecord;
  //    mTracklets.clear();
  return Parse();
}

void TrackletsParser::OutputIncomingData()
{
  LOG(info) << "Data to parse for Tracklets from " << std::hex << mStartParse << " to " << mEndParse;
  int wordcount = 0;
  std::stringstream outputstring;
  auto word = mStartParse;
  outputstring << "tracklet 0x" << std::hex << std::setfill('0') << std::setw(6) << 0 << " :: ";
  while (word <= mEndParse) { // loop over the entire data buffer (a complete link of tracklets and digits)

    if (wordcount != 0 && (wordcount % 8 == 0 || word == mEndParse)) {
      LOG(info) << outputstring.str();
      outputstring.str("");
      outputstring << "tracklet 0x" << std::hex << std::setfill('0') << std::setw(6) << wordcount << " :: ";
    }
    if (wordcount == 0) {
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << HelperMethods::swapByteOrderreturn(*word);
    } else {
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << HelperMethods::swapByteOrderreturn(*word);
    }
    word++;
    wordcount++;
  }
  LOG(info) << "Data buffer to parse for Tracklets end";
  /*for (auto word = mStartParse; word != mEndParse; word+=8) { // loop over the entire data buffer (a complete link of tracklets and digits)
        LOGP(info,"0x{0:08x} :: {1:08x} {2:08x}  {3:08x} {4:08x} {5:08x} {6:08x} {7:08x} {8:08x} ",std::distance(mStartParse,word),
            HelperMethods::swapByteOrderreturn(*word), HelperMethods::swapByteOrderreturn(*std::next(word,1)),
            HelperMethods::swapByteOrderreturn(*std::next(word,2)), HelperMethods::swapByteOrderreturn(*std::next(word,3)),
            HelperMethods::swapByteOrderreturn(*std::next(word,4)), HelperMethods::swapByteOrderreturn(*std::next(word,5)),
            HelperMethods::swapByteOrderreturn(*std::next(word,6)), HelperMethods::swapByteOrderreturn(*std::next(word,7)));
    }
    LOG(info) << "Data to parse for Tracklets end";*/
}

int TrackletsParser::Parse()
{
  auto parsetimestart = std::chrono::high_resolution_clock::now(); // measure total processing time
  //we are handed the buffer payload of an rdh and need to parse its contents.
  //producing a vector of digits.

  mTrackletParsingBad = false;
  if (mHeaderVerbose) {
    OutputIncomingData();
  }
  //mData holds a buffer containing tracklets parse placing tracklets in the output vector.
  //mData holds 2048 digits.
  mCurrentLink = 0;
  mWordsRead = 0;
  mTrackletsFound = 0;
  if (mTrackletHCHeaderState == 0) { //TODO move this to the reader as done for the digithcheader
    // tracklet hc header is never present
    mState = StateTrackletMCMHeader;
  } else {
    if (mTrackletHCHeaderState == 1) {
      auto nextword = std::next(mStartParse);
      if (*nextword != constants::TRACKLETENDMARKER) {
        //we have tracklet data so no TracletHCHeader
        mState = StateTrackletHCHeader;
      } else {
        //we have no tracklet data so no TracletHCHeader
        mState = StateTrackletMCMHeader;
      }
    } else {
      if (mTrackletHCHeaderState != 2) {
        LOG(warn) << "unknwon TrackletHCHeaderState of " << mIgnoreTrackletHCHeader;
      }
      // tracklet hc header is always present
      mState = StateTrackletHCHeader; // we start with a trackletMCMHeader
    }
  }

  int currentLinkStart = 0;
  int mcmtrackletcount = 0;
  int trackletloopcount = 0;
  int headertrackletcount = 0;
  bool ignoreDataTillTrackletEndMarker = false;             // used for when we need to dump the rest of the tracklet data.
  for (auto word = mStartParse; word < mEndParse; ++word) { // loop over the entire data buffer (a complete link of tracklets and digits)

    if (mState == StateFinished) {
      mTrackletparsetime += std::chrono::high_resolution_clock::now() - parsetimestart;
      return mWordsRead;
    }
    //loop over all the words ...
    //check for tracklet end marker 0x1000 0x1000
    int index = std::distance(mStartParse, word);
    int indexend = std::distance(word, mEndParse);
    std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator nextword = word;
    std::advance(nextword, 1);
    uint32_t nextwordcopy = *nextword;

    if (mByteOrderFix) {
      swapByteOrder(*word);
      swapByteOrder(nextwordcopy);
    }

    if (*word == 0x10001000 && nextwordcopy == 0x10001000) {
      if (!StateTrackletEndMarker && !StateTrackletHCHeader) {
        LOG(warn) << "State should be trackletend marker current ?= end marker  ?? " << mState << " ?=" << StateTrackletEndMarker;
      }

      mWordsRead += 2;

      if (mHeaderVerbose) {
        LOG(info) << "***TrackletEndMarker : 0x" << std::hex << *word << " and 0x" << nextwordcopy << " at offset " << std::distance(mStartParse, word);
      }

      mState = StateFinished;
      mTrackletparsetime += std::chrono::high_resolution_clock::now() - parsetimestart;
      return mWordsRead;
    }
    if (*word == o2::trd::constants::CRUPADDING32) {
      //padding word first as it clashes with the hcheader.
      mState = StatePadding;
      //LOG(warn) << "CRU Padding word while parsing tracklets. Corrupt data dumping the rest of this link";
      //mEventRecord.ErrorStats[TRDParsingTrackletCRUPaddingWhileParsingTracklets]++;
      if (mOptions[TRDEnableRootOutputBit]) {
        mParsingErrors->Fill(TRDParsingTrackletCRUPaddingWhileParsingTracklets);
        increment2dHist(TRDParsingTrackletCRUPaddingWhileParsingTracklets);
      }
      //TOOD replace warning with stats increment
      mWordsDumped = std::distance(word, mEndParse);
      ignoreDataTillTrackletEndMarker = true;
      //mWordsRead++;
      word = mEndParse;
      //TODO remove tracklets already added erroneously
      continue; // bail out
      //dumping data

    } else {
      if (ignoreDataTillTrackletEndMarker) {
        mWordsDumped++;
        //LOG(info) << "ignoring till end marker ..... word read:0x"<<std::hex << *word << " at offset 0x"<< std::distance(mStartParse,word);
        //TODO increment counter instead of above log message
        if (mOptions[TRDEnableRootOutputBit]) {
          mParsingErrors->Fill(TRDParsingTrackletBit11NotSetInTrackletHCHeader);
          increment2dHist(TRDParsingTrackletBit11NotSetInTrackletHCHeader);
        }
        continue; //go back to the start of loop, walk the data till the above code of the tracklet end marker is hit, padding is hit or we get to the end of the data.
        //TODO might be good to check for end of digit marker as well?
      }
      //now for Tracklet hc header
      if ((((*word) & (0x1 << 11)) != 0) && !mIgnoreTrackletHCHeader && mState == StateTrackletHCHeader) { //TrackletHCHeader has bit 11 set to 1 always. Check for state because raw data can have bit 11 set!
        if (mState != StateTrackletHCHeader) {
          //   LOG(warn) << "Something wrong with TrackletHCHeader bit 11 is set but state is not " << StateTrackletMCMHeader << " its :" << mState;
          //TODO count remove warning
          //mEventRecord.ErrorStats[TRDParsingTrackletBit11NotSetInTrackletHCHeader]++;
          if (mOptions[TRDEnableRootOutputBit]) {
            mParsingErrors->Fill(TRDParsingTrackletBit11NotSetInTrackletHCHeader);
            increment2dHist(TRDParsingTrackletBit11NotSetInTrackletHCHeader);
          }
        }
        //read the header
        if (mHeaderVerbose) {
          LOG(info) << "*** TrackletHCHeader : 0x" << std::hex << *word << " at offset :0x" << std::distance(mStartParse, word);
        }
        //we actually have a header word.
        mTrackletHCHeader = (TrackletHCHeader*)&word;
        //sanity check of trackletheader ??
        if (!trackletHCHeaderSanityCheck(*mTrackletHCHeader)) {
          //  LOG(warn) << "Sanity check Failure HCHeader : " << std::hex << *word << " at offset :0x" << std::distance(mStartParse, word);
          //TODO count remove warning
          //mEventRecord.ErrorStats[TRDParsingTrackletHCHeaderSanityCheckFailure]++;
          if (mOptions[TRDEnableRootOutputBit]) {
            mParsingErrors->Fill(TRDParsingTrackletHCHeaderSanityCheckFailure);
            increment2dHist(TRDParsingTrackletHCHeaderSanityCheckFailure);
          }
        }
        mWordsRead++;
        mState = StateTrackletMCMHeader;                                // now we should read a MCMHeader next time through loop
      } else {                                                          //not TrackletHCHeader
        if (((*word) & 0x80000001) == 0x80000001 && mState == StateTrackletMCMHeader) { //TrackletMCMHeader has the bits on either end always 1
          //mcmheader
          mTrackletMCMHeader = (TrackletMCMHeader*)&(*word);
          if (mHeaderVerbose) {
            LOG(info) << "***TrackletMCMHeader : 0x" << std::hex << *word << " at offset: 0x" << std::distance(mStartParse, word);
            TrackletMCMHeader a;
            a.word = *word;
            printTrackletMCMHeader(a);
          }
          if (!trackletMCMHeaderSanityCheck(*mTrackletMCMHeader)) {
            //   LOG(warn) << "***TrackletMCMHeader SanityCheckFailure: 0x" << std::hex << *word << " at offset: 0x" << std::distance(mStartParse, word);
            //mEventRecord.ErrorStats[TRDParsingTrackletMCMHeaderSanityCheckFailure]++;
            if (mOptions[TRDEnableRootOutputBit]) {
              mParsingErrors->Fill(TRDParsingTrackletMCMHeaderSanityCheckFailure);
              increment2dHist(TRDParsingTrackletMCMHeaderSanityCheckFailure);
            }
          }
          headertrackletcount = getNumberofTracklets(*mTrackletMCMHeader);
          if (headertrackletcount > 0) {
            mState = StateTrackletMCMData; // afrter reading a header we should then have data for next round through the loop
          } else {
            mState = StateTrackletMCMHeader;
          }

          mcmtrackletcount = 0;
          mWordsRead++;
        } else {
          if (mState == StateTrackletMCMHeader || (mState == StateTrackletHCHeader && !mOptions[mIgnoreTrackletHCHeader])) {
            // if we are here something is wrong, dump the data. The else of line 227 should imply we are in StateTrackletMCMData;
            ignoreDataTillTrackletEndMarker = true;
            //mEventRecord.ErrorStats[
            if (mOptions[TRDEnableRootOutputBit]) {
              mParsingErrors->Fill(TRDParsingTrackletStateMCMHeaderButParsingMCMData);
              increment2dHist(TRDParsingTrackletStateMCMHeaderButParsingMCMData);
            }
            //mWordsRead++;
            mWordsDumped = std::distance(word, mEndParse);
            ignoreDataTillTrackletEndMarker = true;
            word = mEndParse;
            continue;
          }
          mState = StateTrackletMCMData;
          //tracklet data;
          mTrackletMCMData = (TrackletMCMData*)&(*word);
          mWordsRead++;
          if (mHeaderVerbose) {
            LOG(info) << "*** TrackletMCMData : 0x" << std::hex << *word << " at offset :0x" << std::distance(mStartParse, word);
            printTrackletMCMData(*mTrackletMCMData);
          }
          // do we have more tracklets than the header allows?
          if (headertrackletcount < mcmtrackletcount) {
            ignoreDataTillTrackletEndMarker = true;
            //dump the rest of the data ... undo any tracklets already written?
            //cant dump till mEndParse and digits are after the tracklets
            //we can assume the mcmtrackletcountth (n from the end) last tracklets in the vector are to be removed.
            //mEventRecord.ErrorStats[TRDParsingTrackletTrackletCountGTThatDeclaredInMCMHeader]++;
            if (mOptions[TRDEnableRootOutputBit]) {
              mParsingErrors->Fill(TRDParsingTrackletTrackletCountGTThatDeclaredInMCMHeader);
              increment2dHist(TRDParsingTrackletTrackletCountGTThatDeclaredInMCMHeader);
            }
            mEventRecord->popTracklets(mcmtrackletcount); // our format is always 4
            //TODO count remove warning
          }
          // take the header and this data word and build the underlying 64bit tracklet.
          int q0, q1, q2;
          int qa, qb;
          switch (mcmtrackletcount) {
            case 0:
              qa = mTrackletMCMHeader->pid0;
              break;
            case 1:
              qa = mTrackletMCMHeader->pid1;
              break;
            case 2:
              qa = mTrackletMCMHeader->pid2;
              break;
            default:
              LOG(warn) << "mcmtrackletcount is not in [0:2] count=" << mcmtrackletcount << " headertrackletcount=" << headertrackletcount << " something very wrong parsing the TrackletMCMData fields with data of : 0x" << std::hex << mTrackletMCMData->word;
              //mEventRecord.ErrorStats[TRDParsingTrackletInvalidTrackletCount]++;
              if (mOptions[TRDEnableRootOutputBit]) {
                mParsingErrors->Fill(TRDParsingTrackletInvalidTrackletCount);
                increment2dHist(TRDParsingTrackletInvalidTrackletCount);
              }
              //this should have been caught above by the headertrackletcount to mcmtrackletcount
              ignoreDataTillTrackletEndMarker = true;
              break;
          }
          if (!ignoreDataTillTrackletEndMarker) {
            q0 = getQFromRaw(mTrackletMCMHeader, mTrackletMCMData, 0, mcmtrackletcount);
            q1 = getQFromRaw(mTrackletMCMHeader, mTrackletMCMData, 1, mcmtrackletcount);
            q2 = getQFromRaw(mTrackletMCMHeader, mTrackletMCMData, 2, mcmtrackletcount);
            int padrow = mTrackletMCMHeader->padrow;
            int col = mTrackletMCMHeader->col;
            int pos = mTrackletMCMData->pos;
            int slope = mTrackletMCMData->slope;
            int hcid = mDetector * 2 + mRobSide;
            if (mDataVerbose) {
              if (mTrackletHCHeaderState) {
                LOG(info) << "Tracklet HCID : " << hcid << " mDetector:" << mDetector << " robside:" << mRobSide << " " << mTrackletMCMHeader->padrow << ":" << mTrackletMCMHeader->col << " ---- " << mTrackletHCHeader->supermodule << ":" << mTrackletHCHeader->stack << ":" << mTrackletHCHeader->layer << ":" << mTrackletHCHeader->side << " rawhcheader : 0x" << std::hex << std::hex << mTrackletHCHeader->word;
              } else {
                LOG(info) << "Tracklet HCID : " << hcid << " mDetector:" << mDetector << " robside:" << mRobSide << " " << mTrackletMCMHeader->padrow << ":" << mTrackletMCMHeader->col;
              }
            }
            //TODO cross reference hcid to somewhere for a check. mDetector is assigned at the time of parser init.
            //mEventRecord.TrackletCounts[mcm][mcmtrackletcount]++;
            mEventRecord->getTracklets().emplace_back(4, hcid, padrow, col, pos, slope, q0, q1, q2); // our format is always 4
            if (mDataVerbose) {
              LOG(info) << "Tracklet added:" << 4 << "-" << hcid << "-" << padrow << "-" << col << "-" << pos << "-" << slope << "-" << q0 << ":" << q1 << ":" << q2;
              Tracklet64 a(4, hcid, padrow, col, pos, slope, q0, q1, q2);
              LOG(info) << "Tracklet added:" << a;
              LOG(info) << "Tracklet64 output ended";
            }
            mTrackletsFound++;
            mcmtrackletcount++;
            if (mcmtrackletcount == headertrackletcount) { // headertrackletcount and mcmtrackletcount are not zero based counting
              // at the end of the tracklet output of this mcm
              // next to come can either be an mcmheaderword or a trackletendmarker.
              // check next word if its a trackletendmarker
              auto nextdataword = std::next(word, 1);
              // the check is unambigous between trackletendmarker and mcmheader
              if ((*nextdataword) == constants::TRACKLETENDMARKER) {
                mState = StateTrackletEndMarker;
              } else {
                mState = StateTrackletMCMHeader;
              }
            }
          }
        }
      }
    } // else
    trackletloopcount++;
  } //end of for loop
  //sanity check
  //LOG(warn) << " end of Trackelt parsing but we are exiting with out a tracklet end marker with " << mWordsRead << " 32bit words read";
  //mEventRecord.ErrorStats[TRDParsingTrackletExitingNoTrackletEndMarker]++;
  if (mOptions[TRDEnableRootOutputBit]) {
    mParsingErrors->Fill(TRDParsingTrackletExitingNoTrackletEndMarker);
    increment2dHist(TRDParsingTrackletExitingNoTrackletEndMarker);
  }
  mTrackletparsetime += std::chrono::high_resolution_clock::now() - parsetimestart;
  mTrackletParsingBad = true;
  return mWordsRead;
}

} // namespace o2::trd
