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
                           TRDFeeID feeid, int halfchamberside, int detector, int stack, int layer,
                           EventRecord* eventrecord, EventStorage* eventrecords, std::bitset<16> options, bool cleardigits, int usetracklethcheader)
{
  mStartParse = start;
  mEndParse = end;
  mDetector = detector;
  mFEEID = feeid;
  mHalfChamberSide = halfchamberside;
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
  mEventRecords = eventrecords;
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
}

int TrackletsParser::Parse()
{
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
  //TODO move this to the reader as done for the digithcheader
  //TODO this is in fact moved to outside parsing for new reader, left here so as to change as little code as possible.
  if (mTrackletHCHeaderState == 0) {
    // tracklet hc header is never present
    mState = StateTrackletMCMHeader;
    LOG(error) << " This option of TrackletHalfChamberHeader 0 is no longer permitted";
    return 0;
  } else {
    if (mTrackletHCHeaderState == 1) {
      // we either have a tracklet half chamber header word or a digit one.
      // digit has 01 at the end last 2 bits and the supermodule in bits 9 to 13 [0:17]
      // tracklet has bit 11 (zero based) set to 1
      //we have tracklet data so no TracletHCHeader
      mState = StateTrackletHCHeader;
      TrackletHCHeader hcheader;

      hcheader.word = *mStartParse;
      uint32_t tmpheader = *mStartParse;
      if (!trackletHCHeaderSanityCheck(hcheader)) {
        //we dont have a tracklethcheader so no tracklet data.
        if (mHeaderVerbose) {
          LOG(info) << "Returning 0 from tracklet parsing " << std::hex << (tmpheader & 0x3) << " supermodule : " << ((tmpheader >> 9) & 0x1f);
        }

        return 0; //mWordsRead;
      }
      //NBNBNBNB
      //digit half chamber header ends with 01b and has the supermodule in position (9-13).
      //this of course can conflict with a tracklet hc header, hence should not be used!
      //NBNBNBNB
      if ((tmpheader & 0x3) == 0x1 && (((tmpheader >> 9) & 0x1f) == mHCID / 30)) {
        if (mHeaderVerbose) {
          LOG(info) << " we seem to be on a digit halfchamber header";
        }
        return 0;
      }
      mState = StateTrackletHCHeader;
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
      return mWordsRead;
    }
    if (*word == o2::trd::constants::CRUPADDING32) {
      //padding word first as it clashes with the hcheader.
      mState = StatePadding;
      incParsingError(TRDParsingTrackletCRUPaddingWhileParsingTracklets);
      mWordsDumped = std::distance(word, mEndParse);
      ignoreDataTillTrackletEndMarker = true;
      word = mEndParse;
      //TODO remove tracklets already added erroneously
      continue; // bail out

    } else {
      if (ignoreDataTillTrackletEndMarker) {
        mWordsDumped++;
        incParsingError(TRDParsingTrackletBit11NotSetInTrackletHCHeader);
        continue; //go back to the start of loop, walk the data till the above code of the tracklet end marker is hit, padding is hit or we get to the end of the data.
      }
      //fix to missing bit on supermodule 16 and 17, to set the uniquely identifying bit.
      if (mState == StateTrackletHCHeader) {
        if ((mFEEID.supermodule > 15) && mOptions[TRDFixSM1617Bit] && mTrackletHCHeaderState == 2) {
          *word |= 1 << 11; //flip bit eleven for the tracklethcheader for the last 2 supermodules (bug/misconfiguration/broken/other) not sure why its like this yet, but it is.
        }
      }
      //now for Tracklet hc header
      if ((((*word) & (0x1 << 11)) != 0) && !mIgnoreTrackletHCHeader && mState == StateTrackletHCHeader) { //TrackletHCHeader has bit 11 set to 1 always. Check for state because raw data can have bit 11 set!
        if (mState != StateTrackletHCHeader) {
          incParsingError(TRDParsingTrackletBit11NotSetInTrackletHCHeader);
        }
        //read the header
        if (mHeaderVerbose) {
          LOG(info) << "*** TrackletHCHeader : 0x" << std::hex << *word << " at offset :0x" << std::distance(mStartParse, word);
        }
        //we actually have a header word.
        mTrackletHCHeader = (TrackletHCHeader*)&word;
        //sanity check of trackletheader ??
        if (!trackletHCHeaderSanityCheck(*mTrackletHCHeader)) {
          incParsingError(TRDParsingTrackletHCHeaderSanityCheckFailure);
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
            incParsingError(TRDParsingTrackletMCMHeaderSanityCheckFailure);
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
            incParsingError(TRDParsingTrackletStateMCMHeaderButParsingMCMData);
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
            incParsingError(TRDParsingTrackletTrackletCountGTThatDeclaredInMCMHeader);
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
              LOG(alarm) << "mcmtrackletcount is not in [0:2] count=" << mcmtrackletcount << " headertrackletcount=" << headertrackletcount << " something very wrong parsing the TrackletMCMData fields with data of : 0x" << std::hex << mTrackletMCMData->word;
              incParsingError(TRDParsingTrackletInvalidTrackletCount);
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
            int hcid = mDetector * 2 + mHalfChamberSide;
            if (mHeaderVerbose) {
              if (mTrackletHCHeaderState) {
                LOG(info) << "Tracklet HCID : " << hcid << " mDetector:" << mDetector << " robside:" << mHalfChamberSide << " " << mTrackletMCMHeader->padrow << ":" << mTrackletMCMHeader->col << " ---- " << mTrackletHCHeader->supermodule << ":" << mTrackletHCHeader->stack << ":" << mTrackletHCHeader->layer << ":" << mTrackletHCHeader->side << " rawhcheader : 0x" << std::hex << std::hex << mTrackletHCHeader->word;
              } else {
                LOG(info) << "Tracklet HCID : " << hcid << " mDetector:" << mDetector << " robside:" << mHalfChamberSide << " " << mTrackletMCMHeader->padrow << ":" << mTrackletMCMHeader->col;
              }
            }
            //TODO cross reference hcid to somewhere for a check. mDetector is assigned at the time of parser init.
            mEventRecord->getTracklets().emplace_back(4, hcid, padrow, col, pos, slope, q0, q1, q2); // our format is always
            mEventRecord->incTrackletsFound(1);
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
  incParsingError(TRDParsingTrackletExitingNoTrackletEndMarker);

  mTrackletParsingBad = true;
  return mWordsRead;
}

} // namespace o2::trd
