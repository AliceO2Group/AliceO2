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

// TODO come back and figure which of below headers I actually need.
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iomanip>
#include <iostream>

namespace o2::trd
{

int TrackletsParser::Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data,
                           std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start,
                           std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end,
                           TRDFeeID feeid, int halfchamberside, int detector, int stack, int layer,
                           EventRecord* eventrecord, EventStorage* eventrecords, std::bitset<16> options, int usetracklethcheader)
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
  mWordsRead = 0;
  mWordsDumped = 0;
  mTrackletsFound = 0;
  mPaddingWordsCounter = 0;
  mTrackletHCHeaderState = usetracklethcheader; // what to with the tracklet half chamber header 0,1,2
  mIgnoreTrackletHCHeader = options[TRDIgnoreTrackletHCHeaderBit];
  mEventRecord = eventrecord;
  mEventRecords = eventrecords;
  if (std::distance(start, end) > o2::trd::constants::MAXDATAPERLINK32) { // full event is all digits and 3 tracklets per mcm,
    // sanity check that the length of data to scan is less the possible maximum for a link
    LOG(warn) << "Attempt to parse a block of data for tracklets that is longer than a link can poossibly be : " << std::distance(start, end) << " should be less than : " << o2::trd::constants::MAXDATAPERLINK32 << " dumping this block of data.";
    return -1;
  }
  if (eventrecord == nullptr) {
    return -1;
  }
  if (eventrecords == nullptr) {
    return -1;
  }
  return Parse();
}

void TrackletsParser::incParsingError(int error, std::string message)
{
  int sector = mFEEID.supermodule;
  int stack = mStack;
  int layer = mLayer;
  int side = mHalfChamberSide;
  int hcid = sector * constants::NCHAMBER * constants::NSTACK * constants::NLAYER + stack * constants::NSTACK * constants::NLAYER + layer + side;
  if (side > 1 || side < 0) {
    side = 0;
  }
  if (mFEEID.supermodule > 17 || mFEEID.supermodule < 0) {
    sector = 0;
  }
  if (mStack > 4 || mStack < 0) {
    stack = 0;
  }
  if (layer > 5 || mLayer < 0) {
    layer = 0;
  }
  // error is too big ?
  if (mOptions[TRDGenerateStats] && error <= TRDLastParsingError) {
    mEventRecords->incParsingError(error, hcid);
  }
  if (mOptions[TRDVerboseErrorsBit]) {
    LOG(info) << "PE Parsing Error : " << o2::trd::ParsingErrorsString[error] << " sector:stack:layer:side :: " << sector << ":" << stack << ":" << layer << ":" << side;
    if (message != "") {
      LOG(info) << message;
    }
  }
  if (mOptions[TRDVerboseLinkBit]) {
    mDumpLink = true;
  }
}

void TrackletsParser::OutputIncomingData()
{
  LOG(info) << "TrackletParse incoming data stream from " << std::hex << mStartParse << " to " << mEndParse;
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
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << *word;
    } else {
      outputstring << " 0x" << std::hex << std::setfill('0') << std::setw(8) << *word;
    }
    word++;
    wordcount++;
  }
  LOG(info) << "TrackletParse end of incoming stream";
}

int TrackletsParser::Parse()
{
  // we are handed the buffer payload of an rdh and need to parse its contents.
  // producing a vector of tracklets.

  mTrackletParsingBad = false;
  //mData holds a buffer containing tracklets parse placing tracklets in the output vector.
  mCurrentLink = 0;
  mWordsRead = 0;
  mTrackletsFound = 0;
  if (mTrackletHCHeaderState == 0) {
    // tracklet hc header is never present
    mState = StateTrackletMCMHeader;
    LOG(error) << " This option of TrackletHalfChamberHeader 0 is no longer permitted";
    return -1;
  } else {
    if (mTrackletHCHeaderState == 1) {
      // we either have a tracklet half chamber header word or a digit one.
      // digit has 01 at the end last 2 bits and the supermodule in bits 9 to 13 [0:17]
      // tracklet has bit 11 (zero based) set to 1
      // we have tracklet data so no TrackletHCHeader
      mState = StateTrackletHCHeader;
      TrackletHCHeader hcheader;

      hcheader.word = *mStartParse;
      DigitHCHeader digithcheader; // candidate digit half chamber header.
      digithcheader.word = *mStartParse;
      if (!sanityCheckTrackletHCHeader(hcheader)) {
        //we dont have a tracklethcheader so no tracklet data.
        if (mOptions[TRDVerboseBit]) {
          LOG(info) << "Returning 0 from tracklet parsing " << std::hex << (digithcheader.res & 0x3) << " supermodule : " << (digithcheader.supermodule & 0x1f);
        }

        return -1; // mWordsRead;
      }
      // NBNBNBNB
      // digit half chamber header ends with 01b and has the supermodule in position (9-13).
      // this of course can conflict with a tracklet hc header, hence should not be used!
      // NBNBNBNB
      if ((digithcheader.res & 0x3) == 0x1 && (((digithcheader.supermodule) & 0x1f) == mHCID / constants::NHCPERSEC)) {
        if (mOptions[TRDVerboseBit]) {
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
  std::array<uint8_t, 3> qcharges;
  bool ignoreDataTillTrackletEndMarker = false;
  // used for when we need to dump the rest of the tracklet data.
  if (std::distance(mStartParse, mEndParse) > o2::trd::constants::MAXDATAPERLINK32) {
    // full event is all digits and 3 tracklets per mcm, and all associated headers,
    LOG(warn) << "Attempt to parse a block of data for tracklets that is longer than a \
                  link can poossibly be : "
              << std::distance(mStartParse, mEndParse) << " should be less than : " << o2::trd::constants::MAXDATAPERLINK32 << " dumping this data.";
    // sanity check that the length of data to scan is less the possible maximum for a link
    return -1;
  }
  for (auto word = mStartParse; word < mEndParse; ++word) {
    // loop over the entire data buffer (a complete link of tracklets and digits)
    if (mOptions[TRDVerboseWordBit]) {
      LOGF(info, "parsing word:0x%08x", *word);
    }

    if (mState == StateFinished) {
      return mWordsRead;
    }
    // loop over all the words ...
    // check first for tracklet end marker 0x1000 0x1000
    int index = std::distance(mStartParse, word);
    int indexend = std::distance(word, mEndParse);
    std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator nextword = word;
    std::advance(nextword, 1);
    uint32_t nextwordcopy = *nextword;

    if (mOptions[TRDByteSwapBit]) {
      HelperMethods::swapByteOrder(*word);
      HelperMethods::swapByteOrder(nextwordcopy);
    }

    if (*word == 0x10001000 && nextwordcopy == 0x10001000) {
      if (!StateTrackletEndMarker && !StateTrackletHCHeader) {
        LOG(warn) << "State should be TrackletEndMarker, current ?= end marker  ?? " << mState << " ?=" << StateTrackletEndMarker;
      }

      mWordsRead += 2;

      if (mOptions[TRDVerboseBit]) {
        LOG(info) << "***TrackletEndMarker : 0x" << std::hex << *word << " and 0x" << nextwordcopy << " at offset " << std::distance(mStartParse, word);
      }

      mState = StateFinished;
      return mWordsRead;
    }
    if (*word == o2::trd::constants::CRUPADDING32) {
      // padding word first as it clashes with the hcheader.
      mState = StatePadding;
      incParsingError(TrackletCRUPaddingWhileParsingTracklets, fmt::format("Tracklet Parsing but found Padding word at offset : {}", std::distance(mStartParse, word)));
      mWordsDumped = std::distance(word, mEndParse);
      ignoreDataTillTrackletEndMarker = true;
      word = mEndParse;
      // TODO remove tracklets already added erroneously
      continue; // bail out
    } else {
      if (ignoreDataTillTrackletEndMarker) {
        mWordsDumped++;
        continue; // go back to the start of loop, walk the data till the above code of the tracklet end marker is hit, padding is hit or we get to the end of the data.
      }
      if (mState == StateTrackletHCHeader) {
        if (mOptions[TRDVerboseBit]) {
          LOG(info) << "mFEEID : 0x" << std::hex << mFEEID.word << " supermodule : 0x" << (int)mFEEID.supermodule << " tracklethcheader : 0x" << *word;
          TrackletHCHeader a;
          a.word = *word;
          printTrackletHCHeader(a);
        }
      }
      // now for Tracklet hc header
      if ((isTrackletHCHeader(*word)) && !mIgnoreTrackletHCHeader && mState == StateTrackletHCHeader) { // TrackletHCHeader has bit 11 set to 1 always. Check for state because raw data can have bit 11 set!
        if (mState != StateTrackletHCHeader) {
          incParsingError(TrackletTrackletHCHeaderButWrongState, fmt::format("mFEEID : {:#08x} supermodule : {:#08x} trackletheader : {:#08x}", mFEEID.word, (int)mFEEID.supermodule, *word));
        }
        // we actually have a header word.
        mTrackletHCHeader.word = *word;
        if (mOptions[TRDVerboseWordBit]) {
          TrackletHCHeader a;
          a.word = *word;
          printTrackletHCHeader(a);
        }
        // sanity check of trackletheader ??
        if (!sanityCheckTrackletHCHeader(mTrackletHCHeader)) {
          incParsingError(TrackletHCHeaderSanityCheckFailure);
          TrackletHCHeader a;
          a.word = *word;
          printTrackletHCHeader(a);
          // now dump and run
        }
        mWordsRead++;
        mState = StateTrackletMCMHeader;                                      // now we should read a MCMHeader next time through loop
      } else {                                                                // not TrackletHCHeader
        if (isTrackletMCMHeader(*word) && mState == StateTrackletMCMHeader) { // TrackletMCMHeader has the bits on either end always 1
          // mcmheader
          mTrackletMCMHeader = (TrackletMCMHeader*)&(*word);
          if (mOptions[TRDVerboseWordBit]) {
            TrackletMCMHeader a;
            a.word = *word;
            printTrackletMCMHeader(a);
          }
          qcharges.fill(0xff);
          if (!sanityCheckTrackletMCMHeader(mTrackletMCMHeader)) {
            incParsingError(TrackletMCMHeaderSanityCheckFailure);
            if (mOptions[TRDVerboseErrorsBit]) {
              TrackletMCMHeader a;
              a.word = *word;
              printTrackletMCMHeader(a);
            }
            mWordsDumped = std::distance(word, mEndParse);
            ignoreDataTillTrackletEndMarker = true;
            word = mEndParse;
            continue;
          }
          headertrackletcount = getNumberOfTrackletsFromHeader(mTrackletMCMHeader);
          if (headertrackletcount > 0) {
            mState = StateTrackletMCMData; // after reading a header we should then have data for next round through the loop
          } else {
            mState = StateTrackletMCMHeader;
          }

          mcmtrackletcount = 0;
          std::fill(mTrackletMCMData.begin(), mTrackletMCMData.end(), TrackletMCMData{0});
          mWordsRead++;
        } else {
          if (mState == StateTrackletMCMHeader || (mState == StateTrackletHCHeader && !mOptions[mIgnoreTrackletHCHeader])) {
            // if we are here something is wrong, dump the data. The else of line 227 should imply we are in StateTrackletMCMData;
            ignoreDataTillTrackletEndMarker = true;
            incParsingError(TrackletStateMCMHeaderButParsingMCMData, fmt::format("Parsing MCMDATA but state is MCMHeader and word is : Raw{:#08x} unpacked 3 options below", *word));
            if (mOptions[TRDVerboseErrorsBit]) {
              TrackletMCMData a;
              TrackletMCMHeader b;
              a.word = *word;
              b.word = *word;
              printTrackletMCMData(a);
              printTrackletMCMHeader(b);
            }
            mWordsDumped = std::distance(word, mEndParse);
            ignoreDataTillTrackletEndMarker = true;
            word = mEndParse;
            continue;
          }
          mState = StateTrackletMCMData;
          // tracklet data;
          mTrackletMCMData[mcmtrackletcount].word = *word;
          mWordsRead++;
          if (mOptions[TRDVerboseWordBit]) {
            printTrackletMCMData(mTrackletMCMData[mcmtrackletcount]);
          }
          // do we have more tracklets than the header allows?
          if (headertrackletcount < mcmtrackletcount) {
            ignoreDataTillTrackletEndMarker = true;
            // dump the rest of the data ... undo any tracklets already written?
            // cant dump till mEndParse and digits are after the tracklets
            // we assume the mcmtrackletcount (n from the end) last tracklets in the vector are to be removed.
            incParsingError(TrackletTrackletCountGTThatDeclaredInMCMHeader, fmt::format(" mcmtrackletcount: {} headertrackletcount:{}", mcmtrackletcount, headertrackletcount));
            if (mOptions[TRDVerboseErrorsBit]) {
              LOG(info) << mTrackletMCMHeader;
            }
            mEventRecord->popTracklets(mcmtrackletcount); // our format is always 4
          }
          // take the header and this data word and build the underlying 64bit tracklet.
          int q0, q1, q2;
          if (mcmtrackletcount > 2) {
            incParsingError(TrackletInvalidTrackletCount, fmt::format("mcmtrackletcount is not in [0:2] count={} headertrackletcount {} error in TrackletMCMData of : {:#08x}", mcmtrackletcount, headertrackletcount, *word));
            // this should have been caught above by the headertrackletcount to mcmtrackletcount
            ignoreDataTillTrackletEndMarker = true;
            break;
          }
          if (!ignoreDataTillTrackletEndMarker) {
            auto validheader = getChargesFromRawHeaders(mTrackletHCHeader, mTrackletMCMHeader, mTrackletMCMData, qcharges, mcmtrackletcount);
            if (validheader == -1) {
              //tracklet unpacking went wrong.
              incParsingError(TrackletTrackletCountGTThatDeclaredInMCMHeader);
              mWordsDumped = std::distance(word, mEndParse);
              ignoreDataTillTrackletEndMarker = true;
              word = mEndParse;
              continue;
            }
            int padrow = mTrackletMCMHeader->padrow;
            int col = mTrackletMCMHeader->col;
            int pos = mTrackletMCMData[mcmtrackletcount].pos;
            int slope = mTrackletMCMData[mcmtrackletcount].slope;
            // The 8-th bit of position and slope are always flipped in the FEE.
            // We flip them back while reading the raw data so that they are stored
            // without flipped bits in the CTFs
            pos = pos ^ 0x80;
            slope = slope ^ 0x80;
            int hcid = mDetector * 2 + mHalfChamberSide;
            if (mOptions[TRDVerboseBit]) {
              if (mTrackletHCHeaderState) {
                LOG(info) << "Tracklet HCID : " << hcid << " mDetector:" << mDetector << " robside:"
                          << mHalfChamberSide << " " << mTrackletMCMHeader->padrow << ":"
                          << mTrackletMCMHeader->col << " ---- " << mTrackletHCHeader.supermodule
                          << ":" << mTrackletHCHeader.stack << ":" << mTrackletHCHeader.layer << ":"
                          << mTrackletHCHeader.side << " rawhcheader : 0x" << std::hex << std::hex
                          << mTrackletHCHeader.word;
              } else {
                LOG(info) << "Tracklet HCID : " << hcid << " mDetector:" << mDetector << " robside:"
                          << mHalfChamberSide << " " << mTrackletMCMHeader->padrow << ":"
                          << mTrackletMCMHeader->col;
              }
            }
            //TODO cross reference hcid to somewhere for a check. mDetector is assigned at the time of parser init.
            if (mOptions[TRDVerboseBit]) {
              LOG(info) << "TTT format : " << (int)mTrackletHCHeader.format << " hcid: " << hcid
                        << " padrow:" << padrow << " col:" << col << " pos:" << pos << " slope:"
                        << slope << " q::" << q0 << " " << q1 << " " << q2;
            }
            mEventRecord->getTracklets().emplace_back((int)mTrackletHCHeader.format, hcid, padrow,
                                                      col, pos, slope, q0, q1, q2);
            mEventRecord->incTrackletsFound(1);
            mTrackletsFound++;
            mcmtrackletcount++;
            if (mcmtrackletcount == headertrackletcount) {
              // headertrackletcount and mcmtrackletcount are not zero based counting
              // at the end of the tracklet output of this mcm
              // next to come can either be an mcmheaderword or a trackletendmarker.
              // check next word if its a trackletendmarker
              auto nextdataword = std::next(word, 1);
              // the check is unambiguous between trackletendmarker and mcmheader
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
  } // end of for loop
  // sanity check
  incParsingError(TrackletExitingNoTrackletEndMarker);

  mTrackletParsingBad = true;
  return mWordsRead;
}

} // namespace o2::trd
