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

/// @file   TrackletsParser.h
/// @author Sean Murray
/// @brief  TRD parse tracklet o2 payoload and build tracklets.

#ifndef O2_TRD_TRACKLETPARSER
#define O2_TRD_TRACKLETPARSER

#include <fstream>
#include <vector>
#include <bitset>

#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "TRDReconstruction/EventRecord.h"

namespace o2::trd
{

class TrackletsParser
{
 public:
  TrackletsParser() = default;
  ~TrackletsParser() = default;
  void setData(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data) { mData = data; }
  int Parse(); // presupposes you have set everything up already.
  int Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start, std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, TRDFeeID feeid, int robside,
            int detector, int stack, int layer, EventRecord* eventrecord, EventStorage* eventrecords, std::bitset<16> option, bool cleardigits = false, int usetracklethcheader = 0);
  void setVerbose(bool verbose, bool header = false, bool data = false)
  {
    mVerbose = verbose;
    mHeaderVerbose = header;
    mDataVerbose = data;
  }
  void setByteSwap(bool swap) { mByteOrderFix = swap; }
  int getDataWordsRead() { return mWordsRead; }
  int getDataWordsDumped() { return mWordsDumped; }
  int getTrackletsFound() { return mTrackletsFound; }
  void setIgnoreTrackletHCHeader(bool ignore) { mIgnoreTrackletHCHeader = ignore; }
  bool getIgnoreTrackletHCHeader() { return mIgnoreTrackletHCHeader; }
  enum TrackletParserState { StateTrackletHCHeader, // always the start of a half chamber.
                             StateTrackletMCMHeader,
                             StateTrackletMCMData,
                             StatePadding,
                             StateTrackletEndMarker,
                             StateFinished };
  std::vector<Tracklet64>& getTracklets() { return mTracklets; }
  inline void swapByteOrder(unsigned int& ui);
  bool getTrackletParsingState() { return mTrackletParsingBad; }
  void clear()
  {
    mTracklets.clear();
  }
  void OutputIncomingData();

  void incParsingError(int error)
  {
    int sector = mFEEID.supermodule;
    int stack = mStack;
    int layer = mLayer;
    int side = mHalfChamberSide;
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
      mEventRecords->incParsingError(error, sector, side, stack * constants::NLAYER + layer);
    }
  }

 private:
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* mData;
  std::vector<Tracklet64> mTracklets;
  // pointers to keep track of the currently parsing headers and data.
  TrackletHCHeader mTrackletHCHeader;
  TrackletMCMHeader* mTrackletMCMHeader;
  std::array<TrackletMCMData, 3> mTrackletMCMData;

  int mState{0};            // state that the parser is currently in.
  int mWordsRead{0};        // number of words read from buffer
  uint64_t mWordsDumped{0}; // number of words ignored from buffer
  int mTrackletsFound{0};   // tracklets found in the data block, mostly used for debugging.
  int mPaddingWordsCounter{0}; // count of padding words encoutnered
  Tracklet64 mCurrentTrack; // the current track we are looking at, used to accumulate the possibly 3 tracks from the parsing 4 incoming data words
  bool mVerbose{false};     // user verbose output, put debug statement in output from commandline.
  bool mHeaderVerbose{false};
  bool mDataVerbose{false};
  int mTrackletHCHeaderState{0}; //what to with the tracklet half chamber header 0,1,2
  bool mIgnoreTrackletHCHeader{false}; // Is the data with out the tracklet HC Header? defaults to having it in.
  bool mByteOrderFix{false};           // simulated data is not byteswapped, real is, so deal with it accodringly.
  std::bitset<16> mOptions;
  bool mTrackletParsingBad{false}; // store weather we should dump the rest of the link buffer after working through this tracklet buffer.
  uint16_t mEventCounter{0};
  std::chrono::duration<double> mTrackletparsetime;                                        // store the time it takes to parse
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, effectively the link limits to parse on.
  //uint32_t mCurrentLinkDataPosition256;                // count of data read for current link in units of 256 bits
  EventRecord* mEventRecord;
  EventStorage* mEventRecords;

  uint16_t mCurrentLink{0}; // current link within the halfcru we are parsing 0-14
  uint16_t mCRUEndpoint{0}; // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID{0};
  uint16_t mHCID{0};
  uint16_t mDetector{0};
  uint16_t mHalfChamberSide{0};
  uint16_t mStack{0};
  uint16_t mLayer{0};
  TRDFeeID mFEEID; // current Fee ID working on
  uint16_t mMCM{0};
  uint16_t mROB{0};
  //  std::array<uint32_t, 16> mAverageNumTrackletsPerTrap; TODO come back to this stat.
};

} // namespace o2::trd

#endif // O2_TRD_TRACKLETPARSER
