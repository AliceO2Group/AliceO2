// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"

namespace o2::trd
{
//TODO put o2::trd::constants::HBFBUFFERMAX in constants
//
class TrackletsParser
{
 public:
  TrackletsParser() = default;
  ~TrackletsParser() = default;
  void setData(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data) { mData = data; }
  int Parse(); // presupposes you have set everything up already.
  int Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start,
            std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, uint32_t feeid, int robside, int detector, int stack, int layer, bool cleardigits = false, bool disablebyteswap = false, bool verbose = true, bool headerverbose = false, bool dataverbose = false) //, std::array<uint32_t, 15>& lengths) // change to calling per link.
  {
    mStartParse = start;
    mEndParse = end;
    mDetector = detector;
    mFEEID = feeid;
    mRobSide = robside;
    mStack = stack;
    mLayer = layer;
    setData(data);
    setVerbose(verbose, headerverbose, dataverbose);
    setByteSwap(disablebyteswap);
    mWordsRead = 0;
    mDataWordsParsed = 0;
    mTrackletsFound = 0;
    mPaddingWordsCounter = 0;
    //    mTracklets.clear();
    return Parse();
  };
  void setVerbose(bool verbose, bool header = false, bool data = false)
  {
    mVerbose = verbose;
    mHeaderVerbose = header;
    mDataVerbose = data;
  }
  void setByteSwap(bool swap) { mByteOrderFix = swap; }
  int getDataWordsParsed() { return mDataWordsParsed; }
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
  void clear()
  {
    mTracklets.clear();
  }

 private:
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* mData;
  std::vector<Tracklet64> mTracklets;
  // pointers to keep track of the currently parsing headers and data.
  TrackletHCHeader* mTrackletHCHeader;
  TrackletMCMHeader* mTrackletMCMHeader;
  TrackletMCMData* mTrackletMCMData;

  int mState;               // state that the parser is currently in.
  int mDataWordsParsed;     // count of data wordsin data that have been parsed in current call to parse.
  int mTrackletsFound;      // tracklets found in the data block, mostly used for debugging.
  int mPaddingWordsCounter; // count of padding words encoutnered
  Tracklet64 mCurrentTrack; // the current track we are looking at, used to accumulate the possibly 3 tracks from the parsing 4 incoming data words
  int mWordsRead;           // number of words read frombuffer
  bool mVerbose{false};     // user verbose output, put debug statement in output from commandline.
  bool mHeaderVerbose{false};
  bool mDataVerbose{false};

  bool mIgnoreTrackletHCHeader{false}; // Is the data with out the tracklet HC Header? defaults to having it in.
  bool mByteOrderFix{false};           // simulated data is not byteswapped, real is, so deal with it accodringly.
  bool mReturnVector{false};           // whether weare returing a vector or the raw data buffer.

  uint16_t mEventCounter;
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, effectively the link limits to parse on.
  //uint32_t mCurrentLinkDataPosition256;                // count of data read for current link in units of 256 bits

  uint16_t mCurrentLink; // current link within the halfcru we are parsing 0-14
  uint16_t mCRUEndpoint; // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID;
  uint16_t mHCID;
  uint16_t mDetector;
  uint16_t mRobSide;
  uint16_t mStack;
  uint16_t mLayer;
  uint16_t mFEEID; // current Fee ID working on
  uint16_t mMCM;
  uint16_t mROB;
  //  std::array<uint32_t, 16> mAverageNumTrackletsPerTrap; TODO come back to this stat.
};

} // namespace o2::trd

#endif // O2_TRD_TRACKLETPARSER
