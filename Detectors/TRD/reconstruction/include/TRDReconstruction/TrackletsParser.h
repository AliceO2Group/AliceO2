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
//TODO put o2::trd::constants::CRUBUFFERMAX in constants
//
class TrackletsParser
{
 public:
  TrackletsParser() = default;
  ~TrackletsParser() = default;
  void setData(std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>* data) { mData = data; }
  int Parse(); // presupposes you have set everything up already.
  int Parse(std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>::iterator start,
            std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>::iterator end, int detector, bool cleardigits = false, bool verbose = true) //, std::array<uint32_t, 15>& lengths) // change to calling per link.
  {
    mStartParse = start;
    mEndParse = end;
    mDetector = detector;
    setData(data);
    return Parse();
  };
  void setVerbose(bool verbose) { mVerbose = verbose; }
  int getDataWordsParsed() { return mDataWordsParsed; }
  int getTrackletsFound() { return mTrackletsFound; }
  enum TrackletParserState { StateTrackletHCHeader, // always the start of a half chamber.
                             StateTrackletMCMHeader,
                             StateTrackletMCMData,
                             StatePadding };
  std::vector<Tracklet64>& getTracklets() { return mTracklets; }
  inline void swapByteOrder(unsigned int& ui);

 private:
  std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>* mData;
  std::vector<Tracklet64> mTracklets;
  int mState;
  int mDataWordsParsed; // count of data wordsin data that have been parsed in current call to parse.
  int mTrackletsFound;  // tracklets found in the data block, mostly used for debugging.
  int mPaddingWordsCounter; // count of padding words encoutnered
  Tracklet64 mCurrentTrack;
  int mWordsRead;
  bool mVerbose{false};
  TrackletHCHeader* mTrackletHCHeader;
  TrackletMCMHeader* mTrackletMCMHeader;
  TrackletMCMData* mTrackletMCMData;

  bool mDisableByteOrderFix{false}; // simulated data is not byteswapped, real is, so deal with it accodringly.
  bool mReturnVector{false};        // whether weare returing a vector or the raw data buffer.
  
  uint16_t mDetector;
  uint16_t mMCM;
  uint16_t mROB;
  uint16_t mEventCounter;
  std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, effectively the link limits to parse on.
  //uint32_t mCurrentLinkDataPosition256;                // count of data read for current link in units of 256 bits

  uint16_t mCurrentLink; // current link within the halfcru we are parsing 0-14
  uint16_t mCRUEndpoint; // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID;
  uint16_t mHCID;
  uint16_t mFEEID; // current Fee ID working on
  //  std::array<uint32_t, 16> mAverageNumTrackletsPerTrap; TODO come back to this stat.
};

} // namespace o2::trd

#endif // O2_TRD_TRACKLETPARSER
