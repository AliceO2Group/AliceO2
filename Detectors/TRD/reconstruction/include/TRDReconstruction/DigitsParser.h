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

/// @file   EpnRawReaderTask.h
/// @author Sean Murray
/// @brief  TRD cru output data to tracklet task

#ifndef O2_TRD_DIGITSPARSER
#define O2_TRD_DIGITSPARSER

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"

#include <fstream>
#include <bitset>
#include "TH1F.h"
#include "TH2F.h"
#include "TList.h"

//using namespace o2::framework;

namespace o2::trd
{
class Digit;
class EventRecord;
// class to Parse a single link of digits data.
// calling class splits data by link and this gets called per link.

class DigitsParser
{

 public:
  DigitsParser() = default;
  ~DigitsParser() = default;
  void setData(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data) { mData = data; }
  int Parse(bool verbose = false); // presupposes you have set everything up already.
  int Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start,
            std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, int detector, int stack, int layer, int side, DigitHCHeader& hcheader,
            TRDFeeID& feeid, unsigned int linkindex, EventRecord* eventrecord, std::bitset<16> options, bool cleardigits = false);

  enum DigitParserState { StateDigitHCHeader, // always the start of a half chamber.
                          StateDigitMCMHeader,
                          StateDigitMCMData,
                          StatePadding,
                          StateDigitEndMarker };

  inline void swapByteOrder(unsigned int& word);

  int getDigitsFound() { return mDigitsFound; }
  bool getVerbose() { return mVerbose; }
  void setVerbose(bool value, bool header, bool data)
  {
    mVerbose = value;
    mHeaderVerbose = header;
    mDataVerbose = data;
  }
  void setByteSwap(bool byteswap) { mByteOrderFix = byteswap; }
  std::vector<Digit>& getDigits() { return mDigits; }
  void clearDigits() { mDigits.clear(); }
  void clear() { mDigits.clear(); }
  uint64_t getDumpedDataCount() { return mWordsDumped; }
  uint64_t getDataWordsParsed() { return mDataWordsParsed; }
  void tryFindMCMHeaderAndDisplay(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse);
  //void tryFindMCMHeaderAndDisplay(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse);
  void OutputIncomingData();
  void setParsingHisto(TH1F* parsingerrors, TList* parsingerrors2d)
  {
    mParsingErrors = parsingerrors;
    mParsingErrors2d = parsingerrors2d;
  }
  void increment2dHist(int hist)
  {
    ((TH2F*)mParsingErrors2d->At(hist))->Fill(mFEEID.supermodule * 2 + mFEEID.side, mStack * constants::NLAYER + mLayer);
  }

 private:
  int mState;
  int mDataWordsParsed; // count of data wordsin data that have been parsed in current call to parse.
  int mDigitsFound;     // digits found in the data block, mostly used for debugging.
  int mBufferLocation;
  int mPaddingWordsCounter;
  bool mSanityCheck{true};
  bool mDumpUnknownData{false}; // if the various sanity checks fail, bail out and dump the rest of the data, keeps stats.
  bool mByteOrderFix{false}; // simulated data is not byteswapped, real is, so deal with it accodringly.
  bool mReturnVector{true};  // whether we are returing a vector or the raw data buffer.
  // yes this is terrible design but it works,
  int mReturnVectorPos;

  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* mData = nullptr; // compressed data return space.
  std::vector<Digit> mDigits;                                              // outgoing parsed digits
  // subtle point, mDigits is not cleared between parsings,only between events.
  // this means that successive calls to Parse simply appends the new digits onto the vector.
  // at the end of the event the calling object must pull/copy the vector and clear or clear on next parse.
  //
  uint64_t mWordsDumped{0}; // words rejected for various reasons.
  DigitHCHeader mDigitHCHeader;
  DigitMCMHeader* mDigitMCMHeader;
  DigitMCMADCMask* mDigitMCMADCMask;
  uint32_t mADCMask;
  DigitMCMData* mDigitMCMData;
  EventRecord* mEventRecord;
  bool mVerbose{false};
  bool mHeaderVerbose{false};
  bool mDataVerbose{false};
  bool mvVerbose{false};
  std::bitset<16> mOptions;
  bool mIgnoreDigitHCHeader{false}; // whether to ignore the contents of the half chamber header and take the rdh/cru header as authoritative

  uint16_t mDetector;
  uint16_t mMCM;
  uint16_t mROB;
  uint16_t mCurrentADCChannel;
  uint16_t mDigitWordCount;
  uint16_t mStack;
  uint16_t mLayer;
  uint16_t mSector;
  uint16_t mSide;

  uint16_t mEventCounter;
  TRDFeeID mFEEID;
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, effectively the link limits to parse on.
  std::array<uint16_t, constants::TIMEBINS> mADCValues{};
  std::array<uint16_t, constants::MAXMCMCOUNT> mMCMstats; // bit pattern for errors current event for a given mcm;
  TH1F* mParsingErrors;
  TList* mParsingErrors2d;
};

} // namespace o2::trd

#endif // O2_TRD_DIGITSPARSER
