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
#include "DataFormatsTRD/RawDataStats.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "TRDReconstruction/EventRecord.h"
#include <fstream>
#include <bitset>

//using namespace o2::framework;

namespace o2::trd
{
class Digit;
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
            std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, int detector, int stack, int layer, int side, DigitHCHeader& hcheader, uint16_t timebins,
            TRDFeeID& feeid, unsigned int linkindex, EventRecord* eventrecord, EventStorage* eventrecords, std::bitset<16> options);

  enum DigitParserState { StateDigitHCHeader, // always the start of a half chamber.
                          StateDigitMCMHeader,
                          StateDigitMCMData,
                          StatePadding,
                          StateDigitEndMarker };


  int getDigitsFound() { return mDigitsFound; }
  std::vector<Digit>& getDigits() { return mDigits; }
  void clearDigits() { mDigits.clear(); }
  void clear() { mDigits.clear(); }
  uint64_t getDumpedDataCount() { return mWordsDumped; }
  uint64_t getDataWordsParsed() { return mDataWordsParsed; }
  void OutputIncomingData();
  void checkNoErr();
  void checkNoWarn();
  void incParsingError(int error, std::string message = "");
  bool dumpLink()
  {
    if (mDumpLink) {
      mDumpLink = false;
      return true;
    }
    return false;
  }

 private:
  int mState;
  int mDataWordsParsed; // count of data wordsin data that have been parsed in current call to parse.
  int mDigitsFound;     // digits found in the data block, mostly used for debugging.
  int mBufferLocation;
  int mPaddingWordsCounter;
  bool mSanityCheck{true};
  bool mDumpUnknownData{false}; // if the various sanity checks fail, bail out and dump the rest of the data, keeps stats.
  bool mReturnVector{true};  // whether we are returing a vector or the raw data buffer.
  // yes this is terrible design but it works,
  int mReturnVectorPos;
  bool mDumpLink{false};

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
  EventStorage* mEventRecords;
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
  uint16_t mHalfChamberSide;
  uint16_t mStackLayer; //store these values to prevent numerous recalculation;
  uint16_t mTimeBins;   // timebins used defaults is constants::TIMEBINS

  uint16_t mEventCounter;
  TRDFeeID mFEEID;
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, effectively the link limits to parse on.
  std::array<uint16_t, constants::TIMEBINS> mADCValues{};
  int mMaxErrsPrinted;
};

} // namespace o2::trd

#endif // O2_TRD_DIGITSPARSER
