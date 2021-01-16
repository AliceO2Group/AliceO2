// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <fstream>
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "TRDBase/Digit.h"

using namespace o2::framework;

namespace o2::trd
{

// class to Parse a single link of digits data.
// calling class splits data by link and this gets called per link.

class DigitsParser
{

 public:
  DigitsParser() = default;
  ~DigitsParser() = default;
  void setData(std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>* data) { mData = data; }
  //  void setLinkLengths(std::array<uint32_t, 15>& lengths) { mCurrentHalfCRULinkLengths = lengths; };
  int Parse(); // presupposes you have set everything up already.
  int Parse(std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>::iterator start,
            std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>::iterator end, int detector) //, std::array<uint32_t, 15>& lengths) // change to calling per link.
  {
    setData(data);
    //   setLinkLengths(lengths);
    mStartParse = start;
    mEndParse = end;
    mDetector = detector;
    return Parse();
  };
  enum DigitParserState { StateDigitHCHeader, // always the start of a half chamber.
                          StateDigitMCMHeader,
                          StateDigitMCMData,
                          StatePadding };

  inline void swapByteOrder(unsigned int& word);

 private:
  int mState;
  int mDataWordsParsed; // count of data wordsin data that have been parsed in current call to parse.
  int mDigitsFound;     // tracklets found in the data block, mostly used for debugging.
  int mBufferLocation;
  std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>* mData = nullptr; // parsed in vector of raw data to parse.
  std::vector<Digit> mDigits;                                              // outgoing parsed digits
  std::vector<TriggerRecord> mTriggerRecords;                              // trigger records to index into the digits vector.
  int mParsedWords{0};                                                     // words parsed in data vector, last complete bit is not parsed, and left for another round of data update.
  DigitHCHeader* mDigitHCHeader;
  DigitMCMHeader* mDigitMCMHeader;
  DigitMCMData* mDigitMCMData;

  uint16_t mDetector;
  std::array<uint32_t, o2::trd::constants::CRUBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, effectively the link limits to parse on.
  //uint32_t mCurrentLinkDataPosition256;                // count of data read for current link in units of 256 bits
  //uint32_t mCurrentLinkDataPosition;                   // count of data read for current link in units of 256 bits
  //uhint32_t mCurrentHalfCRUDataPosition256;             //count of data read for this half cru.
  //  std::array<uint32_t, 15> mCurrentHalfCRULinkLengths; // not in units of 256 bits or 32 bytes or 8 words
};

} // namespace o2::trd

#endif // O2_TRD_DIGITSPARSER
