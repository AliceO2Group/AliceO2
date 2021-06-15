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
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"

#include <fstream>

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
  //  void setLinkLengths(std::array<uint32_t, 15>& lengths) { mCurrentHalfCRULinkLengths = lengths; };
  int Parse(bool verbose = false); // presupposes you have set everything up already.
  int Parse(std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>* data, std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator start,
            std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator end, int detector, bool cleardigits = false, bool disablebyteswap = false, bool verbose = false, bool headerverbose = false, bool dataverbose = false)
  {
    setData(data);
    //   setLinkLengths(lengths);
    mStartParse = start;
    mEndParse = end;
    mDetector = detector;
    setVerbose(verbose, headerverbose, dataverbose);
    if (cleardigits) {
      clearDigits();
    }
    setByteSwap(disablebyteswap);
    mReturnVectorPos = 0;
    return Parse();
  };
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

 private:
  int mState;
  int mDataWordsParsed; // count of data wordsin data that have been parsed in current call to parse.
  int mDigitsFound;     // digits found in the data block, mostly used for debugging.
  int mBufferLocation;
  int mPaddingWordsCounter;
  bool mSanityCheck{true};
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
  int mParsedWords{0}; // words parsed in data vector, last complete bit is not parsed, and left for another round of data update.
  DigitHCHeader* mDigitHCHeader;
  DigitMCMHeader* mDigitMCMHeader;
  DigitMCMADCMask* mDigitMCMADCMask;
  uint32_t mADCMask;
  DigitMCMData* mDigitMCMData;
  bool mVerbose{false};
  bool mHeaderVerbose{false};
  bool mDataVerbose{false};
  bool mvVerbose{false};
  uint16_t mDetector;
  uint16_t mMCM;
  uint16_t mROB;
  uint16_t mChannel;
  uint16_t mEventCounter;
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, effectively the link limits to parse on.
                                                                                           // std::array<uint16_t, 60>/*constants::TIMEBINS>*/ mADCValues;
  //uint32_t mCurrentLinkDataPosition256;                // count of data read for current link in units of 256 bits
  //uint32_t mCurrentLinkDataPosition;                   // count of data read for current link in units of 256 bits
  //uhint32_t mCurrentHalfCRUDataPosition256;             //count of data read for this half cru.
  //  std::array<uint32_t, 15> mCurrentHalfCRULinkLengths; // not in units of 256 bits or 32 bytes or 8 words
};

} // namespace o2::trd

#endif // O2_TRD_DIGITSPARSER
