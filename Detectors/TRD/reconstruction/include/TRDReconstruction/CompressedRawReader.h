// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedRawReader.h
/// @author Sean Murray
/// @brief  compressed raw reader, this is the part that parses the raw data
//          after it is compressed on the flp in the compressor, effectively a decompressor.
//          Data format is simply an event based vector of tracklet64, and the event header is
//          a custom 64 bit "tracklet" with even info and an offset to the next header (blocksize)

#ifndef O2_TRD_COMPRESSEDRAWREADER
#define O2_TRD_COMPRESSEDRAWREADER

#include <fstream>
#include <string>
#include <cstdint>
#include <array>
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/CompressedDigit.h"
#include "TRDBase/Digit.h"

namespace o2
{
namespace trd
{

class CompressedRawReader
{

  static constexpr bool debugparsing = true;
  enum CRUSate { CompressedStateHeader,
                 CompressedStateTracklets,
                 CompressedStatePadding };

 public:
  CompressedRawReader() = default;
  ~CompressedRawReader() = default;

  inline bool run()
  {
    rewind();
    uint32_t dowhilecount = 0;
    do {
      int datareadfromhbf = processHBFs();
    } while (mDataReadIn < mDataBufferSize);

    return false;
  };

  void checkSummary();
  void resetCounters();

  void setDataBuffer(const char* val) { mDataBuffer = val; };
  void setDataBufferSize(long val) { mDataBufferSize = val; };

  inline uint32_t getDecoderByteCounter() const { return reinterpret_cast<const char*>(mDataPointer) - mDataBuffer; };

  void setVerbosity(bool v) { mVerbose = v; };
  void setDataVerbosity(bool v) { mDataVerbose = v; };
  void setHeaderVerbosity(bool v) { mHeaderVerbose = v; };
  void configure(bool byteswap, bool verbose, bool headerverbose, bool dataverbose)
  {
    mByteSwap = byteswap;
    mVerbose = verbose;
    mHeaderVerbose = headerverbose;
    mDataVerbose = dataverbose;
  }
  std::vector<Tracklet64>& getTracklets() { return mEventTracklets; };
  std::vector<Digit>& getDigits() { return mEventDigits; };
  std::vector<CompressedDigit>& getCompressedDigits() { return mCompressedEventDigits; };
  std::vector<o2::trd::TriggerRecord> getIR() { return mEventTriggers; }

 protected:
  uint32_t processHBFs();
  bool buildCRUPayLoad();
  bool processBlock();

  /** decoder private functions and data members **/

  inline void rewind()
  {
    LOG(debug) << "!!!rewinding";
    mDataPointer = reinterpret_cast<const uint32_t*>(mDataBuffer);
  };

  int mJumpRDH = 0;

  std::ifstream mDecoderFile;
  const char* mDataBuffer = nullptr;
  long mDataBufferSize;
  uint64_t mDataReadIn = 0;
  const uint32_t* mDataPointer = nullptr;
  const uint32_t* mDataPointerMax = nullptr;
  const uint32_t* mDataEndPointer = nullptr;
  const uint32_t* mDataPointerNext = nullptr;
  uint8_t mDataNextWord = 1;
  uint8_t mDataNextWordStep = 2;
  const o2::header::RDHAny* mDataRDH;
  // no need to waste time doing the copy  std::array<uint32_t,8> mCurrentCRUWord; // data for a cru comes in words of 256 bits.
  uint32_t mCurrentLinkDataPosition; // count of data read for current link in units of 256 bits
  uint32_t mCompressedState;         // the state of what we are expecting to read currently from the data stream, *not* what we have just read.
  bool mError = false;
  bool mFatal = false;
  uint16_t mCurrentLink; // current link within the halfcru we are parsing 0-14
  uint16_t mCRUEndpoint; // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID;
  uint16_t mHCID;
  uint16_t mFEEID; // current Fee ID working on
  //pointers to the data as we read them in, again no point in copying.
  /** checker private functions and data members **/

  bool checkerCheck();
  void checkerCheckRDH();
  bool mVerbose{false};
  bool mHeaderVerbose{false};
  bool mDataVerbose{false};
  bool mByteSwap{true};
  int mState; // basic state machine for where we are in the parsing.
              // we parse rdh to rdh but data is cru to cru.
  uint32_t mEventCounter;
  uint32_t mFatalCounter;
  uint32_t mErrorCounter;

  std::vector<Tracklet64> mEventTracklets; // when this runs properly it will only 6 for the flp its runnung on.
  std::vector<o2::trd::TriggerRecord> mEventTriggers;
  std::vector<Digit> mEventDigits;
  std::vector<CompressedDigit> mCompressedEventDigits;
  o2::InteractionRecord mIR;
  struct TRDDataCounters_t {
    std::array<uint32_t, 1080> LinkWordCounts;    //units of 256bits "cru word"
    std::array<uint32_t, 1080> LinkPadWordCounts; // units of 32 bits the data word size.
    std::array<uint32_t, 1080> LinkFreq;          //units of 256bits "cru word"
  } TRDStatCounters;

  /** summary data **/
};

} // namespace trd
} // namespace o2

#endif
