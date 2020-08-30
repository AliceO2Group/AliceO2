// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Cru2Tracklet.h
/// @author Sean Murray
/// @brief  Cru data to tracklet converter "compressor"

#ifndef O2_TRD_CRU2TRACKLETTRANSLATOR
#define O2_TRD_CRU2TRACKLETTRANSLATOR

#include <fstream>
#include <string>
#include <cstdint>
#include <array>
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"

namespace o2
{
namespace trd
{

class Cru2TrackletTranslator
{

  static constexpr bool debugparsing = true;
  enum CRUSate { CRUStateHalfCRUHeader,
                 CRUStateHalfChamber,
                 CRUStateTrackletMCMHeader,
                 CRUStateTrackletMCMData,
                 CRUStatePadding };

 public:
  Cru2TrackletTranslator() = default;
  ~Cru2TrackletTranslator() = default;

  inline bool run()
  {
    rewind();
    uint32_t dowhilecount = 0;
    do {
      LOG(info) << "do while loop count " << dowhilecount++;
      LOG(info) << " data readin : " << mDataReadIn;
      LOG(info) << " mDataBuffer :" << (void*)mDataBuffer;
      processHBFs();
    } while (mDataReadIn < mDataBufferSize);

    return false;
  };

  void checkSummary();
  void resetCounters();

  void setDataBuffer(const char* val) { mDataBuffer = val; };
  void setDataBufferSize(long val) { mDataBufferSize = val; };

  inline uint32_t getDecoderByteCounter() const { return reinterpret_cast<const char*>(mDataPointer) - mDataBuffer; };

  // benchmarks
  double mIntegratedBytes = 0.;
  double mIntegratedTime = 0.;

 protected:
  uint32_t processHBFs();
  bool buildCRUPayLoad();
  bool processHalfCRU();
  bool processCRULink();

  /** decoder private functions and data members **/

  inline void rewind() { mDataPointer = reinterpret_cast<const uint32_t*>(mDataBuffer); };

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
  HalfCRUHeader mCurrentHalfCRUHeader; // are we waiting for new header or currently parsing the payload of on
  uint16_t mCurrentLink;               // current link within the halfcru we are parsing 0-14
  uint16_t mCRUEndpoint;               // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID;
  uint16_t mHCID;
  uint16_t mFEEID; // current Fee ID working on
  std::array<uint32_t, 15> mCurrentHalfCRULinkLengths;
  std::array<uint32_t, 15> mCurrentHalfCRULinkErrorFlags;
  // no need to waste time doing the copy  std::array<uint32_t,8> mCurrentCRUWord; // data for a cru comes in words of 256 bits.
  uint32_t mCurrentLinkDataPosition256;    // count of data read for current link in units of 256 bits
  uint32_t mCurrentLinkDataPosition;       // count of data read for current link in units of 256 bits
  uint32_t mCurrentHalfCRUDataPosition256; //count of data read for this half cru.
  uint32_t mTotalHalfCRUDataLength;
  uint32_t mCRUState; // the state of what we are expecting to read currently from the data stream, *not* what we have just read.
  bool mError = false;
  bool mFatal = false;
  char mSaveBuffer[1048576];
  uint32_t mSaveBufferDataSize = 0;
  uint32_t mSaveBufferDataLeft = 0;
  uint32_t mcruFeeID = 0;
  //pointers to the data as we read them in, again no point in copying.
  HalfCRUHeader* mhalfcruheader;
  TrackletHCHeader* mTrackletHCHeader;
  TrackletMCMHeader* mTrackletMCMHeader;
  TrackletMCMData* mTrackletMCMData;
  /** checker private functions and data members **/

  bool checkerCheck();
  void checkerCheckRDH();
  int mState; // basic state machine for where we are in the parsing.
              // we parse rdh to rdh but data is cru to cru.
  uint32_t mEventCounter;
  uint32_t mFatalCounter;
  uint32_t mErrorCounter;

  std::array<std::vector<Tracklet64>, 72> mEventTracklets; // when this runs properly it will only 6 for the flp its runnung on.
  std::vector<std::array<uint32_t, 72>> mEventStartPositions;
  struct TRDDataCounters_t {
    std::array<uint32_t, 1080> LinkWordCounts;    //units of 256bits "cru word"
    std::array<uint32_t, 1080> LinkPadWordCounts; // units of 32 bits the data word size.
    std::array<uint32_t, 1080> LinkFreq;          //units of 256bits "cru word"
                                                  //from the above you can get the stats for supermodule and detector.
  } TRDStatCounters;

  /** summary data **/
};

} // namespace trd
} // namespace o2

#endif /** O2_TOF_COMPRESSOR **/
