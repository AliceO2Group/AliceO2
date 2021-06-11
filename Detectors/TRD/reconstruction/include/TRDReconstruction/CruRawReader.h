// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CruRawReader.h
/// @author Sean Murray
/// @brief  Cru raw data reader, this is the part that parses the raw data
//          it runs on the flp(pre compression) or on the epn(pre tracklet64 array generation)
//          it hands off blocks of cru pay load to the parsers.

#ifndef O2_TRD_CRURAWREADER
#define O2_TRD_CRURAWREADER

#include <fstream>
#include <iostream>
#include <string>
#include <cstdint>
#include <array>
#include <vector>
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTRD/RawData.h"
#include "TRDReconstruction/DigitsParser.h"
#include "TRDReconstruction/TrackletsParser.h"
#include "DataFormatsTRD/Constants.h"
#include "TRDBase/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTRD/EventRecord.h"

namespace o2::trd
{
class Tracklet64;
class TriggerRecord;
class Digit;

class CruRawReader
{

  static constexpr bool debugparsing = true;
  enum CRUSate { CRUStateHalfCRUHeader = 0,
                 CRUStateHalfCRU };
  enum Dataformats { TrackletsDataFormat = 0,
                     DigitsDataFormat,
                     TestPatternDataFormat,
                     ConfigEventDataFormat };

 public:
  CruRawReader() = default;
  ~CruRawReader() = default;

  bool run();

  void checkSummary();
  void resetCounters();
  void configure(bool byteswap, bool verbose, bool headerverbose, bool dataverbose)
  {
    mByteSwap = byteswap;
    mVerbose = verbose;
    mHeaderVerbose = headerverbose;
    mDataVerbose = dataverbose;
  }
  void setBlob(bool returnblob) { mReturnBlob = returnblob; }; //set class to produce blobs and not vectors. (compress vs pass through)`
  void setDataBuffer(const char* val)
  {
    mDataBuffer = val;
  };
  void setDataBufferSize(long val)
  {
    if (mVerbose) {
      LOG(info) << " Setting buffer size to : " << val;
    }
    mDataBufferSize = val;
  };
  void setVerbose(bool verbose) { mVerbose = verbose; }
  void setDataVerbose(bool verbose) { mDataVerbose = verbose; }
  void setHeaderVerbose(bool verbose) { mHeaderVerbose = verbose; }
  inline uint32_t getDecoderByteCounter() const { return reinterpret_cast<const char*>(mDataPointer) - mDataBuffer; };
  bool buildBlobOutput(char* outputbuffer); // should probably go into a writer object.
  // benchmarks
  double mIntegratedBytes = 0.;
  double mIntegratedTime = 0.;

  std::vector<Tracklet64>& getTracklets(InteractionRecord& ir) { return mEventRecords.getTracklets(ir); };
  std::vector<Digit>& getDigits(InteractionRecord& ir) { return mEventRecords.getDigits(ir); };
  //  std::vector<o2::trd::TriggerRecord> getIR() { return mEventTriggers; }
  void getParsedObjects(std::vector<Tracklet64>& tracklets, std::vector<Digit>& cdigits, std::vector<TriggerRecord>& triggers);
  int getDigitsFound() { return mTotalDigitsFound; }
  int getTrackletsFound() { return mTotalTrackletsFound; }
  int sumTrackletsFound() { return mEventRecords.sumTracklets(); }
  int sumDigitsFound() { return mEventRecords.sumDigits(); }
  void clearall()
  {
    mEventRecords.clear();
    clear();
  }
  void clear()
  {
    mTrackletsParser.clear();
    mDigitsParser.clear();
  }

 protected:
  bool processHBFs(int datasizealreadyread = 0, bool verbose = false);
  bool processHBFsa(int datasizealreadyread = 0, bool verbose = false);
  bool buildCRUPayLoad();
  int processHalfCRU(int cruhbfstartoffset);
  bool processCRULink();
  bool skipRDH();

  inline void rewind()
  {
    if (mVerbose) {
      LOG(info) << "rewinding crurawreader incoming data buffer";
    }
    mDataPointer = reinterpret_cast<const uint32_t*>(mDataBuffer);
  };

  int mJumpRDH = 0;
  bool mVerbose{false};
  bool mHeaderVerbose{false};
  bool mDataVerbose{false};
  bool mByteSwap{false};
  const char* mDataBuffer = nullptr;
  static const uint32_t mMaxHBFBufferSize = o2::trd::constants::HBFBUFFERMAX;
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX> mHBFPayload; //this holds the O2 payload held with in the HBFs to pass to parsing.
  uint32_t mHalfCRUPayLoadRead{0};                                    // the words current read in for the currnt cru payload.
  uint32_t mO2PayLoadRead{0};                                         // the words current read in for the currnt cru payload.
  int mCurrentHalfCRULinkHeaderPoisition = 0;
  // no need to waste time doing the copy  std::array<uint32_t,8> mCurrentCRUWord; // data for a cru comes in words of 256 bits.
  uint32_t mCurrentLinkDataPosition256;    // count of data read for current link in units of 256 bits
  uint32_t mCurrentLinkDataPosition;       // count of data read for current link in units of 256 bits
  uint32_t mCurrentHalfCRUDataPosition256; //count of data read for this half cru.
  uint32_t mTotalHalfCRUDataLength;
  uint32_t mTotalHalfCRUDataLength256;

  uint32_t mTotalTrackletsFound{0};
  uint32_t mTotalDigitsFound{0};

  long mDataBufferSize;
  uint64_t mDataReadIn = 0;
  const uint32_t* mDataPointer = nullptr; // pointer to the current position in the rdh
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
  uint32_t mCRUState; // the state of what we are expecting to read currently from the data stream, *not* what we have just read.
  bool mError = false;
  bool mFatal = false;
  uint32_t mSaveBufferDataSize = 0;
  uint32_t mSaveBufferDataLeft = 0;
  uint32_t mcruFeeID = 0;
  uint32_t mDatareadfromhbf;
  uint32_t mTotalHBFPayLoad = 0; // total data payload of the heart beat frame in question.
  uint32_t mHBFoffset32 = 0;     // total data payload of the heart beat frame in question.
  //pointers to the data as we read them in, again no point in copying.
  HalfCRUHeader* mhalfcruheader;
  o2::InteractionRecord mIR;

  bool checkerCheck();
  void checkerCheckRDH();
  int mState; // basic state machine for where we are in the parsing.
  // we parse rdh to rdh but data is cru to cru.
  //the relevant parsers. Not elegant but we need both so pointers to base classes and sending them in with templates or some other such mechanism seems impossible, or its just late and I cant think.
  //TODO think of a more elegant way of incorporating the parsers.
  TrackletsParser mTrackletsParser;
  DigitsParser mDigitsParser;
  //used to surround the outgoing data with a coherent rdh coming from the incoming stream.
  o2::header::RDHAny* mOpenRDH;
  o2::header::RDHAny* mCloseRDH;

  uint32_t mEventCounter;
  uint32_t mFatalCounter;
  uint32_t mErrorCounter;

  EventStorage mEventRecords; // store data range indexes into the above vectors.
  bool mReturnBlob{0};        // whether to return blobs or vectors;
  struct TRDDataCounters_t {  //thisis on a per event basis
    //TODO this should go into a dpl message for catching by qc ?? I think.
    std::array<uint32_t, 1080> LinkWordCounts;    //units of 256bits "cru word"
    std::array<uint32_t, 1080> LinkPadWordCounts; // units of 32 bits the data pad word size.
    std::array<uint32_t, 1080> LinkFreq;          //units of 256bits "cru word"
    //from the above you can get the stats for supermodule and detector.
    std::array<bool, 1080> LinkEmpty; // Link only has padding words, probably not serious in pp.
    uint32_t EmptyLinks;
    //maybe change this to actual traps ?? but it will get large.
    std::array<uint32_t, 1080> LinkTrackletPerTrap1; // incremented if a trap on this link has 1 tracklet
    std::array<uint32_t, 1080> LinkTrackletPerTrap2; // incremented if a trap on this link has 2 tracklet
    std::array<uint32_t, 1080> LinkTrackletPerTrap3; // incremented if a trap on this link has 3 tracklet
    std::vector<uint32_t> EmptyTraps;                // MCM indexes of traps that are empty ?? list might better
  } TRDStatCounters;

  /** summary data **/
};

} // namespace o2::trd
// namespace o2

#endif
