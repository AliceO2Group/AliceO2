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

// Cru raw data reader, this is the part that parses the raw data
// it runs on the flp(pre compression) or on the epn(pre tracklet64 array generation)
// it hands off blocks of cru pay load to the parsers.

#ifndef O2_TRD_CRURAWREADER
#define O2_TRD_CRURAWREADER

#include <fstream>
#include <iostream>
#include <string>
#include <cstdint>
#include <array>
#include <vector>
#include <chrono>
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/RawDataStats.h"
#include "TRDReconstruction/DigitsParser.h"
#include "TRDReconstruction/TrackletsParser.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "TRDReconstruction/EventRecord.h"

#include "TH2F.h"

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
  void configure(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::bitset<16> options)
  {
    mVerbose = options[TRDVerboseBit];
    mHeaderVerbose = options[TRDHeaderVerboseBit];
    mDataVerbose = options[TRDDataVerboseBit];
    mFixDigitEndCorruption = options[TRDFixDigitCorruptionBit];
    mTrackletHCHeaderState = tracklethcheader;
    mHalfChamberWords = halfchamberwords;
    mHalfChamberMajor = halfchambermajor;
    mRootOutput = options[TRDEnableRootOutputBit];
    mEnableTimeInfo = options[TRDEnableTimeInfoBit];
    mEnableStats = options[TRDEnableStatsBit];
    mOptions = options;
    mTimeBins = constants::TIMEBINS; // set to value from constants incase the DigitHCHeader1 header is not present.
    mPreviousDigitHCHeadersvnver = 0xffffffff;
    mPreviousDigitHCHeadersvnrver = 0xffffffff;
  }

  void setMaxErrWarnPrinted(int nerr, int nwar)
  {
    mMaxErrsPrinted = nerr < 0 ? std::numeric_limits<int>::max() : nerr;
    mMaxWarnPrinted = nwar < 0 ? std::numeric_limits<int>::max() : nwar;
  }
  void checkNoWarn();
  void checkNoErr();

  void setBlob(bool returnblob) { mReturnBlob = returnblob; }; //set class to produce blobs and not vectors. (compress vs pass through)`
  void setDataBuffer(const char* val)
  {
    mDataBuffer = val;
    if (mVerbose) {
      if (val == nullptr) {
        LOG(error) << "Data buffer is being assigned to a null ptr";
      }
    }
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
  void getParsedObjectsandClear(std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits, std::vector<TriggerRecord>& triggers);
  void buildDPLOutputs(o2::framework::ProcessingContext& outputs);
  int getDigitsFound() { return mTotalDigitsFound; }
  int getTrackletsFound() { return mTotalTrackletsFound; }
  int sumTrackletsFound() { return mEventRecords.sumTracklets(); }
  int sumDigitsFound() { return mEventRecords.sumDigits(); }
  int getWordsRead() { return mWordsAccepted + mTotalDigitWordsRead + mTotalTrackletWordsRead; }
  int getWordsRejected() { return mWordsRejected + mTotalDigitWordsRejected + mTotalTrackletWordsRejected; }

  std::shared_ptr<EventStorage*> getEventStorage() { return std::make_shared<EventStorage*>(&mEventRecords); }
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
  void OutputHalfCruRawData();
  // void setStats(o2::trd::TRDDataCountersPerTimeFrame* trdstats){mTimeFrameStats=trdstats;}
  //void setHistos(std::array<TH2F*, 10> hist, std::array<TH2F*, constants::MAXPARSEERRORHISTOGRAMS> parsingerrors2d)

 protected:
  bool processHBFs(int datasizealreadyread = 0, bool verbose = false);
  bool buildCRUPayLoad();
  int processHalfCRU(int cruhbfstartoffset, int numberOfPreviousCRU, unsigned int maxdatawrittentobuffer);
  bool processCRULink();
  int parseDigitHCHeader();
  int checkDigitHCHeader();
  int checkTrackletHCHeader();
  bool compareRDH(const o2::header::RDHAny* firstrdh, const o2::header::RDHAny* rdh);
  bool checkRDH(const o2::header::RDHAny* rdh);
  bool skipRDH();
  void updateLinkErrorGraphs(int currentlinkindex, int supermodule_half, int stack_layer);
  void incrementErrors(int hist, int sector = -1, int side = 0, int stack = 0, int layer = 0)
  {
    if (sector > 17) {
      sector = 0;
    }
    if (stack > 4) {
      stack = 0;
    }
    if (layer > 5) {
      layer = 0;
    }
    if (sector < -1) {
      sector = 0;
    }
    mEventRecords.incParsingError(hist, sector, side, stack * constants::NLAYER + layer);
    if (mVerbose) {
      LOG(info) << "Parsing error: " << hist << " sector:" << sector << " side:" << side << " stack:" << stack << " layer:" << layer;
    }
  }
  void dumpRDHAndNextHeader(const o2::header::RDHAny* rdh);

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
  bool mFixDigitEndCorruption{false};
  int mTrackletHCHeaderState{0};
  int mHalfChamberWords{0};
  int mHalfChamberMajor{0};
  bool mRootOutput{0};
  bool mEnableTimeInfo{0};
  bool mEnableStats{0};
  std::bitset<16> mOptions;
  const char* mDataBuffer = nullptr;
  static const uint32_t mMaxHBFBufferSize = o2::trd::constants::HBFBUFFERMAX;
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX> mHBFPayload; //this holds the O2 payload held with in the HBFs to pass to parsing.
  uint32_t mHalfCRUPayLoadRead{0};                                    // the words current read in for the currnt cru payload.
  uint32_t mO2PayLoadRead{0};                                         // the words current read in for the currnt cru payload.
  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX>::iterator mStartParse, mEndParse; // limits of parsing, start and end points for parsing.
  std::array<uint16_t, constants::TIMEBINS> mADCValues{};
  int mCurrentHalfCRULinkHeaderPoisition = 0;
  // no need to waste time doing the copy  std::array<uint32_t,8> mCurrentCRUWord; // data for a cru comes in words of 256 bits.
  uint32_t mCurrentLinkDataPosition256;    // count of data read for current link in units of 256 bits
  uint32_t mCurrentLinkDataPosition;       // count of data read for current link in units of 256 bits
  uint32_t mCurrentHalfCRUDataPosition256; //count of data read for this half cru.
  uint32_t mTotalHalfCRUDataLength;
  uint32_t mTotalHalfCRUDataLength256;

  uint32_t mTotalTrackletsFound{0};
  uint32_t mTotalDigitsFound{0};

  int mMaxErrsPrinted = 20;
  int mMaxWarnPrinted = 20;

  long mDataBufferSize;
  uint32_t mDataReadIn = 0;
  const uint32_t* mDataPointer = nullptr; // pointer to the current position in the rdh
  const uint32_t* mDataPointerMax = nullptr;
  const uint32_t* mDataEndPointer = nullptr;
  const uint32_t* mDataPointerNext = nullptr;
  uint8_t mDataNextWord = 1;
  uint8_t mDataNextWordStep = 2;

  const o2::header::RDHAny* mDataRDH;
  HalfCRUHeader mCurrentHalfCRUHeader; // are we waiting for new header or currently parsing the payload of on
  HalfCRUHeader mPreviousHalfCRUHeader; // are we waiting for new header or currently parsing the payload of on
  DigitHCHeader mDigitHCHeader;        // Digit HalfChamber header we are currently on.
  DigitHCHeader1 mDigitHCHeader1;      // this and the next 2 are option are and variable in order, hence
  uint16_t mTimeBins;
  DigitHCHeader2 mDigitHCHeader2;      // the individual seperation instead of an array.
  DigitHCHeader3 mDigitHCHeader3;
  uint32_t mPreviousDigitHCHeadersvnver;  // svn ver in the digithalfchamber header, used for validity checks
  uint32_t mPreviousDigitHCHeadersvnrver; // svn release ver also used for validity checks
  TrackletHCHeader mTrackletHCHeader;  // Tracklet HalfChamber header we are currently on.
  uint16_t mCurrentLink;               // current link within the halfcru we are parsing 0-14
  uint16_t mCRUEndpoint;               // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID;
  uint16_t mHCID;
  TRDFeeID mFEEID; // current Fee ID working on
  // the store of the 3 ways we can determine this information, link,rdh,halfchamber
  std::array<int, 3> mDetector;
  std::array<int, 3> mSector;
  std::array<int, 3> mStack;
  std::array<int, 3> mLayer;
  std::array<int, 3> mSide;
  std::array<int, 3> mEndPoint;
  std::array<int, 3> mHalfChamberSide;
  int mWhichData; // index used into the above arrays once decided on which source is "correct"
  o2::InteractionRecord mIR;
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
  uint32_t mDigitWordsRead = 0;
  uint32_t mDigitWordsRejected = 0;
  uint32_t mTotalDigitWordsRead = 0;
  uint32_t mTotalDigitWordsRejected = 0;
  uint32_t mTrackletWordsRead = 0;
  uint32_t mTrackletWordsRejected = 0;
  uint32_t mTotalTrackletWordsRejected = 0;
  uint32_t mTotalTrackletWordsRead = 0;
  uint32_t mWordsRejected = 0; // those words rejected before tracklet and digit parsing together with the digit and tracklet rejected words;
  uint32_t mWordsAccepted = 0; // those words before before tracklet and digit parsing together with the digit and tracklet rejected words;
  //pointers to the data as we read them in, again no point in copying.
  HalfCRUHeader* mhalfcruheader;

  bool checkerCheck();
  void checkerCheckRDH();
  int mState; // basic state machine for where we are in the parsing.
  // we parse rdh to rdh but data is cru to cru.
  //the relevant parsers. Not elegant but we need both so pointers to base classes and sending them in with templates or some other such mechanism seems impossible, or its just late and I cant think.
  //TODO think of a more elegant way of incorporating the parsers.
  TrackletsParser mTrackletsParser;
  DigitsParser mDigitsParser;
  //used to surround the outgoing data with a coherent rdh coming from the incoming stream.
  const o2::header::RDHAny* mOpenRDH;
  const o2::header::RDHAny* mCloseRDH;

  uint32_t mEventCounter;
  uint32_t mFatalCounter;
  uint32_t mErrorCounter;

  EventStorage mEventRecords; // store data range indexes into the above vectors.
  EventRecord* mCurrentEvent; // the current event we are looking at, info extracted from cru half chamber header.

  bool mReturnBlob{0};        // whether to return blobs or vectors;
  o2::trd::TRDDataCountersRunning mStatCountersRunning;
};

} // namespace o2::trd
// namespace o2

#endif
