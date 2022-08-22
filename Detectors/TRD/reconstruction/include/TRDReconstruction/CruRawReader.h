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
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "TRDReconstruction/EventRecord.h"

namespace o2::trd
{
class Tracklet64;
class TriggerRecord;
class Digit;

class CruRawReader
{
 public:
  CruRawReader() = default;
  ~CruRawReader() = default;

  // top-level method, takes the full payload data of the DPL raw data message and delegates to processHBFs()
  // probably this method can be removed and we can directly go to processHBFs()
  void run();

  // configure the raw reader, done once at the init() stage
  void configure(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::bitset<16> options);

  // settings in order to avoid InfoLogger flooding
  void setMaxErrWarnPrinted(int nerr, int nwar)
  {
    mMaxErrsPrinted = nerr < 0 ? std::numeric_limits<int>::max() : nerr;
    mMaxWarnPrinted = nwar < 0 ? std::numeric_limits<int>::max() : nwar;
  }
  void checkNoWarn();
  void checkNoErr();

  // set the input data buffer
  void setDataBuffer(const char* val) { mDataBufferPtr = val; }

  // set the input data buffer size in bytes
  void setDataBufferSize(long val) { mDataBufferSize = val; }

  // set the mapping from Link ID to HCID and vice versa
  void setLinkMap(const LinkToHCIDMapping* map) { mLinkMap = map; }

  // probably better to make mEventRecords available to the outside and then use that directly, can clean up this header a lot more
  void buildDPLOutputs(o2::framework::ProcessingContext& outputs);
  int getDigitsFound() { return mTotalDigitsFound; }
  int getTrackletsFound() { return mTotalTrackletsFound; }
  int sumTrackletsFound() { return mEventRecords.sumTracklets(); }
  int sumDigitsFound() { return mEventRecords.sumDigits(); }
  int getWordsRead() { return mWordsAccepted + mTotalDigitWordsRead + mTotalTrackletWordsRead; }
  int getWordsRejected() { return mWordsRejected + mTotalDigitWordsRejected + mTotalTrackletWordsRejected; }

  void clearall()
  {
    mEventRecords.clear();
    mDigitsParser.clear();
  }

  enum TrackletParserState { StateTrackletHCHeader,
                             StateTrackletMCMHeader,
                             StateTrackletMCMData,
                             StateMoveToEndMarker,
                             StateSecondEndmarker,
                             StateFinished };

 private:
  // the parsing starts here, payload from all available RDHs is copied into mHBFPayload and afterwards processHalfCRU() is called
  // returns the total number of bytes read, including RDH header
  int processHBFs();

  // process the data which is stored inside the mHBFPayload for the current half-CRU. The iteration corresponds to the trigger index inside the HBF
  bool processHalfCRU(int iteration);

  // parse the digit HC headers, possibly update settings as the number of time bins from the header word
  bool parseDigitHCHeaders(int hcid);

  // compare the link ID information from the digit HC header with what we know from RDH header
  void checkDigitHCHeader(int hcidRef);

  // helper function to compare two consecutive RDHs
  bool compareRDH(const o2::header::RDHAny* rdhPrev, const o2::header::RDHAny* rdhCurr);

  // sanity check on individual RDH (can find unconfigured FLP or invalid data)
  bool checkRDH(const o2::header::RDHAny* rdh);

  // given the total link size and the hcid from the RDH
  // parse the tracklet data. Overwrite hcid from TrackletHCHeader if mismatch is detected
  // trackletWordsRejected:  count the number of words which were skipped (subset of words read)
  // returns total number of words read (no matter if parsed successfully or not)
  int parseTrackletLinkData(int linkSize32, int& hcid, int& trackletWordsRejected);

  // check validity of TrackletHCHeader (always once bit needs to be set and hcid needs to be consistent with what we expect from RDH)
  // FIXME currently hcid can be overwritten from TrackletHCHeader
  bool isTrackletHCHeaderOK(const TrackletHCHeader& header, int& hcid);

  // helper function to create Tracklet64 from link data
  Tracklet64 assembleTracklet64(int format, TrackletMCMHeader& mcmHeader, TrackletMCMData& mcmData, int cpu, int hcid) const;

  // important function to keep track of all errors, if possible accounted to a certain link / half-chamber ID
  // FIXME:
  // - whenever we don't know the half-chamber, fill it for HCID == 1080 so that we can still disentangle errors from half-chamber 0?
  // - probably enough to only pass the half-chamber ID, if we know it
  // - or what about being more granular? dump ROB or MCM, if we know it?
  void incrementErrors(int error, int hcid = -1, std::string message = "");

  // helper function to dump the whole input payload including RDH headers
  void dumpInputPayload() const;

  // dump out a link with in a half cru buffer
  void outputLinkRawData(int link);

  // ###############################################################
  // ## class member variables
  // ###############################################################

  bool mFixDigitEndCorruption{false};
  int mTrackletHCHeaderState{0};
  int mHalfChamberWords{0};
  int mHalfChamberMajor{0};
  std::bitset<16> mOptions;

  std::array<uint32_t, o2::trd::constants::HBFBUFFERMAX> mHBFPayload; // this holds the O2 payload held with in the HBFs to pass to parsing.

  uint32_t mTotalTrackletsFound{0}; // accumulated number of tracklets found
  uint32_t mTotalDigitsFound{0};    // accumulated number of digits found

  // InfoLogger flood protection settings
  int mMaxErrsPrinted = 20;
  int mMaxWarnPrinted = 20;

  // helper pointers, counters for the input buffer
  const char* mDataBufferPtr = nullptr; // pointer to the beginning of the whole payload data
  long mDataBufferSize;                 // the total payload size of the raw data message from the FLP (typically a single HBF from one half-CRU)
  const char* mCurrRdhPtr = nullptr;    // points inside the payload data at the current RDH position
  uint32_t mTotalHBFPayLoad = 0;        // total data payload of the heart beat frame in question (up to wich array index mHBFPayload is filled with data)
  uint32_t mHBFoffset32 = 0;            // points to the current position inside mHBFPayload we are currently reading
  uint32_t mHalfCRUStartOffset = 0;     // store the start of the halfcru we are currently on.

  HalfCRUHeader mCurrentHalfCRUHeader; // are we waiting for new header or currently parsing the payload of on
  HalfCRUHeader mPreviousHalfCRUHeader; // are we waiting for new header or currently parsing the payload of on
  DigitHCHeader mDigitHCHeader;         // Digit HalfChamber header we are currently on.
  uint16_t mTimeBins;                  // the number of time bins to be read out (default 30, can be overwritten from digit HC header)
  bool mHaveSeenDigitHCHeader3{false};
  uint32_t mPreviousDigitHCHeadersvnver;  // svn ver in the digithalfchamber header, used for validity checks
  uint32_t mPreviousDigitHCHeadersvnrver; // svn release ver also used for validity checks
  TrackletHCHeader mTrackletHCHeader;  // Tracklet HalfChamber header we are currently on.
  uint16_t mCurrentLink;               // current link within the halfcru we are parsing 0-14
  uint16_t mCRUEndpoint;               // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID;                     // CRU ID taken from the FEEID of the RDH
  TRDFeeID mFEEID; // current Fee ID working on

  o2::InteractionRecord mIR;
  std::array<uint32_t, 15> mCurrentHalfCRULinkLengths;
  std::array<uint32_t, 15> mCurrentHalfCRULinkErrorFlags;

  const LinkToHCIDMapping* mLinkMap = nullptr; // to retrieve HCID from Link ID

  // FIXME for all counters need to check which one is really needed
  uint32_t mTotalDigitWordsRead = 0;
  uint32_t mTotalDigitWordsRejected = 0;
  uint32_t mTotalTrackletWordsRejected = 0;
  uint32_t mTotalTrackletWordsRead = 0;
  uint32_t mWordsRejected = 0; // those words rejected before tracklet and digit parsing together with the digit and tracklet rejected words;
  uint32_t mWordsAccepted = 0; // those words before before tracklet and digit parsing together with the digit and tracklet rejected words;

  DigitsParser mDigitsParser;

  EventStorage mEventRecords; // store data range indexes into the above vectors.
  EventRecord* mCurrentEvent; // the current event we are looking at, info extracted from cru half chamber header.
};

} // namespace o2::trd
// namespace o2

#endif
