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

#include <string>
#include <cstdint>
#include <bitset>
#include <set>
#include <utility>
#include <array>
#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/RawDataStats.h"
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

  // both the tracklet and the digit parsing is implemented as a state machine
  enum ParsingState { StateTrackletHCHeader,
                      StateTrackletMCMHeader,
                      StateTrackletMCMData,
                      StateDigitMCMHeader,
                      StateDigitADCMask,
                      StateDigitMCMData,
                      StateMoveToDigitMCMHeader,
                      StateMoveToEndMarker,
                      StateSecondEndmarker,
                      StateFinished };

  // top-level method, takes the full payload data of the DPL raw data message and delegates to processHBFs()
  // probably this method can be removed and we can directly go to processHBFs()
  void run();

  // configure the raw reader, done once at the init() stage
  void configure(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::bitset<16> options);

  // set number of time bins to fixed value instead of reading from DigitHCHeader
  // (but still complain if DigitHCHeader is not consistent)
  void setNumberOfTimeBins(int tb)
  {
    mTimeBins = tb;
    mTimeBinsFixed = true;
  }

  // settings in order to avoid InfoLogger flooding
  void setMaxErrWarnPrinted(int nerr, int nwar)
  {
    mMaxErrsPrinted = nerr < 0 ? std::numeric_limits<int>::max() : nerr;
    mMaxWarnPrinted = nwar < 0 ? std::numeric_limits<int>::max() : nwar;
  }
  void checkNoWarn(bool silently = true);
  void checkNoErr();

  // set the input data buffer
  void setDataBuffer(const char* val) { mDataBufferPtr = val; }

  // set the input data buffer size in bytes
  void setDataBufferSize(long val) { mDataBufferSize = val; }

  // set the mapping from Link ID to HCID and vice versa
  void setLinkMap(const LinkToHCIDMapping* map) { mLinkMap = map; }

  // assemble output for full TF and send it out
  void buildDPLOutputs(o2::framework::ProcessingContext& outputs);

  int getDigitsFound() const { return mDigitsFound; }
  int getTrackletsFound() const { return mTrackletsFound; }

  int getWordsRejected() const { return mWordsRejected + mDigitWordsRejected + mTrackletWordsRejected; }

  // reset the event storage and the counters
  void reset();

  // the parsing starts here, payload from all available RDHs is copied into mHBFPayload and afterwards processHalfCRU() is called
  // returns the total number of bytes read, including RDH header
  int processHBFs();

  // process the data which is stored inside the mHBFPayload for the current half-CRU. The iteration corresponds to the trigger index inside the HBF
  bool processHalfCRU(int iteration);

  // parse the digit HC headers, possibly update settings as the number of time bins from the header word
  bool parseDigitHCHeaders(int hcid);

  // helper function to compare two consecutive RDHs
  bool compareRDH(const o2::header::RDHAny* rdhPrev, const o2::header::RDHAny* rdhCurr);

  // sanity check on individual RDH (can find unconfigured FLP or invalid data)
  bool checkRDH(const o2::header::RDHAny* rdh);

  // given the total link size and the hcid from the RDH
  // parse the tracklet data. Overwrite hcid from TrackletHCHeader if mismatch is detected
  // trackletWordsRejected:  count the number of words which were skipped (subset of words read)
  // trackletWordsReadOK: count the number of words which could be read consecutively w/o errors
  // numberOfTrackletsFound: count the number of tracklets found
  // returns total number of words read (no matter if parsed successfully or not) or -1 in case of failure
  int parseTrackletLinkData(int linkSize32, int& hcid, int& trackletWordsRejected, int& trackletWordsReadOK, int& numberOfTrackletsFound);

  // the parsing begins after the DigitHCHeaders have been parsed already
  // maxWords32 is the remaining number of words for the given link
  // digitWordsRejected: count the number of words which were skipped (subset of words read)
  // returns total number of words read (no matter if parsed successfully or not)
  int parseDigitLinkData(int maxWords32, int hcid, int& digitWordsRejected);

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

  // to check for which half-chambers we have seen correct headers and for which we have seen wrong ones
  void printHalfChamberHeaderReport() const;

 private:
  // these variables are configured externally
  int mTrackletHCHeaderState{0};
  int mHalfChamberWords{0};
  int mHalfChamberMajor{0};
  std::bitset<16> mOptions;

  std::array<uint32_t, constants::HBFBUFFERMAX> mHBFPayload; // the full input data payload excluding the RDH header(s)

  // InfoLogger flood protection settings
  int mMaxErrsPrinted = 20;
  int mMaxWarnPrinted = 20;

  // helper pointers, counters for the input buffer
  const char* mDataBufferPtr = nullptr; // pointer to the beginning of the whole payload data
  long mDataBufferSize;                 // the total payload size of the raw data message from the FLP (typically a single HBF from one half-CRU)
  const char* mCurrRdhPtr = nullptr;    // points inside the payload data at the current RDH position
  uint32_t mTotalHBFPayLoad = 0;        // total data payload of the heart beat frame in question (up to wich array index mHBFPayload is filled with data)
  uint32_t mHBFoffset32 = 0;            // points to the current position inside mHBFPayload we are currently reading

  HalfCRUHeader mCurrentHalfCRUHeader; // are we waiting for new header or currently parsing the payload of on
  HalfCRUHeader mPreviousHalfCRUHeader; // are we waiting for new header or currently parsing the payload of on
  bool mPreviousHalfCRUHeaderSet;       // flag, whether we can use mPreviousHalfCRUHeader for additional sanity checks
  DigitHCHeader mDigitHCHeader;         // Digit HalfChamber header we are currently on.
  uint16_t mTimeBins{constants::TIMEBINS}; // the number of time bins to be read out (default 30, can be overwritten from digit HC header)
  bool mTimeBinsFixed{false};              // flag, whether number of time bins different from default was configured
  bool mHaveSeenDigitHCHeader3{false};     // flag, whether we can compare an incoming DigitHCHeader3 with a header we have seen before
  uint32_t mPreviousDigitHCHeadersvnver;  // svn ver in the digithalfchamber header, used for validity checks
  uint32_t mPreviousDigitHCHeadersvnrver; // svn release ver also used for validity checks
  uint8_t mPreTriggerPhase = 0;           // Pre trigger phase of the adcs producing the digits, its comes from an optional DigitHCHeader
                                          // It is stored here to carry it around after parsing it from the DigitHCHeader1 if it exists in the data.
  uint16_t mCRUEndpoint; // the upper or lower half of the currently parsed cru 0-14 or 15-29
  uint16_t mCRUID;       // CRU ID taken from the FEEID of the RDH
  TRDFeeID mFEEID;       // current Fee ID working on

  std::set<int> mHalfChamberHeaderOK;                   // keep track of the half chambers for which we have seen correct headers
  std::set<std::pair<int, int>> mHalfChamberMismatches; // first element is HCID from RDH and second element is HCID from TrackletHCHeader

  o2::InteractionRecord mIR;
  std::array<uint16_t, 15> mCurrentHalfCRULinkLengths;
  std::array<uint8_t, 15> mCurrentHalfCRULinkErrorFlags;

  const LinkToHCIDMapping* mLinkMap = nullptr; // to retrieve HCID from Link ID

  // these counters are reset after every TF
  uint32_t mTrackletsFound{0};         // accumulated number of tracklets found
  uint32_t mDigitsFound{0};            // accumulated number of digits found
  uint32_t mDigitWordsRead = 0;        // number of words read by the digit parser
  uint32_t mDigitWordsRejected = 0;    // number of words rejected by the digit parser
  uint32_t mTrackletWordsRejected = 0; // number of words read by the tracklet parser
  uint32_t mTrackletWordsRead = 0;     // number of words rejected by the tracklet parser
  uint32_t mWordsRejected = 0;         // those words rejected before tracklet and digit parsing could start

  EventRecordContainer mEventRecords; // store data range indexes into the above vectors.
};

} // namespace o2::trd
// namespace o2

#endif
