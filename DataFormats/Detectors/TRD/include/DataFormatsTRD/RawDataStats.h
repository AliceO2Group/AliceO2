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

#ifndef O2_TRD_RAWDATASTATS
#define O2_TRD_RAWDATASTATS

#include <iostream>
#include <string>
#include <cstdint>
#include <array>
#include <vector>
#include <bitset>
#include <gsl/span>
#include "DataFormatsTRD/Constants.h"

namespace o2::trd
{
enum ParsingErrors { TRDParsingNoError,
                     TRDParsingUnrecognisedVersion,
                     TRDParsingBadDigt,
                     TRDParsingBadTracklet,
                     TRDParsingDigitEndMarkerWrongState,                // read a end marker but we were expecting something else due to
                     TRDParsingDigitMCMHeaderSanityCheckFailure,        //essentially we did not see an MCM header see RawData.h for requirement
                     TRDParsingDigitROBDecreasing,                      // sequential headers must have the same or increasing rob number
                     TRDParsingDigitMCMNotIncreasing,                   // sequential headers must have increasing mcm number
                     TRDParsingDigitADCMaskMismatch,                    // mask adc count does not match # of 1s in bitpattern
                     TRDParsingDigitADCMaskAdvanceToEnd,                // in advancing to adcmask we have reached the end of the buffer
                     TRDParsingDigitMCMHeaderBypassButStateMCMHeader,   // we are reading mcmadc data but the state is mcmheader
                     TRDParsingDigitEndMarkerStateButReadingMCMADCData, // read the endmarker while expecting to read the mcmadcdata
                     TRDParsingDigitADCChannel21,                       // ADCMask is zero but we are still on a digit.
                     TRDParsingDigitADCChannelGT22,                     // error allocating digit, so digit channel has error value
                     TRDParsingDigitGT10ADCs,                           // more than 10 adc data words seen
                     TRDParsingDigitSanityCheck,                        // adc failed sanity check see RawData.cxx for faiulre reasons
                     TRDParsingDigitExcessTimeBins,                     // ADC has more than 30 timebins (10 adc words)
                     TRDParsingDigitParsingExitInWrongState,            // exiting parsing in the wrong state ... got to the end of the buffer in wrong state.
                     TRDParsingDigitStackMismatch,                      // mismatch between rdh and hcheader stack calculation/value
                     TRDParsingDigitLayerMismatch,                      // mismatch between rdh and hcheader stack calculation/value
                     TRDParsingDigitSectorMismatch,                     // mismatch between rdh and hcheader stack calculation/value
                     TRDParsingTrackletCRUPaddingWhileParsingTracklets, // reading a padding word while expecting tracklet data
                     TRDParsingTrackletBit11NotSetInTrackletHCHeader,   // bit 11 not set in hc header for tracklets.
                     TRDParsingTrackletHCHeaderSanityCheckFailure,      // HCHeader sanity check failure, see RawData.cxx for reasons.
                     TRDParsingTrackletMCMHeaderSanityCheckFailure,     // MCMHeader sanity check failure, see RawData.cxx for reasons.
                     TRDParsingTrackletMCMHeaderButParsingMCMData,      // state is still MCMHeader but we are parsing MCMData
                     TRDParsingTrackletStateMCMHeaderButParsingMCMData,
                     TRDParsingTrackletTrackletCountGTThatDeclaredInMCMHeader, //mcmheader tracklet count does not match that in we have parsed.
                     TRDParsingTrackletInvalidTrackletCount,                   // invalid tracklet count in header vs data
                     TRDParsingTrackletPadRowIncreaseError,                    // subsequent padrow can not be less than previous one.
                     TRDParsingTrackletColIncreaseError,                       // subsequent col can not be less than previous one
                     TRDParsingTrackletNoTrackletEndMarker,                    // got to the end of the buffer with out finding a tracklet end marker.
                     TRDParsingTrackletExitingNoTrackletEndMarker,             // got to the end of the buffer exiting tracklet parsing with no tracklet end marker
                     TRDParsingDigitHeaderCountGT3,                            // digital half chamber header had more than 3 additional words expected by header. most likely corruption above somewhere.
                     TRDParsingDigitHeaderWrong1,                              // expected header word1 but wrong ending marker
                     TRDParsingDigitHeaderWrong2,                              // expected header word2 but wrong ending marker
                     TRDParsingDigitHeaderWrong3,                              // expected header word3 but wrong ending marker
                     TRDParsingDigitHeaderWrong4,                              // expected header word but have no idea what we are looking at default of switch statement
                     TRDParsingDigitDataStillOnLink,                           // got to the end of digit parsing and there is still data on link, normally not advancing far enough when dumping data.
                     TRDParsingTrackletIgnoringDataTillEndMarker               // for some reason we are bouncing to the end word by word, this counts those words
};

extern std::vector<std::string> ParsingErrorsString;

//enumerations for the options, saves on having a long parameter list.
enum OptionBits {
  TRDByteSwapBit,
  TRDVerboseBit,
  TRDHeaderVerboseBit,
  TRDDataVerboseBit,
  TRDCompressedDataBit,
  TRDFixDigitCorruptionBit,
  TRDEnableTimeInfoBit,
  TRDEnableStatsBit,
  TRDIgnoreDigitHCHeaderBit,
  TRDIgnoreTrackletHCHeaderBit,
  TRDEnableRootOutputBit
};

class TRDDataCountersPerEvent
{ //thisis on a per event basis
 public:
  //TODO this should go into a dpl message for catching by qc ?? I think.
  uint64_t mTimeTaken;                        // time take to process an event (summed trackletparsing and digitparsing) parts not accounted for.
  uint64_t mTimeTakenForDigits;               // time take to process tracklet data blocks [us].
  uint64_t mTimeTakenForTracklets;            // time take to process digit data blocks [us].
  uint64_t mDigitWordsRead;                   // digit words read in
  uint64_t mDigitWordsSkipped;                // digit words skipped for various reasons.
  uint64_t mTrackletWordsRead;                // tracklet words read in
  uint64_t mTrackletWordsSkipped;             // tracklet words skipped for various reasons.
  std::array<uint8_t, 1080> mLinkErrorFlag{}; //status of the error flags for this event, 8bit values from cru halfchamber header.
};

class TRDDataCountersPerTimeFrame
{ //thisis on a per event basis
 public:
  std::array<uint32_t, 1080> mLinkNoData;                                   // Link had no data or was not present.
  std::array<uint32_t, 1080> mLinkWords{};                                  //units of 256bits, read from the cru half chamber header
  std::array<uint32_t, 1080> mLinkWordsRead{};                              // units of 32 bits the data words read before dumping or finishing
  std::array<uint32_t, 1080> mLinkWordsDumped{};                            // units of 32 bits the data dumped due to some or other error
  std::array<int64_t, o2::trd::constants::MAXMCMCOUNT> mLinkMCMsWithData{}; // and its corresponding volume of data.
  std::array<uint32_t, constants::MAXMCMCOUNT> mMCMDigitCount{};
  std::array<uint32_t, constants::MAXMCMCOUNT> mMCMTrackletCount{};
  std::array<uint32_t, 30> mParsingErrors{};              // errors in parsing, indexed by enum above of ParsingErrors
  std::array<uint32_t, 1080 * 30> mParsingErrorsByLink{}; // errors in parsing, indexed by enum above of ParsingErrors
  uint64_t mTimeTaken;                                    // time taken to process the entire timeframe [ms].
  uint64_t mTimeTakenForDigits;                           // time take to process tracklet data blocks [us].
  uint64_t mTimeTakenForTracklets;                        // time take to process digit data blocks [us].
  uint64_t mDigitsFound;                                  // digit found in the time frame.
  uint64_t mTrackletsFound;                               // tracklets found in the time frame.
  uint64_t mDigitWordsRead;                               // digit words read in.
  uint64_t mDigitWordsSkipped;                            // digit words skipped for various reasons.
  uint64_t mTrackletWordsRead;                            // tracklet words read in.
  uint64_t mTrackletWordsSkipped;                         // tracklet words skipped for various reasons.
  uint64_t mDataWordsRead;
  uint64_t mDataWordsRejected;
  //TRDDataCountersPerTimeFrame* operator=(TRDDataCountersPerTimeFrame *old){this=old;return *this;}
};

//TODO not sure this class is needed
class TRDDataCountersRunning
{                                                //those counters that keep counting
  std::array<uint32_t, 1080> mLinkFreq{};        //units of 256bits "cru word"
  std::array<bool, 1080> mLinkEmpty{};           // Link only has padding words only, probably not serious.
  std::array<uint64_t, 65535> mDataFormatRead{}; // 7bits.7bits major.minor version read from HCHeader.
};

} // namespace o2::trd

#endif
