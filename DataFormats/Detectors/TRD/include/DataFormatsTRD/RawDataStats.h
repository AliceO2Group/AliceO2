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

#include "TObject.h"
#include <string>
#include <cstdint>
#include <array>
#include <chrono>
#include <unordered_map>
#include <gsl/span>
#include "DataFormatsTRD/Constants.h"

namespace o2::trd
{

enum ParsingErrors {
  NoError,
  DigitEndMarkerWrongState,            // read a end marker but we were expecting something else
  DigitMCMHeaderSanityCheckFailure,    // the checked bits in the DigitMCMHeader were not correctly set
  DigitMCMNotIncreasing,               // sequential headers must have increasing mcm number
  DigitMCMDuplicate,                   // we saw two DigitMCMHeaders for the same MCM in one trigger
  DigitADCMaskInvalid,                 // mask adc count does not match # of 1s in bitpattern or the check bits are wrongly set
  DigitSanityCheck,                    // adc failed sanity check based on current channel (odd/even) and check bits DigitMCMData.f
  DigitParsingExitInWrongState,        // exiting parsing in the wrong state ... got to the end of the buffer in wrong state.
  DigitParsingNoSecondEndmarker,       // we found a single digit end marker not followed by a second one
  DigitHCHeaderMismatch,               // the half-chamber ID from the digit HC header is not consistent with the one expected from the link ID
  TrackletHCHeaderFailure,             // either reserved bit not set or HCID is not what was expected from RDH
  TrackletMCMHeaderSanityCheckFailure, // MCMHeader sanity check failure, LSB or MSB not set
  TrackletDataWrongOrdering,           // the tracklet data is not arriving in increasing MCM order
  TrackletDataDuplicateMCM,            // we see more than one TrackletMCMHeader for the same MCM
  TrackletNoTrackletEndMarker,         // got to the end of the buffer with out finding a tracklet end marker.
  TrackletNoSecondEndMarker,           // we expected to see a second tracklet end marker, but found something else instead
  TrackletMCMDataFailure,              // invalid word for TrackletMCMData detected
  TrackletDataMissing,                 // we expected tracklet data but got an endmarker instead
  TrackletExitingNoTrackletEndMarker,  // got to the end of the buffer exiting tracklet parsing with no tracklet end marker
  UnparsedTrackletDataRemaining,       // the tracklet parsing has finished correctly, but there is still data left on the link (CRU puts incorrect link size or corrupt data?)
  UnparsedDigitDataRemaining,          // the digit parsing has finished correctly, but there is still data left on the link (CRU puts incorrect link size or corrupt data? RDH > 8kByte before?)
  DigitHeaderCountGT3,                 // digital half chamber header had more than 3 additional words expected by header. most likely corruption above somewhere.
  DigitHeaderWrongType,                // expected digit header, but could not determine type
  HalfCRUSumLength,                    // if the HalfCRU headers summed lengths wont fit into the buffer, implies corruption, its a faster check than the next one.
  BadRDHMemSize,                       // RDH memory size is supposedly zero
  BadRDHFEEID,                         // RDH parsing failure for reasons in the word
  BadRDHEndPoint,                      // RDH parsing failure for reasons in the word
  BadRDHOrbit,                         // RDH parsing failure for reasons in the word
  BadRDHCRUID,                         // RDH parsing failure for reasons in the word
  BadRDHPacketCounter,                 // RDH packet counter not incrementing
  HalfCRUCorrupt,                      // if the HalfCRU headers has values out of range, corruption is assumed.
  DigitHCHeader1Problem,               // multiple instances of Digit HC Header 1
  DigitHCHeader2Problem,               // multiple instances of Digit HC Header 2
  DigitHCHeader3Problem,               // multiple instances of Digit HC Header 3
  DigitHCHeaderSVNMismatch,            // svn version information has changed in the DigitHCHeader3.
  TrackletsReturnedMinusOne,           // trackletparsing returned -1, data was dumped;
  FEEIDIsFFFF,                         // RDH is in error, the FEEID is 0xffff
  FEEIDBadSector,                      // RDH is in error, the FEEID.supermodule is not a valid value.
  HalfCRUBadBC,                        // the BC in the half-CRU header is so low that the BC shift would make it negative
  TRDLastParsingError                  // This is to keep QC happy until we can change it there as well.
};

static const std::unordered_map<int, std::string> ParsingErrorsString = {
  {NoError, "NoError"},
  {DigitEndMarkerWrongState, "DigitEndMarkerWrongState"},
  {DigitMCMHeaderSanityCheckFailure, "DigitMCMHeaderSanityCheckFailure"},
  {DigitMCMNotIncreasing, "DigitMCMNotIncreasing"},
  {DigitMCMDuplicate, "DigitMCMDuplicate"},
  {DigitADCMaskInvalid, "DigitADCMaskInvalid"},
  {DigitSanityCheck, "DigitSanityCheck"},
  {DigitParsingExitInWrongState, "DigitParsingExitInWrongState"},
  {DigitParsingNoSecondEndmarker, "DigitParsingNoSecondEndmarker"},
  {DigitHCHeaderMismatch, "DigitHCHeaderMismatch"},
  {TrackletHCHeaderFailure, "TrackletHCHeaderFailure"},
  {TrackletMCMHeaderSanityCheckFailure, "TrackletMCMHeaderSanityCheckFailure"},
  {TrackletDataWrongOrdering, "TrackletDataWrongOrdering"},
  {TrackletDataDuplicateMCM, "TrackletDataDuplicateMCM"},
  {TrackletNoTrackletEndMarker, "TrackletNoTrackletEndMarker"},
  {TrackletNoSecondEndMarker, "TrackletNoSecondEndMarker"},
  {TrackletMCMDataFailure, "TrackletMCMDataFailure"},
  {TrackletDataMissing, "TrackletDataMissing"},
  {TrackletExitingNoTrackletEndMarker, "TrackletExitingNoTrackletEndMarker"},
  {UnparsedTrackletDataRemaining, "UnparsedTrackletDataRemaining"},
  {UnparsedDigitDataRemaining, "UnparsedDigitDataRemaining"},
  {DigitHeaderCountGT3, "DigitHeaderCountGT3"},
  {DigitHeaderWrongType, "DigitHeaderWrongType"},
  {HalfCRUSumLength, "HalfCRUSumLength"},
  {BadRDHMemSize, "BadRDHMemSize"},
  {BadRDHFEEID, "BadRDHFEEID"},
  {BadRDHEndPoint, "BadRDHEndPoint"},
  {BadRDHOrbit, "BadRDHOrbit"},
  {BadRDHCRUID, "BadRDHCRUID"},
  {BadRDHPacketCounter, "BadRDHPacketCounter"},
  {HalfCRUCorrupt, "HalfCRUCorrupt"},
  {DigitHCHeader1Problem, "DigitHCHeader1Problem"},
  {DigitHCHeader2Problem, "DigitHCHeader2Problem"},
  {DigitHCHeader3Problem, "DigitHCHeader3Problem"},
  {DigitHCHeaderSVNMismatch, "DigitHCHeaderSVNMismatch"},
  {TrackletsReturnedMinusOne, "TrackletsReturnedMinusOne"},
  {FEEIDIsFFFF, "FEEIDIsFFFF"},
  {FEEIDBadSector, "FEEIDBadSector"},
  {HalfCRUBadBC, "HalfCRUBadBC"},
  {TRDLastParsingError, "TRDLastParsingError"}};

//enumerations for the options, saves on having a long parameter list.
enum OptionBits {
  TRDVerboseBit,
  TRDVerboseErrorsBit,
  TRDIgnore2StageTrigger,
  TRDGenerateStats,
  TRDOnlyCalibrationTriggerBit
}; // this is currently 16 options, the array is 16, if you add here you need to change the 16;

//Data to be stored and accumulated on an event basis.
//events are spread out with in the data coming in with a halfcruheader per event, per ... half cru.
//this is looked up via the interaction record (orbit and bunchcrossing).
//this permits averaging in the data that gets senton per timeframe
struct TRDDataCountersPerEvent {
  TRDDataCountersPerEvent() = default;
  double mTimeTaken = 0.;             // time take to process an event (summed trackletparsing and digitparsing) parts not accounted for.
  double mTimeTakenForDigits = 0.;    // time take to process tracklet data blocks [us].
  double mTimeTakenForTracklets = 0.; // time take to process digit data blocks [us].
  uint64_t mWordsRead = 0;            // words read in
  uint64_t mWordsRejected = 0;        // words skipped for various reasons.
  uint16_t mTrackletsFound = 0;       // tracklets found in the event
  uint16_t mDigitsFound = 0;          // digits found in the event
};

//Data to be stored on a timeframe basis to then be sent as a message to be ultimately picked up by qc.
//Some countes include a average over the numbers stored on a per event basis, e.g. digits per event.
class TRDDataCountersPerTimeFrame
{
 public:
  std::array<uint8_t, o2::trd::constants::NSECTOR * 60> mLinkErrorFlag{};                              // status of the error flags for this timeframe, 8bit values from cru halfchamber header.
  std::array<uint16_t, o2::trd::constants::NSECTOR * 60> mLinkNoData;                                  // Link had no data or was not present.
  std::array<uint16_t, o2::trd::constants::NSECTOR * 60> mLinkWords{};                                 // units of 256bits, read from the cru half chamber header
  std::array<uint16_t, o2::trd::constants::NSECTOR * 60> mLinkWordsRead{};                             // units of 32 bits the data words read before dumping or finishing
  std::array<uint16_t, o2::trd::constants::NSECTOR * 60> mLinkWordsRejected{};                         // units of 32 bits the data dumped due to some or other error
  std::array<uint16_t, TRDLastParsingError> mParsingErrors{};                                          // errors in parsing, indexed by enum above of ParsingErrors
  std::array<uint32_t, o2::trd::constants::NSECTOR * 60 * TRDLastParsingError> mParsingErrorsByLink{}; // errors in parsing, indexed by enum above of ParsingErrors
  uint16_t mDigitsPerEvent;                                                                                                  // average digits found per event in this timeframe, ignoring the no digit events where there is no calibration trigger.
  uint16_t mTrackletsPerEvent;                                                                                               // average tracklets found per event in this timeframe
  double mTimeTaken;                                                                                   // time taken to process the entire timeframe [ms].
  double mTimeTakenForDigits;                                                                          // time take to process tracklet data blocks [ms].
  double mTimeTakenForTracklets;                                                                       // time take to process digit data blocks [ms].
  uint32_t mDigitsFound;                                                                               // digits found in the time frame.
  uint32_t mTrackletsFound;                                                                            // tracklets found in the time frame.
  std::array<uint64_t, 256> mDataFormatRead{};                                                         // We just keep the major version number
  void clear()
  {
    mLinkNoData.fill(0);
    mLinkWords.fill(0);
    mLinkWordsRead.fill(0);
    mLinkWordsRejected.fill(0);
    mParsingErrors.fill(0);
    mParsingErrorsByLink.fill(0);
    mDigitsPerEvent = 0;
    mTrackletsPerEvent = 0;
    mTimeTaken = 0;
    mTimeTakenForDigits = 0;
    mTimeTakenForTracklets = 0;
    mDigitsFound = 0;
    mTrackletsFound = 0; //tracklets found in timeframe.
    mDataFormatRead.fill(0);
  };
  ClassDefNV(TRDDataCountersPerTimeFrame, 1); // primarily for serialisation so we can send this as a message in o2
};

} // namespace o2::trd

#endif
