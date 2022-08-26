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
  UnrecognisedVersion,
  BadDigit,
  BadTracklet,
  DigitEndMarkerWrongState,                // read a end marker but we were expecting something else due to
  DigitMCMHeaderSanityCheckFailure,        // essentially we did not see an MCM header see RawData.h for requirement
  DigitROBDecreasing,                      // sequential headers must have the same or increasing rob number
  DigitMCMNotIncreasing,                   // sequential headers must have increasing mcm number
  DigitADCMaskMismatch,                    // mask adc count does not match # of 1s in bitpattern
  DigitADCMaskAdvanceToEnd,                // in advancing to adcmask we have reached the end of the buffer
  DigitMCMHeaderBypassButStateMCMHeader,   // we are reading mcmadc data but the state is mcmheader
  DigitEndMarkerStateButReadingMCMADCData, // read the endmarker while expecting to read the mcmadcdata
  DigitADCChannel21,                       // ADCMask is zero but we are still on a digit.
  DigitADCChannelGT22,                     // error allocating digit, so digit channel has error value
  DigitGT10ADCs,                           // more than 10 adc data words seen
  DigitSanityCheck,                        // adc failed sanity check see RawData.cxx for faiulre reasons
  DigitExcessTimeBins,                     // ADC has more than 30 timebins (10 adc words)
  DigitParsingExitInWrongState,            // exiting parsing in the wrong state ... got to the end of the buffer in wrong state.
  DigitStackMismatch,                      // mismatch between rdh and hcheader stack calculation/value
  DigitLayerMismatch,                      // mismatch between rdh and hcheader stack calculation/value
  DigitHCHeaderMismatch,                   // the half-chamber ID from the digit HC header is not consistent with the one expected from the link ID
  TrackletCRUPaddingWhileParsingTracklets, // reading a padding word while expecting tracklet data
  TrackletTrackletHCHeaderButWrongState,   // read a trackletHCHedaer but not in correct state.
  TrackletHCHeaderSanityCheckFailure,      // HCHeader sanity check failure, see RawData.cxx for reasons.
  TrackletMCMHeaderSanityCheckFailure,     // MCMHeader sanity check failure, see RawData.cxx for reasons.
  TrackletMCMHeaderButParsingMCMData,      // state is still MCMHeader but we are parsing MCMData
  TrackletStateMCMHeaderButParsingMCMData,
  TrackletTrackletCountGTThatDeclaredInMCMHeader, // mcmheader tracklet count does not match that in we have parsed.
  TrackletInvalidTrackletCount,                   // invalid tracklet count in header vs data
  TrackletPadRowIncreaseError,                    // subsequent padrow can not be less than previous one.
  TrackletColIncreaseError,                       // subsequent col can not be less than previous one
  TrackletNoTrackletEndMarker,                    // got to the end of the buffer with out finding a tracklet end marker.
  TrackletExitingNoTrackletEndMarker,             // got to the end of the buffer exiting tracklet parsing with no tracklet end marker
  DigitHeaderCountGT3,                            // digital half chamber header had more than 3 additional words expected by header. most likely corruption above somewhere.
  DigitHeaderWrong1,                              // expected header word1 but wrong ending marker
  DigitHeaderWrong2,                              // expected header word2 but wrong ending marker
  DigitHeaderWrong3,                              // expected header word3 but wrong ending marker
  DigitHeaderWrong4,                              // expected header word but have no idea what we are looking at default of switch statement
  DigitDataStillOnLink,                           // got to the end of digit parsing and there is still data on link, normally not advancing far enough when dumping data.
  TrackletIgnoringDataTillEndMarker,              // for some reason we are bouncing to the end word by word, this counts those words
  GarbageDataAtEndOfHalfCRU,                      // if the first word of the halfcru is wrong i.e. side, eventype, the half cru header is so wrong its not corrupt, its other garbage
  HalfCRUSumLength,                               // if the HalfCRU headers summed lengths wont fit into the buffer, implies corruption, its a faster check than the next one.
  BadRDHMemSize,                                  // RDH memory size is supposedly zero
  BadRDHFEEID,                                    // RDH parsing failure for reasons in the word
  BadRDHEndPoint,                                 // RDH parsing failure for reasons in the word
  BadRDHOrbit,                                    // RDH parsing failure for reasons in the word
  BadRDHCRUID,                                    // RDH parsing failure for reasons in the word
  BadRDHPacketCounter,                            // RDH parsing failure for reasons in the word
  HalfCRUCorrupt,                                 // if the HalfCRU headers has values out of range, corruption is assumed.
  DigitHCHeader1Problem,                          // multiple instances of Digit HC Header 1
  DigitHCHeader2Problem,                          // multiple instances of Digit HC Header 2
  DigitHCHeader3Problem,                          // multiple instances of Digit HC Header 3
  DigitHCHeaderSVNMismatch,                       // svn version information has changed in the DigitHCHeader3.
  TrackletsReturnedMinusOne,                      // trackletparsing returned -1, data was dumped;
  FEEIDIsFFFF,                                    // RDH is in error, the FEEID is 0xffff
  FEEIDBadSector,                                 // RDH is in error, the FEEID.supermodule is not a valid value.
  DigitHCHeaderPreTriggerPhaseOOB,                // pretrigger phase in Digit HC header has to be less than 12, it is not.
  HalfCRUBadBC,                                   // saw a bc below the L0 trigger
  TRDLastParsingError                             // This is to keep QC happy until we can change it there as well.
};

static std::unordered_map<int, std::string> ParsingErrorsString = {
  {NoError, "NoError"},
  {UnrecognisedVersion, "Unrecognised Data Version"},
  {BadDigit, "Bad Digt"},
  {BadTracklet, "Bad Tracklet"},
  {DigitEndMarkerWrongState, "Digit EndMarker but Wrong State"},
  {DigitMCMHeaderSanityCheckFailure, "Digit MCM Header Sanity Check Failure"},
  {DigitROBDecreasing, "Digit ROB not increasing"},
  {DigitMCMNotIncreasing, "Digit MCM number Not Increasing"},
  {DigitADCMaskMismatch, "Digit ADCMask Mismatch"},
  {DigitADCMaskAdvanceToEnd, "Digit ADC Mask problem, advancing to end"},
  {DigitMCMHeaderBypassButStateMCMHeader, "Digit MCM Header bypassed but state is mcm header"},
  {DigitEndMarkerStateButReadingMCMADCData, "Digit End Marker but state is MCMADCData"},
  {DigitADCChannel21, "Digit ADC has Channel 21"},
  {DigitADCChannelGT22, "Digit ADC Channel > 22"},
  {DigitGT10ADCs, "Digit has more than 10 ADCs"},
  {DigitSanityCheck, "Digit Sanity Check Failure"},
  {DigitExcessTimeBins, "Digit has Excess TimeBins"},
  {DigitParsingExitInWrongState, "Digit Parsing Exiting in wrong starte"},
  {DigitStackMismatch, "Digit Stack MisMatch"},
  {DigitLayerMismatch, "Digit Layer MisMatch"},
  {TrackletCRUPaddingWhileParsingTracklets, "Tracklet CRU Padding while parsing trackletsl"},
  {TrackletHCHeaderSanityCheckFailure, "Tracklet HC Header Sanity Check Failure"},
  {TrackletMCMHeaderSanityCheckFailure, "Tracklet MCMHeader Sanity Check Failure"},
  {TrackletMCMHeaderButParsingMCMData, "Tracklet on MCMHeader, but parsing MCMData"},
  {TrackletStateMCMHeaderButParsingMCMData, "Tracklet state MCMHeader but parsing MCMData"},
  {TrackletTrackletCountGTThatDeclaredInMCMHeader, "Tracklet count > than that in the MCM header"},
  {TrackletInvalidTrackletCount, "Tracklet invalid tracklet count"},
  {TrackletPadRowIncreaseError, "Tracklet padrow is not increasing"},
  {TrackletColIncreaseError, "Tracklet column is not increasing"},
  {TrackletNoTrackletEndMarker, "Tracklet  did not find an end marker"},
  {TrackletExitingNoTrackletEndMarker, "Tracklet exiting with out a tracklet end marker"},
  {DigitHeaderCountGT3, "DigitHeaderCountGT3"},
  {DigitHeaderWrong1, "Digit Header word 1 Wrong"},
  {DigitHeaderWrong2, "Digit Header word 2 Wrong"},
  {DigitHeaderWrong3, "Digit Header word 3 Wrong"},
  {DigitHeaderWrong4, "Digit Header word unknown"},
  {DigitDataStillOnLink, "Digit parsing still has data on its link"},
  {TrackletIgnoringDataTillEndMarker, "Tracklet parsing ignoring DataTillEndMarker"},
  {GarbageDataAtEndOfHalfCRU, "GarbageDataAtEndOfHalfCRU"},
  {HalfCRUSumLength, "HalfCRU Sum of Lengths is not valid"},
  {BadRDHFEEID, "Bad RDH FEEID"},
  {BadRDHEndPoint, "Bad RDH EndPoint"},
  {BadRDHOrbit, "Bad RDH Orbit"},
  {BadRDHCRUID, "Bad RDH CRUID"},
  {BadRDHPacketCounter, "Bad RDH Packet Counter not incrementing by 1"},
  {HalfCRUCorrupt, "HalfCRU appears to be Corrupt"},
  {DigitHCHeader1Problem, "Digit HCHeader 1 problem "},
  {DigitHCHeader2Problem, "Digit HCHeader 2"},
  {DigitHCHeader3Problem, "Digit HCHeader 3"},
  {DigitHCHeaderSVNMismatch, "DigitHCHeaderSVNMismatch"},
  {TrackletsReturnedMinusOne, "Tracklets Returned -1"},
  {FEEIDIsFFFF, "FEEID Is FFFx"},
  {FEEIDBadSector, "FEEID Sector is not valid"},
  {DigitHCHeaderPreTriggerPhaseOOB, "Digit Half Chamber Header PreTriggerPhase is out of bounds"},
  {HalfCRUBadBC, "HalfCRU has a bad bunchcrossing"},
  {TRDLastParsingError, "Last Parsing Error"}};

//enumerations for the options, saves on having a long parameter list.
enum OptionBits {
  TRDByteSwapBit,
  TRDVerboseBit,
  TRDVerboseHalfCruBit,
  TRDVerboseLinkBit,
  TRDVerboseWordBit,
  TRDVerboseErrorsBit,
  TRDFixDigitCorruptionBit,
  TRDIgnoreDigitHCHeaderBit,
  TRDIgnoreTrackletHCHeaderBit,
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
  std::array<uint32_t, o2::trd::constants::NSECTOR * 60 * TRDLastParsingError + TRDLastParsingError> mParsingErrorsByLink{}; // errors in parsing, indexed by enum above of ParsingErrors
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

class TRDDataCountersRunning
{                                                //those counters that keep counting
  std::array<uint32_t, 1080> mLinkFreq{};        //units of 256bits "cru word"
  std::array<bool, 1080> mLinkEmpty{};           // Link only has padding words only, probably not serious.
  std::array<uint64_t, 65535> mDataFormatRead{}; // 7bits.7bits major.minor version read from HCHeader.
};

} // namespace o2::trd

#endif
