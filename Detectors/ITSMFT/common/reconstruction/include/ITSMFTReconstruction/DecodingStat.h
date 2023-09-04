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

/// \file DecodingStat.h
/// \brief Alpide Chip and GBT link decoding statistics

#ifndef _ALICEO2_DECODINGSTAT_H_
#define _ALICEO2_DECODINGSTAT_H_

#include <string>
#include <array>
#include <Rtypes.h>
#include "ITSMFTReconstruction/GBTWord.h"

namespace o2
{
namespace itsmft
{
class ChipPixelData;

struct ChipStat {
  enum ActionOnError : int {
    ErrActNone = 0x0,      // do nothing
    ErrActPropagate = 0x1, // propagate to decoded data
    ErrActDump = 0x2       // produce raw data dump
  };

  enum DecErrors : int {
    BusyViolation,                // Busy violation
    DataOverrun,                  // Data overrun
    Fatal,                        // Fatal (?)
    BusyOn,                       // Busy On
    BusyOff,                      // Busy Off
    TruncatedChipEmpty,           // Data was truncated after ChipEmpty
    TruncatedChipHeader,          // Data was truncated after ChipHeader
    TruncatedRegion,              // Data was truncated after Region record
    TruncatedLondData,            // Data was truncated in the LongData record
    WrongDataLongPattern,         // LongData pattern has highest bit set
    NoDataFound,                  // Region is not followed by Short or Long data
    UnknownWord,                  // Unknown word was seen
    RepeatingPixel,               // Same pixel fired more than once
    WrongRow,                     // Non-existing row decoded
    APE_STRIP,                    // lane data stripped for this chip event (behaviour changed with RU FW v1.16.0, for general APE behaviour see  https://alice.its.cern.ch/jira/browse/O2-1717)
    APE_RESERVED_F3,              // reserved F3
    APE_DET_TIMEOUT,              // detector timeout (FATAL)
    APE_OOT_START,                // 8b10b OOT (FATAL, start)
    APE_PROTOCOL_ERROR,           // event protocol error marker (FATAL, start)
    APE_LANE_FIFO_OVERFLOW_ERROR, // lane FIFO overflow error (FATAL)
    APE_FSM_ERROR,                // FSM error (FATAL, SEU error, reached an unknown state)
    APE_OCCUPANCY_RATE_LIMIT,     // pending detector events limit (FATAL)
    APE_OCCUPANCY_RATE_LIMIT_2,   // pending detector events limit in packager(FATAL)
    APE_LANE_PROTOCOL_ERROR,      // lane protocol error
    APE_RESERVED_FC,              // reserved FC
    APE_ERROR_NON_CRITICAL_BYTE,  // Error in non critical byte
    APE_OOT_NON_CRITICAL,         // OOT non-critical
    WrongDColOrder,               // DColumns non increasing
    InterleavedChipData,          // Chip data interleaved on the cable
    TruncatedBuffer,              // truncated buffer, 0 padding
    TrailerAfterHeader,           // trailer seen after header w/o FE of FD set
    NErrorsDefined
  };

  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "BusyViolation flag ON",                        // BusyViolation
    "DataOverrun flag ON",                          // DataOverrun
    "Fatal flag ON",                                // Fatal
    "BusyON",                                       // BusyOn
    "BusyOFF",                                      // BusyOff
    "Data truncated after ChipEmpty",               // TruncatedChipEmpty
    "Data truncated after ChipHeader",              // TruncatedChipHeader
    "Data truncated after Region",                  // TruncatedRegion
    "Data truncated after LongData",                // TruncatedLondData
    "LongData pattern has highest bit set",         // WrongDataLongPattern
    "Region is not followed by Short or Long data", // NoDataFound
    "Unknown word",                                 // UnknownWord
    "Same pixel fired multiple times",              // RepeatingPixel
    "Non-existing row decoded",                     // WrongRow
    "APE_STRIP",                                    // lane data stripped for this chip event (behaviour changed with RU FW v1.16.0, for general APE behaviour see  https://alice.its.cern.ch/jira/browse/O2-1717)
    "APE_RESERVED_F3",                              // reserved F3
    "APE_DET_TIMEOUT",                              // detector timeout (FATAL)
    "APE_OOT_START",                                // 8b10b OOT (FATAL, start)
    "APE_PROTOCOL_ERROR",                           // event event protocol error marker (FATAL, start)
    "APE_LANE_FIFO_OVERFLOW_ERROR",                 // lane FIFO overflow error (FATAL)
    "APE_FSM_ERROR",                                // FSM error (FATAL, SEU error, reached an unknown state)
    "APE_OCCUPANCY_RATE_LIMIT",                     // pending detector events limit (FATAL)
    "APE_OCCUPANCY_RATE_LIMIT_2",                   // pending detector events limit in packager(FATAL)
    "APE_LANE_PROTOCOL_ERROR",                      // lane protocol error
    "APE_RESERVED_FC",                              // reserved
    "APE_ERROR_IN_NON_CRITICAL_BYTE",               // Error in non critical byte
    "APE_OOT_NON_CRITICAL",                         // OOT non-critical
    "DColumns non-increasing",                      // DColumns non increasing
    "Chip data interleaved on the cable",           // Chip data interleaved on the cable
    "TruncatedBuffer",                              // truncated buffer, 0 padding
    "TrailerAfterHeader"                            // trailer seen after header w/o FE of FD set
  };

  static constexpr std::array<uint32_t, NErrorsDefined> ErrActions = {
    ErrActPropagate | ErrActDump, // Busy violation
    ErrActPropagate | ErrActDump, // Data overrun
    ErrActPropagate | ErrActDump, // Fatal (?)
    ErrActNone,                   // Busy On
    ErrActNone,                   // Busy Off
    ErrActPropagate | ErrActDump, // Data was truncated after ChipEmpty
    ErrActPropagate | ErrActDump, // Data was truncated after ChipHeader
    ErrActPropagate | ErrActDump, // Data was truncated after Region record
    ErrActPropagate | ErrActDump, // Data was truncated in the LongData record
    ErrActPropagate | ErrActDump, // LongData pattern has highest bit set
    ErrActPropagate | ErrActDump, // Region is not followed by Short or Long data
    ErrActPropagate | ErrActDump, // Unknown word was seen
    ErrActPropagate,              // Same pixel fired more than once
    ErrActPropagate | ErrActDump, // Non-existing row decoded
    ErrActPropagate | ErrActDump, // lane data stripped for this chip event (behaviour changed with RU FW v1.16.0, for general APE behaviour see  https://alice.its.cern.ch/jira/browse/O2-1717)
    ErrActPropagate | ErrActDump, // reserved F3
    ErrActPropagate | ErrActDump, // detector timeout (FATAL)
    ErrActPropagate | ErrActDump, // 8b10b OOT (FATAL, start)
    ErrActPropagate | ErrActDump, // event protocol error marker (FATAL, start)
    ErrActPropagate | ErrActDump, // lane FIFO overflow error (FATAL)
    ErrActPropagate | ErrActDump, // FSM error (FATAL, SEU error, reached an unknown state)
    ErrActPropagate | ErrActDump, // pending detector events limit (FATAL)
    ErrActPropagate | ErrActDump, // pending detector events limit in packager(FATAL)
    ErrActPropagate | ErrActDump, // lane protocol error
    ErrActPropagate | ErrActDump, // reserved FC
    ErrActPropagate | ErrActDump, // Error in non critical byte
    ErrActPropagate | ErrActDump, // OOT non-critical
    ErrActPropagate | ErrActDump, // DColumns non increasing
    ErrActPropagate | ErrActDump, // Chip data interleaved on the cable
    ErrActPropagate | ErrActDump, // Truncated buffer while something was expected
    ErrActPropagate | ErrActDump  // trailer seen after header w/o FE of FD set
  };
  uint16_t feeID = -1;
  size_t nHits = 0;
  std::array<uint32_t, NErrorsDefined> errorCounts = {};
  ChipStat() = default;
  ChipStat(uint16_t _feeID) : feeID(_feeID) {}

  void clear()
  {
    memset(errorCounts.data(), 0, sizeof(uint32_t) * errorCounts.size());
    nHits = 0;
  }

  static int getAPENonCritical(uint8_t c)
  {
    if (c == 0xfd || c == 0xfe) {
      return APE_STRIP + c - 0xf2;
    }
    return -1;
  }

  // return APE DecErrors code or -1 if not APE error, set fatal flag if needd
  static int getAPECode(uint8_t c, bool& ft)
  {
    if (c < 0xf2 || c > 0xfe) {
      ft = false;
      return -1;
    }
    ft = c >= 0xf2 && c <= 0xfe;
    return APE_STRIP + c - 0xf2;
  }
  uint32_t getNErrors() const;
  uint32_t addErrors(uint32_t mask, uint16_t chID, int verbosity);
  uint32_t addErrors(const ChipPixelData& d, int verbosity);
  void print(bool skipNoErr = true, const std::string& pref = "FEEID") const;

  ClassDefNV(ChipStat, 1);
};

struct ChipError {
  uint32_t id = -1;
  uint32_t nerrors = 0;
  uint32_t errors = 0;

  int16_t getChipID() const { return int16_t(id & 0xffff); }
  uint16_t getFEEID() const { return uint16_t(id >> 16); }
  static uint32_t composeID(uint16_t feeID, int16_t chipID) { return uint32_t(feeID) << 16 | uint16_t(chipID); }
  ClassDefNV(ChipError, 1);
};

/// Statistics for per-link decoding
struct GBTLinkDecodingStat {
  /// counters for format checks

  enum DecErrors : int {
    ErrNoRDHAtStart,             // page does not start with RDH
    ErrPageNotStopped,           // RDH is stopped, but the time is not matching the ~stop packet
    ErrStopPageNotEmpty,         // Page with RDH.stop is not empty
    ErrPageCounterDiscontinuity, // RDH page counters for the same RU/trigger are not continuous
    ErrRDHvsGBTHPageCnt,         // RDH and GBT header page counters are not consistent
    ErrMissingGBTTrigger,        // GBT trigger word was expected but not found
    ErrMissingGBTHeader,         // GBT payload header was expected but not found
    ErrMissingGBTTrailer,        // GBT payload trailer was expected but not found
    ErrNonZeroPageAfterStop,     // all lanes were stopped but the page counter in not 0
    ErrUnstoppedLanes,           // end of FEE data reached while not all lanes received stop
    ErrDataForStoppedLane,       // data was received for stopped lane
    ErrNoDataForActiveLane,      // no data was seen for lane (which was not in timeout)
    ErrIBChipLaneMismatch,       // chipID (on module) was different from the lane ID on the IB stave
    ErrCableDataHeadWrong,       // cable data does not start with chip header or empty chip
    ErrInvalidActiveLanes,       // active lanes pattern conflicts with expected for given RU type
    ErrPacketCounterJump,        // jump in RDH.packetCounter
    ErrPacketDoneMissing,        // packet done is missing in the trailer while CRU page is not over
    ErrMissingDiagnosticWord,    // missing diagnostic word after RDH with stop
    ErrGBTWordNotRecognized,     // GBT word not recognized
    ErrWrongeCableID,            // Invalid cable ID
    ErrWrongAlignmentWord,       // unexpected alignment word
    ErrMissingROF,               // missing ROF (desync?)
    ErrOldROF,                   // old ROF (desync?)
    ErrLinkRecovery,             // data skipped since recovery is declared
    NErrorsDefined
  };
  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "Page data does not start with expected RDH",                        // ErrNoRDHAtStart
    "RDH is stopped, but the time is not matching the stop packet",      // ErrPageNotStopped
    "Page with RDH.stop does not contain diagnostic word only",          // ErrStopPageNotEmpty
    "RDH page counters for the same RU/trigger are not continuous",      // ErrPageCounterDiscontinuity
    "RDH and GBT header page counters are not consistent",               // ErrRDHvsGBTHPageCnt
    "GBT trigger word was expected but not found",                       // ErrMissingGBTTrigger
    "GBT payload header was expected but not found",                     // ErrMissingGBTHeader
    "GBT payload trailer was expected but not found",                    // ErrMissingGBTTrailer
    "All lanes were stopped but the page counter in not 0",              // ErrNonZeroPageAfterStop
    "End of FEE data reached while not all lanes received stop",         // ErrUnstoppedLanes
    "Data was received for stopped lane",                                // ErrDataForStoppedLane
    "No data was seen for lane (which was not in timeout)",              // ErrNoDataForActiveLane
    "ChipID (on module) was different from the lane ID on the IB stave", // ErrIBChipLaneMismatch
    "Cable data does not start with chip header or empty chip",          // ErrCableDataHeadWrong
    "Active lanes pattern conflicts with expected for given RU type",    // ErrInvalidActiveLanes
    "Jump in RDH_packetCounter",                                         // ErrPacketCounterJump
    "Packet done is missing in the trailer while CRU page is not over",  // ErrPacketDoneMissing
    "Wrong/missing diagnostic GBT word after RDH with stop",             // ErrMissingDiagnosticWord
    "GBT word not recognized",                                           // ErrGBTWordNotRecognized
    "Wrong cable ID",                                                    // ErrWrongeCableID
    "Unexpected CRU page alignment padding word",                        // ErrWrongAlignmentWord
    "ROF in future, pause decoding to synchronize",                      // ErrMissingROF
    "Old ROF, discarding",                                               // ErrOldROF
    "Data discarded due to the recovery flag in RDH",                    // ErrLinkRecovery
  };

  uint16_t feeID = 0; // FeeID
  // Note: packet here is meant as a group of CRU pages belonging to the same trigger
  uint32_t nPackets = 0;                                                        // total number of packets (RDH pages)
  uint32_t nTriggers = 0;                                                       // total number of triggers (ROFs)
  std::array<uint32_t, NErrorsDefined> errorCounts = {};                        // error counters
  std::array<uint32_t, GBTDataTrailer::MaxStateCombinations> packetStates = {}; // packet status from the trailer

  void clear()
  {
    nPackets = 0;
    nTriggers = 0;
    errorCounts.fill(0);
    packetStates.fill(0);
  }

  void print(bool skipNoErr = true) const;

  ClassDefNV(GBTLinkDecodingStat, 3);
};

} // namespace itsmft
} // namespace o2
#endif
