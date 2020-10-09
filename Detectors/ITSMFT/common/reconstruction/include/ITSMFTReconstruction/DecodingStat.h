// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

struct ChipStat {

  enum DecErrors : int {
    BusyViolation,
    DataOverrun,
    Fatal,
    BusyOn,
    BusyOff,
    TruncatedChipEmpty,   // Data was truncated after ChipEmpty
    TruncatedChipHeader,  // Data was truncated after ChipHeader
    TruncatedRegion,      // Data was truncated after Region record
    TruncatedLondData,    // Data was truncated in the LongData record
    WrongDataLongPattern, // LongData pattern has highest bit set
    NoDataFound,          // Region is not followed by Short or Long data
    UnknownWord,          // Unknow word was seen
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
    "Unknow word"                                   // UnknownWord
  };

  uint16_t chipID = 0;
  size_t nHits = 0;
  std::array<uint32_t, NErrorsDefined> errorCounts = {};
  ChipStat() = default;
  ChipStat(uint16_t id) : chipID(id) {}

  void clear()
  {
    memset(errorCounts.data(), 0, sizeof(uint32_t) * errorCounts.size());
    nHits = 0;
  }

  uint32_t getNErrors() const;

  void addErrors(uint32_t mask)
  {
    if (mask) {
      for (int i = NErrorsDefined; i--;) {
        if (mask & (0x1 << i)) {
          errorCounts[i]++;
        }
      }
    }
  }

  void print(bool skipEmpty = true) const;

  ClassDefNV(ChipStat, 1);
};

/// Statistics for per-link decoding
struct GBTLinkDecodingStat {
  /// counters for format checks

  enum DecErrors : int {
    ErrNoRDHAtStart,             // page does not start with RDH
    ErrPageNotStopped,           // new HB/trigger page started w/o stopping previous one
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
    NErrorsDefined
  };
  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "Page data not start with expected RDH",                             // ErrNoRDHAtStart
    "New HB/trigger page started w/o stopping previous one",             // ErrPageNotStopped
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
    "Missing diagnostic GBT word after RDH with stop"                    // ErrMissingDiagnosticWord
  };

  uint32_t ruLinkID = 0; // Link ID within RU

  // Note: packet here is meant as a group of CRU pages belonging to the same trigger
  uint32_t nPackets = 0;                                                        // total number of packets
  std::array<uint32_t, NErrorsDefined> errorCounts = {};                        // error counters
  std::array<uint32_t, GBTDataTrailer::MaxStateCombinations> packetStates = {}; // packet status from the trailer

  void clear()
  {
    nPackets = 0;
    errorCounts.fill(0);
    packetStates.fill(0);
  }

  void print(bool skipEmpty = true) const;

  ClassDefNV(GBTLinkDecodingStat, 1);
};

} // namespace itsmft
} // namespace o2
#endif
