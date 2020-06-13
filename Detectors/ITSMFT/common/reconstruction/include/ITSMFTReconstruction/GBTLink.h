// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GBTLink.h
/// \brief Declarations of helper classes for the ITS/MFT raw data decoding

#ifndef _ALICEO2_ITSMFT_GBTLINK_H_
#define _ALICEO2_ITSMFT_GBTLINK_H_

#define _RAW_READER_ERROR_CHECKS_ // comment this to disable error checking

#include <string>
#include <memory>
#include <gsl/gsl>
#include "ITSMFTReconstruction/PayLoadCont.h"
#include "ITSMFTReconstruction/PayLoadSG.h"
#include "ITSMFTReconstruction/GBTWord.h"
#include "ITSMFTReconstruction/RUDecodeData.h"
#include "ITSMFTReconstruction/RUInfo.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "CommonDataFormat/InteractionRecord.h"

#define GBTLINK_DECODE_ERRORCHECK(what) \
  if ((what) == Abort) {                \
    discardData();                      \
    return AbortedOnError;              \
  }

namespace o2
{
namespace itsmft
{

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
    NErrorsDefined
  };
  static constexpr std::array<std::string_view, NErrorsDefined> ErrNames = {
    "Page data not start with expected RDH",                             // ErrNoRDHAtStart
    "New HB/trigger page started w/o stopping previous one",             // ErrPageNotStopped
    "Page with RDH.stop is not empty",                                   // ErrStopPageNotEmpty
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
    "Packet done is missing in the trailer while CRU page is not over"   // ErrPacketDoneMissing
  };

  uint32_t ruLinkID = 0; // Link ID within RU

  // Note: packet here is meant as a group of CRU pages belonging to the same trigger
  uint32_t nPackets = 0;                                                   // total number of packets
  std::array<int, NErrorsDefined> errorCounts = {};                        // error counters
  std::array<int, GBTDataTrailer::MaxStateCombinations> packetStates = {}; // packet status from the trailer

  void clear()
  {
    nPackets = 0;
    errorCounts.fill(0);
    packetStates.fill(0);
  }

  void print(bool skipEmpty = true) const;

  ClassDefNV(GBTLinkDecodingStat, 1);
};

struct RUDecodeData; // forward declaration to allow its linking in the GBTlink
struct GBTHeader;
struct GBTTrailer;
struct GBTTrigger;

/// support for the GBT single link data
struct GBTLink {

  enum Format : int8_t { OldFormat,
                         NewFormat,
                         NFormats };

  enum CollectedDataStatus : int8_t { None,
                                      AbortedOnError,
                                      StoppedOnEndOfData,
                                      DataSeen }; // None is set before starting collectROFCableData

  enum ErrorType : int8_t { NoError,
                            Warning,
                            Abort };

  enum Verbosity : int8_t { Silent = -1,
                            VerboseErrors,
                            VerboseHeaders,
                            VerboseData };

  using RDH = o2::header::RDHAny;
  using RDHUtils = o2::raw::RDHUtils;

  CollectedDataStatus status = None;
  Format format = NewFormat;
  Verbosity verbosity = VerboseErrors;
  uint8_t idInRU = 0;     // link ID within the RU
  uint8_t idInCRU = 0;    // link ID within the CRU
  uint8_t endPointID = 0; // endpoint ID of the CRU
  uint16_t cruID = 0;     // CRU ID
  uint16_t feeID = 0;     // FEE ID
  uint16_t channelID = 0; // channel ID in the reader input
  uint32_t lanes = 0;     // lanes served by this link

  // RS do we need this >> ? // Legacy from old data format encoder
  int lastPageSize = 0; // size of last added page = offset from the end to get to the RDH
  int nTriggers = 0;    // number of triggers loaded (the last one might be incomplete)
  // << ?

  PayLoadCont data; // data buffer used for encoding

  // transient data filled from current RDH
  uint32_t lanesActive = 0;   // lanes declared by the payload header
  uint32_t lanesStop = 0;     // lanes received stop in the payload trailer
  uint32_t lanesTimeOut = 0;  // lanes received timeout
  uint32_t lanesWithData = 0; // lanes with data transmitted
  int32_t packetCounter = -1; // current packet counter from RDH (RDH.packetCounter)
  uint32_t trigger = 0;       // trigger word
  uint32_t errorBits = 0;     // bits of the error code of last frame decoding (if any)
  const RDH* lastRDH = nullptr;
  o2::InteractionRecord ir;       // interaction record
  GBTLinkDecodingStat statistics; // decoding statistics
  RUDecodeData* ruPtr = nullptr;  // pointer on the parent RU

  PayLoadSG rawData;         // scatter-gatter buffer for cached CRU pages, each starting with RDH
  size_t dataOffset = 0;     //
  size_t currOffsWrtRDH = 0; // index of 1st unread element in the current CRU page
  //------------------------------------------------------------------------

  GBTLink() = default;
  GBTLink(uint16_t _cru, uint16_t _fee, uint8_t _ep, uint8_t _idInCru = 0, uint16_t _chan = 0);
  std::string describe() const;
  void clear(bool resetStat = true, bool resetTFRaw = false);

  template <class Mapping>
  CollectedDataStatus collectROFCableData(const Mapping& chmap);

  void cacheData(const void* ptr, size_t sz)
  {
    rawData.add(reinterpret_cast<const PayLoadSG::DataType*>(ptr), sz);
  }

 private:
  void discardData() { rawData.setDone(); }
  void printTrigger(const GBTTrigger* gbtTrg);
  void printHeader(const GBTDataHeader* gbtH);
  void printTrailer(const GBTDataTrailer* gbtT);
  bool nextCRUPage();

#ifndef _RAW_READER_ERROR_CHECKS_ // define dummy inline check methods, will be compiled out
  bool checkErrorsRDH(const RDH& rdh) const
  {
    return true;
  }
  ErrorType checkErrorsRDHStop(const RDH& rdh) const { return NoError; }
  ErrorType checkErrorsRDHStopPageEmpty(const RDH& rdh) const { return NoError; }
  ErrorType checkErrorsTriggerWord(const GBTTrigger* gbtTrg) const { return NoError; }
  ErrorType checkErrorsHeaderWord(const GBTDataHeader* gbtH) const { return NoError; }
  ErrorType checkErrorsActiveLanes(int cables) const { return NoError; }
  ErrorType checkErrorsGBTData(int cablePos) const { return NoError; }
  ErrorType checkErrorsTrailerWord(const GBTDataTrailer* gbtT) const { return NoError; }
  ErrorType checkErrorsPacketDoneMissing(const GBTDataTrailer* gbtT, bool notEnd) const { return NoError; }
  ErrorType checkErrorsLanesStops() const { return NoError; }
#else
  ErrorType checkErrorsRDH(const RDH& rdh);
  ErrorType checkErrorsRDHStop(const RDH& rdh);
  ErrorType checkErrorsRDHStopPageEmpty(const RDH& rdh);
  ErrorType checkErrorsTriggerWord(const GBTTrigger* gbtTrg);
  ErrorType checkErrorsHeaderWord(const GBTDataHeader* gbtH);
  ErrorType checkErrorsActiveLanes(int cables);
  ErrorType checkErrorsGBTData(int cablePos);
  ErrorType checkErrorsTrailerWord(const GBTDataTrailer* gbtT);
  ErrorType checkErrorsPacketDoneMissing(const GBTDataTrailer* gbtT, bool notEnd);
  ErrorType checkErrorsLanesStops();
#endif

  ClassDefNV(GBTLink, 1);
};

///_________________________________________________________________
/// collect cables data for single ROF, return number of real payload words seen,
/// -1 in case of critical error
template <class Mapping>
GBTLink::CollectedDataStatus GBTLink::collectROFCableData(const Mapping& chmap)
{
  int nw = 0;
  status = None;
  auto* currRawPiece = rawData.currentPiece();
  while (currRawPiece) { // we may loop over multiple CRU page
    if (dataOffset >= currRawPiece->size) {
      dataOffset = 0;                              // start of the RDH
      if (!(currRawPiece = rawData.nextPiece())) { // fetch next CRU page
        break;                                     // Data chunk (TF?) is done
      }
    }
    if (!dataOffset) { // here we always start with the RDH
      const auto* rdh = reinterpret_cast<const RDH*>(&currRawPiece->data[dataOffset]);
      if (verbosity >= VerboseHeaders) {
        RDHUtils::printRDH(rdh);
      }
      GBTLINK_DECODE_ERRORCHECK(checkErrorsRDH(*rdh));              // make sure we are dealing with RDH
      GBTLINK_DECODE_ERRORCHECK(checkErrorsRDHStop(*rdh));          // if new HB starts, the lastRDH must have stop
      GBTLINK_DECODE_ERRORCHECK(checkErrorsRDHStopPageEmpty(*rdh)); // end of HBF should be an empty page with stop
      lastRDH = rdh;
      statistics.nPackets++;
      dataOffset += sizeof(RDH);
      continue;
    }

    ruPtr->nCables = ruPtr->ruInfo->nCables; // RSTODO is this needed? TOREMOVE

    // data must start with GBT trigger word (unless we work with old format)
    const GBTTrigger* gbtTrg = nullptr;
    if (format == NewFormat) {
      gbtTrg = reinterpret_cast<const GBTTrigger*>(&currRawPiece->data[dataOffset]); // process GBT trigger
      dataOffset += GBTPaddedWordLength;
      if (verbosity >= VerboseHeaders) {
        printTrigger(gbtTrg);
      }
      GBTLINK_DECODE_ERRORCHECK(checkErrorsTriggerWord(gbtTrg));
    }

    // next the GBTHeader must come
    auto gbtH = reinterpret_cast<const GBTDataHeader*>(&currRawPiece->data[dataOffset]); // process GBT header
    dataOffset += GBTPaddedWordLength;
    if (verbosity >= VerboseHeaders) {
      printHeader(gbtH);
    }
    GBTLINK_DECODE_ERRORCHECK(checkErrorsHeaderWord(gbtH));
    lanesActive = gbtH->activeLanes; // TODO do we need to update this for every page?
    GBTLINK_DECODE_ERRORCHECK(checkErrorsActiveLanes(chmap.getCablesOnRUType(ruPtr->ruInfo->ruType)));
    if (format == OldFormat && RDHUtils::getPageCounter(*lastRDH)) { // RSTODO reset flags in case of 1st page of new ROF (old format: judge by RDH)
      lanesStop = 0;
      lanesWithData = 0;
    } else if (gbtH->packetIdx == 0) { // reset flags in case of 1st page of new ROF (new format: judge by header)
      lanesStop = 0;
      lanesWithData = 0;
    }
    auto gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(&currRawPiece->data[dataOffset]);
    while (!gbtD->isDataTrailer()) { // start reading real payload
      nw++;
      int cableHW = gbtD->getCableID(), cableSW = chmap.cableHW2SW(ruPtr->ruInfo->ruType, cableHW);
      if (verbosity >= VerboseData) {
        gbtD->printX();
      }
      GBTLINK_DECODE_ERRORCHECK(checkErrorsGBTData(chmap.cableHW2Pos(ruPtr->ruInfo->ruType, cableHW)));
      ruPtr->cableData[cableSW].add(gbtD->getW8(), 9);
      ruPtr->cableHWID[cableSW] = cableHW;
      ruPtr->cableLinkID[cableSW] = idInRU;
      ruPtr->cableLinkPtr[cableSW] = this;
      dataOffset += GBTPaddedWordLength;
      gbtD = reinterpret_cast<const o2::itsmft::GBTData*>(&currRawPiece->data[dataOffset]);
    } // we are at the trailer, packet is over, check if there are more data on the next page

    auto gbtT = reinterpret_cast<const o2::itsmft::GBTDataTrailer*>(&currRawPiece->data[dataOffset]); // process GBT trailer
    dataOffset += GBTPaddedWordLength;
    if (verbosity >= VerboseHeaders) {
      printTrailer(gbtT);
    }
    GBTLINK_DECODE_ERRORCHECK(checkErrorsTrailerWord(gbtT));
    // we finished the GBT page, but there might be continuation on the next CRU page
    if (!gbtT->packetDone) {
      GBTLINK_DECODE_ERRORCHECK(checkErrorsPacketDoneMissing(gbtT, dataOffset < currRawPiece->size));
      continue; // keep reading next CRU page
    }
    GBTLINK_DECODE_ERRORCHECK(checkErrorsLanesStops());
    // accumulate packet states
    statistics.packetStates[gbtT->getPacketState()]++;
    // before quitting, store the trigger and IR
    if (format == NewFormat) {
      ir.bc = gbtTrg->bc;
      ir.orbit = gbtTrg->orbit;
      trigger = gbtTrg->triggerType;
    } else {
      ir = RDHUtils::getTriggerIR(lastRDH);
      trigger = RDHUtils::getTriggerType(lastRDH);
    }
    return (status = DataSeen);
  }

  return (status = StoppedOnEndOfData);
}

} // namespace itsmft
} // namespace o2

#endif // _ALICEO2_ITSMFT_GBTLINK_H_
