// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GBTLink.cxx
/// \brief Definitions of GBTLink class used for the ITS/MFT raw data decoding

#include <bitset>
#include <iomanip>
#include "ITSMFTReconstruction/GBTLink.h"
#include "ITSMFTReconstruction/AlpideCoder.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/Logger.h"
#include "CommonConstants/Triggers.h"

using namespace o2::itsmft;

using RDHUtils = o2::raw::RDHUtils;
using RDH = o2::header::RAWDataHeader;

///======================================================================
///                 GBT Link data decoding class
///======================================================================

///_________________________________________________________________
/// create link with given ids
GBTLink::GBTLink(uint16_t _cru, uint16_t _fee, uint8_t _ep, uint8_t _idInCru, uint16_t _chan) : idInCRU(_idInCru), cruID(_cru), feeID(_fee), endPointID(_ep), channelID(_chan)
{
  chipStat.id = _fee;
}

///_________________________________________________________________
/// create string describing the link
std::string GBTLink::describe() const
{
  std::stringstream ss;
  ss << "Link cruID=0x" << std::hex << std::setw(4) << std::setfill('0') << cruID << std::dec
     << "/lID=" << int(idInCRU) << "/feeID=0x" << std::hex << std::setw(4) << std::setfill('0') << feeID << std::dec;
  if (lanes) {
    ss << " lanes: " << std::bitset<28>(lanes).to_string();
  }
  return ss.str();
}

///_________________________________________________________________
/// reset link
void GBTLink::clear(bool resetStat, bool resetTFRaw)
{
  data.clear();
  lastPageSize = 0;
  nTriggers = 0;
  lanes = 0;
  lanesActive = lanesStop = lanesTimeOut = lanesWithData = 0;
  errorBits = 0;
  if (resetTFRaw) {
    rawData.clear();
    dataOffset = 0;
  }
  //  lastRDH = nullptr;
  if (resetStat) {
    statistics.clear();
  }
  status = None;
}

///_________________________________________________________________
void GBTLink::printTrigger(const GBTTrigger* gbtTrg)
{
  gbtTrg->printX();
  std::bitset<12> trb(gbtTrg->triggerType);
  LOG(INFO) << "Trigger : Orbit " << gbtTrg->orbit << " BC: " << gbtTrg->bc << " Trigger: " << trb << " noData:" << gbtTrg->noData << " internal:" << gbtTrg->internal;
}

///_________________________________________________________________
void GBTLink::printCalibrationWord(const GBTCalibration* gbtCal)
{
  gbtCal->printX();
  LOGF(INFO, "Calibration word %5d | user_data 0x%06lx", gbtCal->calibCounter, gbtCal->calibUserField);
}

///_________________________________________________________________
void GBTLink::printHeader(const GBTDataHeader* gbtH)
{
  gbtH->printX();
  std::bitset<28> LA(gbtH->activeLanes);
  LOG(INFO) << "Header : Active Lanes " << LA;
}

///_________________________________________________________________
void GBTLink::printHeader(const GBTDataHeaderL* gbtH)
{
  gbtH->printX();
  std::bitset<28> LA(gbtH->activeLanesL);
  LOG(INFO) << "HeaderL : Active Lanes " << LA;
}

///_________________________________________________________________
void GBTLink::printTrailer(const GBTDataTrailer* gbtT)
{
  gbtT->printX();
  std::bitset<28> LT(gbtT->lanesTimeout), LS(gbtT->lanesStops); // RSTODO
  LOG(INFO) << "Trailer: Done=" << gbtT->packetDone << " Lanes TO: " << LT << " | Lanes ST: " << LS;
}

///_________________________________________________________________
void GBTLink::printDiagnostic(const GBTDiagnostic* gbtD)
{
  gbtD->printX();
  LOG(INFO) << "Diagnostic word";
}

///_________________________________________________________________
void GBTLink::printCableDiagnostic(const GBTCableDiagnostic* gbtD)
{
  gbtD->printX();
  LOGF(INFO, "Diagnostic for %s Lane %d | errorID: %d data 0x%016lx", gbtD->isIB() ? "IB" : "OB", gbtD->getCableID(), gbtD->laneErrorID, gbtD->diagnosticData);
}

///_________________________________________________________________
void GBTLink::printCableStatus(const GBTCableStatus* gbtS)
{
  gbtS->printX();
  LOGF(INFO, "Status data, not processed at the moment");
}

///====================================================================

#ifdef _RAW_READER_ERROR_CHECKS_

///_________________________________________________________________
/// Check RDH correctness
GBTLink::ErrorType GBTLink::checkErrorsRDH(const RDH& rdh)
{
  ErrorType err = NoError;
  if (!RDHUtils::checkRDH(rdh, true)) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrNoRDHAtStart]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrNoRDHAtStart] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrNoRDHAtStart];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrNoRDHAtStart);
    err = Abort;
    return err; // fatal error
  }
  if (format == OldFormat && RDHUtils::getVersion(rdh) > 4) {
    if (verbosity >= VerboseErrors) {
      LOG(ERROR) << "Requested old format requires data with RDH version 3 or 4, RDH version "
                 << RDHUtils::getVersion(rdh) << " is found";
    }
    err = Abort;
    return err;
  }
  if ((RDHUtils::getPacketCounter(rdh) > packetCounter + 1) && packetCounter >= 0) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrPacketCounterJump]
                 << " : jump from " << int(packetCounter) << " to " << int(RDHUtils::getPacketCounter(rdh));
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrPacketCounterJump);
    err = Warning;
  }
  packetCounter = RDHUtils::getPacketCounter(rdh);
  return err;
}

///_________________________________________________________________
/// Check RDH Stop correctness
GBTLink::ErrorType GBTLink::checkErrorsRDHStop(const RDH& rdh)
{
  if (format == NewFormat && lastRDH && RDHUtils::getHeartBeatOrbit(*lastRDH) != RDHUtils::getHeartBeatOrbit(rdh) // new HB starts
      && !RDHUtils::getStop(*lastRDH)) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrPageNotStopped]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrPageNotStopped] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrPageNotStopped];
      RDHUtils::printRDH(*lastRDH);
      RDHUtils::printRDH(rdh);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrPageNotStopped);
    return Warning;
  }
  return NoError;
}

///_________________________________________________________________
/// Check if the RDH Stop page is empty
GBTLink::ErrorType GBTLink::checkErrorsRDHStopPageEmpty(const RDH& rdh)
{
  if (format == NewFormat && RDHUtils::getStop(rdh) && RDHUtils::getMemorySize(rdh) != sizeof(RDH) + sizeof(GBTDiagnostic)) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrStopPageNotEmpty]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrStopPageNotEmpty] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrStopPageNotEmpty];
      RDHUtils::printRDH(rdh);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrStopPageNotEmpty);
    return Warning;
  }
  return NoError;
}

///_________________________________________________________________
/// Check the GBT Trigger word correctness
GBTLink::ErrorType GBTLink::checkErrorsTriggerWord(const GBTTrigger* gbtTrg)
{
  if (!gbtTrg->isTriggerWord()) { // check trigger word
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrigger]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrigger] == 1) {
      gbtTrg->printX();
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTTrigger];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTTrigger);
    return Abort;
  }
  return NoError;
}

///_________________________________________________________________
/// Check the GBT Calibration word correctness
GBTLink::ErrorType GBTLink::checkErrorsCalibrationWord(const GBTCalibration* gbtCal)
{
  // at the moment do nothing
  return NoError;
}

///_________________________________________________________________
/// Check the GBT Header word correctness
GBTLink::ErrorType GBTLink::checkErrorsHeaderWord(const GBTDataHeader* gbtH)
{
  if (!gbtH->isDataHeader()) { // check header word
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader] == 1) {
      gbtH->printX();
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTHeader];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTHeader);
    return Abort;
  }
  return NoError;
}

///_________________________________________________________________
/// Check the GBT Header word correctness
GBTLink::ErrorType GBTLink::checkErrorsHeaderWord(const GBTDataHeaderL* gbtH)
{
  if (!gbtH->isDataHeader()) { // check header word
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
    if (verbosity >= VerboseErrors) {
      gbtH->printX();
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTHeader];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTHeader);
    return Abort;
  }
  int cnt = RDHUtils::getPageCounter(*lastRDH);
  // RSTODO: this makes sense only for old format, where every trigger has its RDH
  if (gbtH->packetIdx != cnt) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt] << ": diff in GBT header "
                 << gbtH->packetIdx << " and RDH page " << cnt << " counters";
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt);
    return Warning;
  }
  // RSTODO CHECK
  if (lanesActive == lanesStop) { // all lanes received their stop, new page 0 expected
    //if (cnt) { // makes sens for old format only
    if (gbtH->packetIdx) {
      statistics.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]++;
      if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop] == 1) {
        LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]
                   << ": Non-0 page counter (" << cnt << ") while all lanes were stopped";
      }
      errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrNonZeroPageAfterStop);
      return Warning;
    }
  }
  return NoError;
}

///_________________________________________________________________
/// Check active lanes status
GBTLink::ErrorType GBTLink::checkErrorsActiveLanes(int cbl)
{
  if (~cbl & lanesActive) { // are there wrong lanes?
    statistics.errorCounts[GBTLinkDecodingStat::ErrInvalidActiveLanes]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrInvalidActiveLanes] == 1) {
      std::bitset<32> expectL(cbl), gotL(lanesActive);
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrInvalidActiveLanes] << ' '
                 << gotL << " vs " << expectL << " skip page";
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrInvalidActiveLanes);
    return Warning;
  }
  return NoError;
}

///_________________________________________________________________
/// Check GBT Data word
GBTLink::ErrorType GBTLink::checkErrorsGBTData(int cablePos)
{
  lanesWithData |= 0x1 << cablePos;    // flag that the data was seen on this lane
  if (lanesStop & (0x1 << cablePos)) { // make sure stopped lanes do not transmit the data
    statistics.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrDataForStoppedLane]
                 << cablePos;
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrDataForStoppedLane);
    return Warning;
  }

  return NoError;
}

///_________________________________________________________________
/// Check GBT Data word ID: it might be diagnostic or status data
GBTLink::ErrorType GBTLink::checkErrorsGBTDataID(const GBTData* gbtD)
{
  if (gbtD->isData()) {
    return NoError;
  }
  statistics.errorCounts[GBTLinkDecodingStat::ErrGBTWordNotRecognized]++;
  if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrGBTWordNotRecognized] == 1) {
    if (gbtD->isCableDiagnostic()) {
      printCableDiagnostic((GBTCableDiagnostic*)gbtD);
    } else if (gbtD->isStatus()) {
      printCableStatus((GBTCableStatus*)gbtD);
    }
    gbtD->printX(true);
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrGBTWordNotRecognized];
  }
  return Skip;
}

///_________________________________________________________________
/// Check the GBT Trailer word correctness
GBTLink::ErrorType GBTLink::checkErrorsTrailerWord(const GBTDataTrailer* gbtT)
{
  if (!gbtT->isDataTrailer()) {
    gbtT->printX();
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTTrailer];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTTrailer);
    return Abort;
  }
  lanesTimeOut |= gbtT->lanesTimeout; // register timeouts
  lanesStop |= gbtT->lanesStops;      // register stops
  return NoError;
}

///_________________________________________________________________
/// Check the Done status in GBT Trailer word
GBTLink::ErrorType GBTLink::checkErrorsPacketDoneMissing(const GBTDataTrailer* gbtT, bool notEnd)
{
  if (!gbtT->packetDone && notEnd) { // Done may be missing only in case of carry-over to new CRU page
    statistics.errorCounts[GBTLinkDecodingStat::ErrPacketDoneMissing]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrPacketDoneMissing] == 1) {
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrPacketDoneMissing];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrPacketDoneMissing);
    return Warning;
  }
  return NoError;
}

///_________________________________________________________________
/// Check that all active lanes received their stop
GBTLink::ErrorType GBTLink::checkErrorsLanesStops()
{
  // make sure all lane stops for finished page are received
  auto err = NoError;
  if ((lanesActive & ~lanesStop)) {
    if (RDHUtils::getTriggerType(*lastRDH) != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
      statistics.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes]++;
      if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes] == 1) {
        std::bitset<32> active(lanesActive), stopped(lanesStop);
        LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrUnstoppedLanes]
                   << " | active: " << active << " stopped: " << stopped;
      }
      errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrUnstoppedLanes);
    }
    err = Warning;
  }
  // make sure all active lanes (except those in time-out) have sent some data
  if ((~lanesWithData & lanesActive) != lanesTimeOut) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane] == 1) {
      std::bitset<32> withData(lanesWithData), active(lanesActive), timeOut(lanesTimeOut);
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrNoDataForActiveLane]
                 << " | with data: " << withData << " active: " << active << " timeOut: " << timeOut;
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrNoDataForActiveLane);
    err = Warning;
  }
  return err;
}

///_________________________________________________________________
/// Check diagnostic word
GBTLink::ErrorType GBTLink::checkErrorsDiagnosticWord(const GBTDiagnostic* gbtD)
{
  if (RDHUtils::getMemorySize(lastRDH) != sizeof(RDH) + sizeof(GBTDiagnostic) || !gbtD->isDiagnosticWord()) { //
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingDiagnosticWord]++;
    if (verbosity >= VerboseErrors || statistics.errorCounts[GBTLinkDecodingStat::ErrMissingDiagnosticWord] == 1) {
      gbtD->printX();
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingDiagnosticWord];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingDiagnosticWord);
    return Abort;
  }
  return NoError;
}

#endif
