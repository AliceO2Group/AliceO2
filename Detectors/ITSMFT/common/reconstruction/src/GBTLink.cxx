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
  chipStat.feeID = _fee;
  statistics.feeID = _fee;
}

///_________________________________________________________________
/// create string describing the link
std::string GBTLink::describe() const
{
  std::string ss = fmt::format("link cruID:{:#06x}/lID{} feeID:{:#06x}", cruID, int(idInCRU), feeID);
  if (lanes) {
    ss += fmt::format(" lanes {}", std::bitset<28>(lanes).to_string());
  }
  return ss;
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
  packetCounter = -1;
  errorBits = 0;
  irHBF.clear();
  if (resetTFRaw) {
    rawData.clear();
    dataOffset = 0;
    gbtErrStatUpadated = false;
  }
  //  lastRDH = nullptr;
  if (resetStat) {
    statistics.clear();
  }
  hbfEntry = 0;
  extTrigVec = nullptr;
  status = None;
}

///_________________________________________________________________
void GBTLink::printTrigger(const GBTTrigger* gbtTrg, int offs)
{
  std::bitset<12> trb(gbtTrg->triggerType);
  LOG(info) << "Offs: " << offs << " Trigger : Orbit " << gbtTrg->orbit << " BC: " << gbtTrg->bc << " Trigger: " << trb << " noData:"
            << gbtTrg->noData << " internal:" << gbtTrg->internal << " continuation:" << gbtTrg->continuation << " on " << describe();
  gbtTrg->printX(expectPadding);
}

///_________________________________________________________________
void GBTLink::printCalibrationWord(const GBTCalibration* gbtCal, int offs)
{
  LOGP(info, "Offs: {} Calibration word {:5} | user_data {:#08x} on {}", offs, gbtCal->calibCounter, gbtCal->calibUserField, describe());
  gbtCal->printX(expectPadding);
}

///_________________________________________________________________
void GBTLink::printHeader(const GBTDataHeader* gbtH, int offs)
{
  std::bitset<28> LA(gbtH->activeLanes);
  LOG(info) << "Offs: " << offs << " Header : Active Lanes " << LA << " on " << describe();
  gbtH->printX(expectPadding);
}

///_________________________________________________________________
void GBTLink::printHeader(const GBTDataHeaderL* gbtH, int offs)
{
  std::bitset<28> LA(gbtH->activeLanesL);
  LOG(info) << "Offs: " << offs << " HeaderL : Active Lanes " << LA << " on " << describe();
  gbtH->printX(expectPadding);
}

///_________________________________________________________________
void GBTLink::printTrailer(const GBTDataTrailer* gbtT, int offs)
{
  std::bitset<28> LT(gbtT->lanesTimeout), LS(gbtT->lanesStops); // RSTODO
  LOG(info) << "Offs: " << offs << " Trailer: Done=" << gbtT->packetDone << " Lanes TO: " << LT << " | Lanes ST: " << LS << " on " << describe();
  gbtT->printX(expectPadding);
}

///_________________________________________________________________
void GBTLink::printDiagnostic(const GBTDiagnostic* gbtD, int offs)
{
  LOG(info) << "Offs: " << offs << " Diagnostic word on " << describe();
  gbtD->printX(expectPadding);
}

///_________________________________________________________________
void GBTLink::printCableDiagnostic(const GBTCableDiagnostic* gbtD)
{
  LOGP(info, "Diagnostic for {} Lane {} | errorID: {} data {:#018x} on {}", gbtD->isIB() ? "IB" : "OB", gbtD->getCableID(), gbtD->laneErrorID, gbtD->diagnosticData, describe());
  gbtD->printX(expectPadding);
}

///_________________________________________________________________
void GBTLink::printCableStatus(const GBTCableStatus* gbtS)
{
  LOGP(info, "Status data, not processed at the moment, on {}", describe());
  gbtS->printX(expectPadding);
}

///====================================================================

#ifdef _RAW_READER_ERROR_CHECKS_

///_________________________________________________________________
/// Check RDH correctness
uint8_t GBTLink::checkErrorsRDH(const RDH& rdh)
{
  uint8_t err = uint8_t(NoError);
  if (!RDHUtils::checkRDH(rdh, true)) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrNoRDHAtStart]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrNoRDHAtStart])) {
      err |= uint8_t(ErrorPrinted);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrNoRDHAtStart];
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrNoRDHAtStart);
    err |= uint8_t(Abort);
    return err; // fatal error
  }
  if (expectPadding && (RDHUtils::getPacketCounter(rdh) > packetCounter + 1) && packetCounter >= 0) { // packet counter check makes sense only for data with padding (no UL)
    if (irHBF.isDummy()) {
      irHBF = RDHUtils::getHeartBeatIR(rdh);
    }
    statistics.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrPacketCounterJump])) {
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrPacketCounterJump]
                << " : jump from " << int(packetCounter) << " to " << int(RDHUtils::getPacketCounter(rdh));
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrPacketCounterJump);
    err |= uint8_t(Warning);
  }
  packetCounter = RDHUtils::getPacketCounter(rdh);
  return err;
}

/// Check encountered alignment padding word correctness
uint8_t GBTLink::checkErrorsAlignmentPadding()
{
  uint8_t err = uint8_t(NoError);
  if (lastPageSize - dataOffset >= CRUPageAlignment) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrWrongAlignmentWord]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrWrongAlignmentWord])) {
      err |= uint8_t(ErrorPrinted);
      LOG(info) << describe() << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrWrongAlignmentWord] << " at offset " << dataOffset << " for page size " << lastPageSize;
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrWrongAlignmentWord);
    err |= uint8_t(Warning);
    return err; // fatal error
  }
  return err;
}

///_________________________________________________________________
/// Check RDH Stop correctness
uint8_t GBTLink::checkErrorsRDHStop(const RDH& rdh)
{
  uint8_t err = uint8_t(NoError);
  if (lastRDH && RDHUtils::getHeartBeatOrbit(*lastRDH) != RDHUtils::getHeartBeatOrbit(rdh) // new HB starts
      && !RDHUtils::getStop(*lastRDH)) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrPageNotStopped]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrPageNotStopped])) {
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrPageNotStopped];
      RDHUtils::printRDH(*lastRDH);
      RDHUtils::printRDH(rdh);
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrPageNotStopped);
    err |= uint8_t(Warning);
  }
  return err;
}

///_________________________________________________________________
/// Check if the RDH Stop page is empty
uint8_t GBTLink::checkErrorsRDHStopPageEmpty(const RDH& rdh)
{
  uint8_t err = uint8_t(NoError);
  if (RDHUtils::getStop(rdh) && RDHUtils::getMemorySize(rdh) != sizeof(RDH) + wordLength) { // there could be only 1 diagnostic GBTWord after stop
    statistics.errorCounts[GBTLinkDecodingStat::ErrStopPageNotEmpty]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrStopPageNotEmpty])) {
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrStopPageNotEmpty];
      RDHUtils::printRDH(rdh);
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrStopPageNotEmpty);
    err |= uint8_t(Warning);
  }
  return err;
}

///_________________________________________________________________
/// Check the GBT Trigger word correctness
uint8_t GBTLink::checkErrorsTriggerWord(const GBTTrigger* gbtTrg)
{
  uint8_t err = uint8_t(NoError);
  if (!gbtTrg->isTriggerWord()) { // check trigger word
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrigger]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrigger])) {
      gbtTrg->printX(expectPadding);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTTrigger];
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTTrigger);
    err |= uint8_t(Abort);
  }
  return err;
}

///_________________________________________________________________
/// Check the GBT Calibration word correctness
uint8_t GBTLink::checkErrorsCalibrationWord(const GBTCalibration* gbtCal)
{
  // at the moment do nothing
  return uint8_t(NoError);
}

///_________________________________________________________________
/// Check the GBT Header word correctness
uint8_t GBTLink::checkErrorsHeaderWord(const GBTDataHeader* gbtH)
{
  uint8_t err = uint8_t(NoError);
  if (!gbtH->isDataHeader()) { // check header word
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader])) {
      gbtH->printX(expectPadding);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTHeader];
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTHeader);
    err |= uint8_t(Abort);
  }
  return err;
}

///_________________________________________________________________
/// Check the GBT Header word correctness
uint8_t GBTLink::checkErrorsHeaderWord(const GBTDataHeaderL* gbtH)
{
  uint8_t err = uint8_t(NoError);
  if (!gbtH->isDataHeader()) { // check header word
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTHeader]++;
    gbtErrStatUpadated = true;
    if (verbosity >= VerboseErrors) {
      gbtH->printX(expectPadding);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTHeader];
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTHeader);
    err |= uint8_t(Abort);
    return err;
  }
  int cnt = RDHUtils::getPageCounter(*lastRDH);
  // RSTODO: this makes sense only for old format, where every trigger has its RDH
  if (gbtH->packetIdx != cnt) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt])) {
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt] << ": diff in GBT header "
                << gbtH->packetIdx << " and RDH page " << cnt << " counters";
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrRDHvsGBTHPageCnt);
    err |= uint8_t(Warning);
    return err;
  }
  // RSTODO CHECK
  if (lanesActive == lanesStop) { // all lanes received their stop, new page 0 expected
    //if (cnt) { // makes sens for old format only
    if (gbtH->packetIdx) {
      statistics.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]++;
      gbtErrStatUpadated = true;
      if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrNonZeroPageAfterStop])) {
        LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrNonZeroPageAfterStop]
                  << ": Non-0 page counter (" << cnt << ") while all lanes were stopped";
        err |= uint8_t(ErrorPrinted);
      }
      errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrNonZeroPageAfterStop);
      err |= uint8_t(Warning);
    }
  }
  return err;
}

///_________________________________________________________________
/// Check active lanes status
uint8_t GBTLink::checkErrorsActiveLanes(int cbl)
{
  uint8_t err = uint8_t(NoError);
  if (~cbl & lanesActive) { // are there wrong lanes?
    statistics.errorCounts[GBTLinkDecodingStat::ErrInvalidActiveLanes]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrInvalidActiveLanes])) {
      std::bitset<32> expectL(cbl), gotL(lanesActive);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrInvalidActiveLanes] << ' '
                << gotL << " vs " << expectL << " skip page";
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrInvalidActiveLanes);
    err |= uint8_t(Warning);
  }
  return err;
}

///_________________________________________________________________
/// Check GBT Data word
uint8_t GBTLink::checkErrorsGBTData(int cablePos)
{
  uint8_t err = uint8_t(NoError);
  lanesWithData |= 0x1 << cablePos;    // flag that the data was seen on this lane
  if (lanesStop & (0x1 << cablePos)) { // make sure stopped lanes do not transmit the data
    statistics.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrDataForStoppedLane])) {
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrDataForStoppedLane] << cablePos;
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrDataForStoppedLane);
    err |= uint8_t(Warning);
  }

  return err;
}

///_________________________________________________________________
/// Check GBT Data word ID: it might be diagnostic or status data
uint8_t GBTLink::checkErrorsGBTDataID(const GBTData* gbtD)
{
  if (gbtD->isData()) {
    return uint8_t(NoError);
  }
  uint8_t err = uint8_t(NoError);
  statistics.errorCounts[GBTLinkDecodingStat::ErrGBTWordNotRecognized]++;
  gbtErrStatUpadated = true;
  if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrGBTWordNotRecognized])) {
    if (gbtD->isCableDiagnostic()) {
      printCableDiagnostic((GBTCableDiagnostic*)gbtD);
    } else if (gbtD->isStatus()) {
      printCableStatus((GBTCableStatus*)gbtD);
    }
    gbtD->printX(true);
    LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrGBTWordNotRecognized];
    err |= uint8_t(ErrorPrinted);
  }
  err |= uint8_t(Skip);
  return err;
}

///_________________________________________________________________
/// Check the GBT Trailer word correctness
uint8_t GBTLink::checkErrorsTrailerWord(const GBTDataTrailer* gbtT)
{
  uint8_t err = uint8_t(NoError);
  if (!gbtT->isDataTrailer()) {
    gbtT->printX(expectPadding);
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrMissingGBTTrailer])) {
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingGBTTrailer];
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingGBTTrailer);
    err |= uint8_t(Abort);
    return err;
  }
  lanesTimeOut |= gbtT->lanesTimeout; // register timeouts
  lanesStop |= gbtT->lanesStops;      // register stops
  return err;
}

///_________________________________________________________________
/// Check the Done status in GBT Trailer word
uint8_t GBTLink::checkErrorsPacketDoneMissing(const GBTDataTrailer* gbtT, bool notEnd)
{
  uint8_t err = uint8_t(NoError);
  if (!gbtT || (!gbtT->packetDone && notEnd)) { // Done may be missing only in case of carry-over to new CRU page
    statistics.errorCounts[GBTLinkDecodingStat::ErrPacketDoneMissing]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrPacketDoneMissing])) {
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrPacketDoneMissing];
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrPacketDoneMissing);
    err |= uint8_t(Warning);
  }
  return err;
}

///_________________________________________________________________
/// Check that all active lanes received their stop
uint8_t GBTLink::checkErrorsLanesStops()
{
  // make sure all lane stops for finished page are received
  uint8_t err = uint8_t(NoError);
  if ((lanesActive & ~lanesStop)) {
    if (RDHUtils::getTriggerType(*lastRDH) != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
      statistics.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes]++;
      gbtErrStatUpadated = true;
      if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrUnstoppedLanes])) {
        std::bitset<32> active(lanesActive), stopped(lanesStop);
        LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrUnstoppedLanes]
                  << " | active: " << active << " stopped: " << stopped;
        err |= uint8_t(ErrorPrinted);
      }
      errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrUnstoppedLanes);
    }
    err |= uint8_t(Warning);
  }
  // make sure all active lanes (except those in time-out) have sent some data
  if ((~lanesWithData & lanesActive) != lanesTimeOut) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrNoDataForActiveLane])) {
      std::bitset<32> withData(lanesWithData), active(lanesActive), timeOut(lanesTimeOut);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrNoDataForActiveLane]
                << " | with data: " << withData << " active: " << active << " timeOut: " << timeOut;
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrNoDataForActiveLane);
    err |= uint8_t(Warning);
  }
  return err;
}

///_________________________________________________________________
/// Check diagnostic word
uint8_t GBTLink::checkErrorsDiagnosticWord(const GBTDiagnostic* gbtD)
{
  uint8_t err = uint8_t(NoError);
  if (!gbtD->isDiagnosticWord() || (lastPageSize > sizeof(RDH) + CRUPageAlignment)) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrMissingDiagnosticWord]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrMissingDiagnosticWord])) {
      gbtD->printX(expectPadding);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrMissingDiagnosticWord];
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrMissingDiagnosticWord);
    err |= uint8_t(Abort);
  }
  return err;
}

///_________________________________________________________________
/// Check cable ID validity
uint8_t GBTLink::checkErrorsCableID(const GBTData* gbtD, uint8_t cableSW)
{
  uint8_t err = uint8_t(NoError);
  if (cableSW == 0xff) {
    statistics.errorCounts[GBTLinkDecodingStat::ErrWrongeCableID]++;
    gbtErrStatUpadated = true;
    if (needToPrintError(statistics.errorCounts[GBTLinkDecodingStat::ErrWrongeCableID])) {
      gbtD->printX(expectPadding);
      LOG(info) << describe() << ' ' << irHBF << ' ' << statistics.ErrNames[GBTLinkDecodingStat::ErrWrongeCableID] << ' ' << gbtD->getCableID();
      err |= uint8_t(ErrorPrinted);
    }
    errorBits |= 0x1 << int(GBTLinkDecodingStat::ErrWrongeCableID);
    err |= uint8_t(Skip);
  }
  return err;
}

#endif
