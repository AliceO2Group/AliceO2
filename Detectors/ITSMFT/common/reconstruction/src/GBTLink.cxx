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

///======================================================================
///             Decoding statistics for a single GBT link
///======================================================================
using GBTS = GBTLinkDecodingStat;
using RDHUtils = o2::raw::RDHUtils;
using RDH = o2::header::RAWDataHeader;

constexpr std::array<std::string_view, GBTS::NErrorsDefined> GBTS::ErrNames;

///_________________________________________________________________
/// print link decoding statistics
void GBTS::print(bool skipEmpty) const
{
  int nErr = 0;
  for (int i = NErrorsDefined; i--;) {
    nErr += errorCounts[i];
  }
  printf("GBTLink#0x%d Packet States Statistics (total packets: %d)\n", ruLinkID, nPackets);
  for (int i = 0; i < GBTDataTrailer::MaxStateCombinations; i++) {
    if (packetStates[i]) {
      std::bitset<GBTDataTrailer::NStatesDefined> patt(i);
      printf("counts for triggers B[%s] : %d\n", patt.to_string().c_str(), packetStates[i]);
    }
  }
  printf("Decoding errors: %d\n", nErr);
  for (int i = 0; i < NErrorsDefined; i++) {
    if (!skipEmpty || errorCounts[i]) {
      printf("%-70s: %d\n", ErrNames[i].data(), errorCounts[i]);
    }
  }
}

///======================================================================
///                 GBT Link data decoding class
///======================================================================

///_________________________________________________________________
/// create link with given ids
GBTLink::GBTLink(uint16_t _cru, uint16_t _fee, uint8_t _ep, uint8_t _idInCru, uint16_t _chan) : idInCRU(_idInCru), cruID(_cru), feeID(_fee), endPointID(_ep), channelID(_chan)
{
}

///_________________________________________________________________
/// create string describing the link
std::string GBTLink::describe() const
{
  std::stringstream ss;
  ss << "Link cruID=0x" << std::hex << std::setw(4) << std::setfill('0') << cruID << std::dec
     << "/lID=" << int(idInCRU) << "/feeID=0x" << std::hex << std::setw(4) << std::setfill('0') << feeID << std::dec
     << " lanes: " << std::bitset<28>(lanes).to_string();
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
}

///_________________________________________________________________
void GBTLink::printRDH(const RDH* rdh)
{
  o2::raw::RDHUtils::printRDH(*rdh);
}

///_________________________________________________________________
void GBTLink::printTrigger(const GBTTrigger* gbtTrg)
{
  gbtTrg->printX();
  LOG(INFO) << "Trigger : Orbit " << gbtTrg->orbit << " BC: " << gbtTrg->bc;
}

///_________________________________________________________________
void GBTLink::printHeader(const GBTDataHeader* gbtH)
{
  gbtH->printX();
  std::bitset<28> LA(gbtH->activeLanes);
  LOG(INFO) << "Header : Active Lanes " << LA;
}

///_________________________________________________________________
void GBTLink::printTrailer(const GBTDataTrailer* gbtT)
{
  gbtT->printX();
  std::bitset<28> LT(gbtT->lanesTimeout), LS(gbtT->lanesStops); // RSTODO
  LOG(INFO) << "Trailer: Done=" << gbtT->packetDone << " Lanes TO: " << LT << " | Lanes ST: " << LS;
}

///====================================================================

#ifdef _RAW_READER_ERROR_CHECKS_

///_________________________________________________________________
/// Check RDH correctness
GBTLink::ErrorType GBTLink::checkErrorsRDH(const RDH* rdh)
{
  if (!RDHUtils::checkRDH(*rdh, true)) {
    statistics.errorCounts[GBTS::ErrNoRDHAtStart]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrNoRDHAtStart];
    errorBits |= 0x1 << int(GBTS::ErrNoRDHAtStart);
    return Abort; // fatal error
  }
  if ((rdh->packetCounter > packetCounter + 1) && packetCounter >= 0) {
    statistics.errorCounts[GBTS::ErrPacketCounterJump]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrPacketCounterJump]
               << " : jump from " << int(packetCounter) << " to " << int(rdh->packetCounter);
    errorBits |= 0x1 << int(GBTS::ErrPacketCounterJump);
    return Warning;
  }
  packetCounter = rdh->packetCounter;
  return NoError;
}

///_________________________________________________________________
/// Check RDH Stop correctness
GBTLink::ErrorType GBTLink::checkErrorsRDHStop(const RDH* rdh)
{
  if (lastRDH && RDHUtils::getHeartBeatOrbit(*lastRDH) != RDHUtils::getHeartBeatOrbit(*rdh) // new HB starts
      && !lastRDH->stop) {
    statistics.errorCounts[GBTS::ErrPageNotStopped]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrPageNotStopped];
    RDHUtils::printRDH(*lastRDH);
    RDHUtils::printRDH(*rdh);
    errorBits |= 0x1 << int(GBTS::ErrPageNotStopped);
    return Warning;
  }
  return NoError;
}

///_________________________________________________________________
/// Check if the RDH Stop page is empty
GBTLink::ErrorType GBTLink::checkErrorsRDHStopPageEmpty(const RDH* rdh)
{
  if (rdh->stop && rdh->memorySize != sizeof(RDH)) {
    statistics.errorCounts[GBTS::ErrStopPageNotEmpty]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrStopPageNotEmpty];
    errorBits |= 0x1 << int(GBTS::ErrStopPageNotEmpty);
    RDHUtils::printRDH(*rdh);
    return Warning;
  }
  return NoError;
}

///_________________________________________________________________
/// Check the GBT Trigger word correctness
GBTLink::ErrorType GBTLink::checkErrorsTriggerWord(const GBTTrigger* gbtTrg)
{
  if (!gbtTrg->isTriggerWord()) { // check trigger word
    gbtTrg->printX();
    statistics.errorCounts[GBTS::ErrMissingGBTTrigger]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrMissingGBTTrigger];
    errorBits |= 0x1 << int(GBTS::ErrMissingGBTTrigger);
    return Abort;
  }
  return NoError;
}

///_________________________________________________________________
/// Check the GBT Header word correctness
GBTLink::ErrorType GBTLink::checkErrorsHeaderWord(const GBTDataHeader* gbtH)
{
  if (!gbtH->isDataHeader()) { // check header word
    statistics.errorCounts[GBTS::ErrMissingGBTHeader]++;
    gbtH->printX();
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrMissingGBTHeader];
    errorBits |= 0x1 << int(GBTS::ErrMissingGBTHeader);
    return Abort;
  }
  /* RSTODO: this makes sense only for old format, where every trigger has its RDH
  if (gbtH->packetIdx != lastRDH->pageCnt) {
    statistics.errorCounts[GBTS::ErrRDHvsGBTHPageCnt]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrRDHvsGBTHPageCnt] << ": diff in GBT header "
	       << gbtH->packetIdx << " and RDH page " << lastRDH->pageCnt << " counters";
    errorBits |= 0x1<<int(GBTS::ErrRDHvsGBTHPageCnt);
    return Warning;
  }
  */
  // RSTODO CHECK
  if (lanesActive == lanesStop) { // all lanes received their stop, new page 0 expected
    //if (lastRDH->pageCnt) { // makes sens for old format only
    if (gbtH->packetIdx) {
      statistics.errorCounts[GBTS::ErrNonZeroPageAfterStop]++;
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrNonZeroPageAfterStop]
                 << ": Non-0 page counter (" << lastRDH->pageCnt << ") while all lanes were stopped";
      errorBits |= 0x1 << int(GBTS::ErrNonZeroPageAfterStop);
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
    statistics.errorCounts[GBTS::ErrInvalidActiveLanes]++;
    std::bitset<32> expectL(cbl), gotL(lanesActive);
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrInvalidActiveLanes]
               << gotL << " vs " << expectL << " skip page";
    errorBits |= 0x1 << int(GBTS::ErrInvalidActiveLanes);
    return Warning;
  }
  return NoError;
}

///_________________________________________________________________
/// Check GBT Data word
GBTLink::ErrorType GBTLink::checkErrorsGBTData(int cableHW, int cableSW)
{
  lanesWithData |= 0x1 << cableSW;    // flag that the data was seen on this lane
  if (lanesStop & (0x1 << cableSW)) { // make sure stopped lanes do not transmit the data
    statistics.errorCounts[GBTS::ErrDataForStoppedLane]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrDataForStoppedLane]
               << cableHW << " (sw:" << cableSW << ")";
    errorBits |= 0x1 << int(GBTS::ErrDataForStoppedLane);
    return Warning;
  }

  return NoError;
}

///_________________________________________________________________
/// Check the GBT Trailer word correctness
GBTLink::ErrorType GBTLink::checkErrorsTrailerWord(const GBTDataTrailer* gbtT)
{
  if (!gbtT->isDataTrailer()) {
    gbtT->printX();
    statistics.errorCounts[GBTS::ErrMissingGBTTrailer]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrMissingGBTTrailer];
    errorBits |= 0x1 << int(GBTS::ErrMissingGBTTrailer);
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
    statistics.errorCounts[GBTS::ErrPacketDoneMissing]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrPacketDoneMissing];
    errorBits |= 0x1 << int(GBTS::ErrPacketDoneMissing);
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
    if (lastRDH->triggerType != o2::trigger::SOT) { // only SOT trigger allows unstopped lanes?
      statistics.errorCounts[GBTS::ErrUnstoppedLanes]++;
      std::bitset<32> active(lanesActive), stopped(lanesStop);
      LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrUnstoppedLanes]
                 << " | active: " << active << " stopped: " << stopped;
      errorBits |= 0x1 << int(GBTS::ErrUnstoppedLanes);
    }
    err = Warning;
  }
  // make sure all active lanes (except those in time-out) have sent some data
  if ((~lanesWithData & lanesActive) != lanesTimeOut) {
    std::bitset<32> withData(lanesWithData), active(lanesActive), timeOut(lanesTimeOut);
    statistics.errorCounts[GBTS::ErrNoDataForActiveLane]++;
    LOG(ERROR) << describe() << ' ' << statistics.ErrNames[GBTS::ErrNoDataForActiveLane]
               << " | with data: " << withData << " active: " << active << " timeOut: " << timeOut;
    errorBits |= 0x1 << int(GBTS::ErrNoDataForActiveLane);
    err = Warning;
  }
  return err;
}

#endif
