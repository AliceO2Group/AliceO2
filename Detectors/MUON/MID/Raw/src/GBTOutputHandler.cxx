// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/GBTOutputHandler.cxx
/// \brief  MID GBT decoder output handler
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 April 2020

#include "MIDRaw/GBTOutputHandler.h"

namespace o2
{
namespace mid
{

void GBTOutputHandler::clear()
{
  /// Rewind bytes
  mData.clear();
  mROFRecords.clear();
}

void GBTOutputHandler::setIR(uint16_t bc, uint32_t orbit, int pageCnt)
{
  /// Sets the interaction record if needed
  if (mIRs[0].isDummy()) {
    for (auto& ir : mIRs) {
      ir.bc = bc;
      // The reset changes depending on the way we synch with the orbit
      // (see processOrbitTrigger for details)
      // FIXME: pick one of the two
      ir.orbit = orbit - 1; // with orbit increase
      // ir.orbit = orbit; // with reset to RDH
    }
    mLastClock.fill(constants::lhc::LHCMaxBunches);
  }

  if (pageCnt == 0) {
    // FIXME: in the tests, the BC counter increases at each RDH
    // However, the inner clock of the RO is not reset,
    // so if we want to have the absolute BC,
    // we need to store only the BC counter of the first page.
    // Not sure how it will work on data...
    mIRFirstPage.bc = bc;
    mIRFirstPage.orbit = orbit;
  }
}

bool GBTOutputHandler::checkLoc(size_t ilink, const ELinkDecoder& decoder)
{
  /// Performs checks on the local board
  return (ilink == decoder.getId() % 8);
}

EventType GBTOutputHandler::processSelfTriggered(size_t ilink, uint16_t localClock, uint16_t& correctedClock)
{
  /// Processes the self-triggered event
  correctedClock = localClock - mElectronicsDelay.BCToLocal;
  uint16_t linkMask = 1 << ilink;
  if ((mReceivedCalibration & linkMask) && (localClock == mExpectedFETClock[ilink])) {
    // Reset the calibration flag for this e-link
    mReceivedCalibration &= ~linkMask;
    return EventType::Dead;
  }
  return EventType::Standard;
}

EventType GBTOutputHandler::processCalibrationTrigger(size_t ilink, uint16_t localClock)
{
  /// Processes the calibration event
  mExpectedFETClock[ilink] = localClock + mElectronicsDelay.calibToFET;
  mReceivedCalibration |= (1 << ilink);
  return EventType::Noise;
}

void GBTOutputHandler::processOrbitTrigger(size_t ilink, uint16_t localClock, uint8_t triggerWord)
{
  /// Processes the orbit trigger event

  // The local clock is reset: we are now in synch with the new HB
  // We have to way to account for the orbit change:
  // - increase the orbit counter by 1 for this e-link
  //   (CAVEAT: synch is lost if we lose some orbit)
  // - set the orbit to the one found in RDH
  //   (CAVEAT: synch is lost if we have lot of data, spanning over two orbits)
  // FIXME: pick one of the two
  ++mIRs[ilink].orbit; // orbit increase
  // mIRs[ilink] = mIRFirstPage; // reset to RDH
  if ((triggerWord & raw::sSOX) == 0) {
    mLastClock[ilink] = localClock;
  }
  // The orbit trigger resets the clock.
  // If we received a calibration trigger, we need to change the value of the expected clock accordingly
  if (mReceivedCalibration & (1 << ilink)) {
    mExpectedFETClock[ilink] -= (localClock + 1);
  }
}

bool GBTOutputHandler::processTrigger(size_t ilink, const ELinkDecoder& decoder, EventType& eventType, uint16_t& correctedClock)
{
  /// Processes the trigger information
  /// Returns true if the event should be further processed,
  /// returns false otherwise.
  uint16_t linkMask = 1 << ilink;
  auto localClock = decoder.getCounter();

  if (decoder.getTriggerWord() == 0) {
    // This is a self-triggered event
    eventType = processSelfTriggered(ilink, localClock, correctedClock);
    return true;
  }

  // From here we treat triggered events
  bool goOn = false;
  correctedClock = localClock;
  if (decoder.getTriggerWord() & raw::sCALIBRATE) {
    // This is an answer to a calibration trigger
    eventType = processCalibrationTrigger(ilink, localClock);
    goOn = true;
  }

  if (decoder.getTriggerWord() & raw::sORB) {
    // This is the answer to an orbit trigger
    processOrbitTrigger(ilink, localClock, decoder.getTriggerWord());
    eventType = EventType::Standard;
  }

  return goOn;
}

void GBTOutputHandler::addLoc(size_t ilink, const ELinkDecoder& decoder, EventType eventType, uint16_t correctedClock)
{
  /// Adds the local board to the output data vector
  auto firstEntry = mData.size();
  mData.push_back({decoder.getStatusWord(), decoder.getTriggerWord(), crateparams::makeUniqueLocID(crateparams::getCrateIdFromROId(mFeeId), decoder.getId()), decoder.getInputs()});
  InteractionRecord intRec(mIRs[ilink].bc + correctedClock, mIRs[ilink].orbit);
  mROFRecords.emplace_back(intRec, eventType, firstEntry, 1);
  for (int ich = 0; ich < 4; ++ich) {
    if ((mData.back().firedChambers & (1 << ich))) {
      mData.back().patternsBP[ich] = decoder.getPattern(0, ich);
      mData.back().patternsNBP[ich] = decoder.getPattern(1, ich);
    }
  }
}

void GBTOutputHandler::onDoneLoc(size_t ilink, const ELinkDecoder& decoder)
{
  /// Performs action on decoded local board
  EventType eventType;
  uint16_t correctedClock;
  if (processTrigger(ilink, decoder, eventType, correctedClock) && checkLoc(ilink, decoder)) {
    addLoc(ilink, decoder, eventType, correctedClock);
  }
}

void GBTOutputHandler::onDoneLocDebug(size_t ilink, const ELinkDecoder& decoder)
{
  /// This always adds the local board to the output, without performing tests
  EventType eventType;
  uint16_t correctedClock;
  processTrigger(ilink, decoder, eventType, correctedClock);
  addLoc(ilink, decoder, eventType, correctedClock);
  if (decoder.getTriggerWord() & raw::sORB) {
    // The local clock is increased when receiving an orbit trigger,
    // but the local counter returned in answering the trigger
    // belongs to the previous orbit
    --mROFRecords.back().interactionRecord.orbit;
  }
}

void GBTOutputHandler::onDoneRegDebug(size_t ilink, const ELinkDecoder& decoder)
{
  /// Performs action on decoded regional board in debug mode.
  EventType eventType;
  uint16_t correctedClock;
  processTrigger(ilink, decoder, eventType, correctedClock);
  // If we want to distinguish the two regional e-links, we can use the link ID instead
  auto firstEntry = mData.size();
  mData.push_back({decoder.getStatusWord(), decoder.getTriggerWord(), crateparams::makeUniqueLocID(crateparams::getCrateIdFromROId(mFeeId), ilink + 8 * (crateparams::getGBTIdInCrate(mFeeId) - 1)), decoder.getInputs()});

  auto orbit = (decoder.getTriggerWord() & raw::sORB) ? mIRs[ilink].orbit - 1 : mIRs[ilink].orbit;

  InteractionRecord intRec(mIRs[ilink].bc + correctedClock, orbit);
  if (decoder.getTriggerWord() == 0) {
    if (intRec.bc < mElectronicsDelay.regToLocal) {
      // In the tests, the HB does not really correspond to a change of orbit
      // So we need to keep track of the last clock at which the HB was received
      // and come back to that value
      // FIXME: Remove this part as well as mLastClock when tests are no more needed
      intRec -= (constants::lhc::LHCMaxBunches - mLastClock[ilink] - 1);
    }
    // This is a self-triggered event.
    // In this case the regional card needs to wait to receive the tracklet decision of each local
    // which result in a delay that needs to be subtracted if we want to be able to synchronize
    // local and regional cards for the checks
    intRec -= mElectronicsDelay.regToLocal;
  }
  mROFRecords.emplace_back(intRec, eventType, firstEntry, 1);
}

} // namespace mid
} // namespace o2
