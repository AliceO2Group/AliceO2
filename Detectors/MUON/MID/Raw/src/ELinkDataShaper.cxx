// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/ELinkDataShaper.cxx
/// \brief  Properly formats and sets the absolute timestamp of the raw data
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 March 2021

#include "MIDRaw/ELinkDataShaper.h"

#include "CommonConstants/LHCConstants.h"

namespace o2
{
namespace mid
{

ELinkDataShaper::ELinkDataShaper(bool isDebugMode, bool isLoc, uint8_t uniqueId) : mUniqueId(uniqueId)
{
  /// Ctr
  if (isDebugMode) {
    if (isLoc) {
      mOnDone = &ELinkDataShaper::onDoneLocDebug;
    } else {
      mOnDone = &ELinkDataShaper::onDoneRegDebug;
    }
  } else {
    if (isLoc) {
      mOnDone = &ELinkDataShaper::onDoneLoc;
    } else {
      mOnDone = &ELinkDataShaper::onDoneReg;
    }
  }
}

void ELinkDataShaper::set(uint32_t orbit)
{
  /// Sets the orbit and the output data vectors
  mRDHOrbit = orbit;

  if (mIR.isDummy()) {
    mIR.bc = 0;
    // The reset changes depending on the way we synch with the orbit
    // (see processOrbitTrigger for details)
    // FIXME: pick one of the two
    // mIR.orbit = orbit - 1; // with orbit increase
    mIR.orbit = orbit; // with reset to RDH
    mLastClock = constants::lhc::LHCMaxBunches;
  }
}

bool ELinkDataShaper::checkLoc(const ELinkDecoder& decoder)
{
  /// Performs checks on the local board
  return (decoder.getId() == (mUniqueId & 0xF));
}

EventType ELinkDataShaper::processSelfTriggered(uint16_t localClock, uint16_t& correctedClock)
{
  /// Processes the self-triggered event
  correctedClock = localClock - mElectronicsDelay.BCToLocal;
  if (mReceivedCalibration && (localClock == mExpectedFETClock)) {
    // Reset the calibration flag for this e-link
    mReceivedCalibration = false;
    return EventType::Dead;
  }
  return EventType::Standard;
}

EventType ELinkDataShaper::processCalibrationTrigger(uint16_t localClock)
{
  /// Processes the calibration event
  mExpectedFETClock = localClock + mElectronicsDelay.calibToFET;
  mReceivedCalibration = true;
  return EventType::Noise;
}

void ELinkDataShaper::processOrbitTrigger(uint16_t localClock, uint8_t triggerWord)
{
  /// Processes the orbit trigger event

  // The local clock is reset: we are now in synch with the new HB
  // We have two ways to account for the orbit change:
  // - increase the orbit counter by 1 for this e-link
  //   (CAVEAT: synch is lost if we lose some orbit)
  // - set the orbit to the one found in RDH
  //   (CAVEAT: synch is lost if we have lot of data, spanning over two orbits)
  // FIXME: pick one of the two
  // ++mIR.orbit; // orbit increase
  mIR.orbit = mRDHOrbit; // reset to RDH
  if ((triggerWord & raw::sSOX) == 0) {
    mLastClock = localClock;
  }
  // The orbit trigger resets the clock.
  // If we received a calibration trigger, we need to change the value of the expected clock accordingly
  if (mReceivedCalibration) {
    mExpectedFETClock -= (localClock + 1);
  }
}

bool ELinkDataShaper::processTrigger(const ELinkDecoder& decoder, EventType& eventType, uint16_t& correctedClock)
{
  /// Processes the trigger information
  /// Returns true if the event should be further processed,
  /// returns false otherwise.
  auto localClock = decoder.getCounter();

  if (decoder.getTriggerWord() == 0) {
    // This is a self-triggered event
    eventType = processSelfTriggered(localClock, correctedClock);
    return true;
  }

  // From here we treat triggered events
  bool goOn = false;
  correctedClock = localClock;
  if (decoder.getTriggerWord() & raw::sCALIBRATE) {
    // This is an answer to a calibration trigger
    eventType = processCalibrationTrigger(localClock);
    goOn = true;
  }

  if (decoder.getTriggerWord() & raw::sORB) {
    // This is the answer to an orbit trigger
    processOrbitTrigger(localClock, decoder.getTriggerWord());
    eventType = EventType::Standard;
  }

  return goOn;
}

void ELinkDataShaper::addLoc(const ELinkDecoder& decoder, EventType eventType, uint16_t correctedClock, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// Adds the local board to the output data vector
  auto firstEntry = data.size();
  data.push_back({decoder.getStatusWord(), decoder.getTriggerWord(), mUniqueId, decoder.getInputs()});
  InteractionRecord intRec(mIR.bc + correctedClock, mIR.orbit);
  rofs.emplace_back(intRec, eventType, firstEntry, 1);
  for (int ich = 0; ich < 4; ++ich) {
    if ((data.back().firedChambers & (1 << ich))) {
      data.back().patternsBP[ich] = decoder.getPattern(0, ich);
      data.back().patternsNBP[ich] = decoder.getPattern(1, ich);
    }
  }
}

void ELinkDataShaper::onDoneLoc(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// Performs action on decoded local board
  EventType eventType;
  uint16_t correctedClock;
  if (processTrigger(decoder, eventType, correctedClock) && checkLoc(decoder)) {
    addLoc(decoder, eventType, correctedClock, data, rofs);
  }
}

void ELinkDataShaper::onDoneLocDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// This always adds the local board to the output, without performing tests
  EventType eventType;
  uint16_t correctedClock;
  processTrigger(decoder, eventType, correctedClock);
  addLoc(decoder, eventType, correctedClock, data, rofs);
  if (decoder.getTriggerWord() & raw::sORB) {
    // The local clock is increased when receiving an orbit trigger,
    // but the local counter returned in answering the trigger
    // belongs to the previous orbit
    --rofs.back().interactionRecord.orbit;
  }
}

void ELinkDataShaper::onDoneRegDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// Performs action on decoded regional board in debug mode.
  EventType eventType;
  uint16_t correctedClock;
  processTrigger(decoder, eventType, correctedClock);
  // If we want to distinguish the two regional e-links, we can use the link ID instead
  auto firstEntry = data.size();
  data.push_back({decoder.getStatusWord(), decoder.getTriggerWord(), mUniqueId, decoder.getInputs()});

  auto orbit = (decoder.getTriggerWord() & raw::sORB) ? mIR.orbit - 1 : mIR.orbit;

  InteractionRecord intRec(mIR.bc + correctedClock, orbit);
  if (decoder.getTriggerWord() == 0) {
    if (intRec.bc < mElectronicsDelay.regToLocal) {
      // In the tests, the HB does not really correspond to a change of orbit
      // So we need to keep track of the last clock at which the HB was received
      // and come back to that value
      // FIXME: Remove this part as well as mLastClock when tests are no more needed
      intRec -= (constants::lhc::LHCMaxBunches - mLastClock - 1);
    }
    // This is a self-triggered event.
    // In this case the regional card needs to wait to receive the tracklet decision of each local
    // which result in a delay that needs to be subtracted if we want to be able to synchronize
    // local and regional cards for the checks
    intRec -= mElectronicsDelay.regToLocal;
  }
  rofs.emplace_back(intRec, eventType, firstEntry, 1);
}

} // namespace mid
} // namespace o2
