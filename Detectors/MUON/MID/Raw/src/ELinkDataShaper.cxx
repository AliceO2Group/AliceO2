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

ELinkDataShaper::ELinkDataShaper(bool isDebugMode, bool isLoc, uint8_t uniqueId, const ElectronicsDelay& electronicsDelay) : mUniqueId(uniqueId), mElectronicsDelay(electronicsDelay)
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
  if (!isLoc) {
    mElectronicsDelay.BCToLocal += electronicsDelay.regToLocal;
    mElectronicsDelay.calibToFET += electronicsDelay.regToLocal;
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

EventType ELinkDataShaper::processSelfTriggered(uint16_t localClock, InteractionRecord& ir)
{
  /// Processes the self-triggered event

  // This is a self-triggered events.
  // The physics data arrives with a delay compared to the BC,
  // which is due to the travel time of muons up to the MID chambers
  // plus the travel time of the signal to the readout electronics.
  // In the case of regional cards, a further delay is expected
  // since this card needs to wait for the tracklet decision of each local card.
  // For simplicity, this delay is added to the BCToLocal in the constructor.
  // In both cases, we need to correct for the delay in order to go back to the real BC.
  if (ir.bc < mElectronicsDelay.BCToLocal) {
    // If the bc is smaller than the delay, it means that the local clock was reset
    // This events therefore belongs to the previous orbit.
    // We therefore add the value of the last BC (+1 to account for the reset)
    // and we decrease the orbit by 1.
    ir.bc += mLastClock + 1;
    --ir.orbit;
  }
  // We can now safely subtract the delay, which, thanks to the above protection,
  // will not result in a negative number
  ir.bc -= mElectronicsDelay.BCToLocal;
  if (mReceivedCalibration && (localClock == mExpectedFETClock)) {
    // Reset the calibration flag for this e-link
    mReceivedCalibration = false;
    return EventType::FET;
  }
  return EventType::Standard;
}

EventType ELinkDataShaper::processCalibrationTrigger(uint16_t localClock)
{
  /// Processes the calibration event
  mExpectedFETClock = localClock + mElectronicsDelay.calibToFET;
  mReceivedCalibration = true;
  return EventType::Calib;
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

bool ELinkDataShaper::processTrigger(const ELinkDecoder& decoder, EventType& eventType, InteractionRecord& ir)
{
  /// Processes the trigger information
  /// Returns true if the event should be further processed,
  /// returns false otherwise.
  auto localClock = decoder.getCounter();

  // FIXME: So far the bc information is not used
  // since it seems that it is always equal to 0
  // (at least in the RDH of the first page, which is what we need).
  // In this case the local clock and the BC coincide.
  // If this is not the case, the bc of the orbit trigger should be correctly taken into account
  ir.bc = localClock;
  // ir.bc = mIR.bc + localClock;
  ir.orbit = mIR.orbit;

  if (decoder.getTriggerWord() == 0) {
    // This is a self-triggered event
    eventType = processSelfTriggered(localClock, ir);
    return true;
  }

  // From here we treat triggered events
  bool goOn = false;
  eventType = EventType::Standard;
  if (decoder.getTriggerWord() & raw::sCALIBRATE) {
    // This is an answer to a calibration trigger
    eventType = processCalibrationTrigger(localClock);
    goOn = true;
  }

  if (decoder.getTriggerWord() & raw::sORB) {
    // This is the answer to an orbit trigger
    processOrbitTrigger(localClock, decoder.getTriggerWord());
  }

  return goOn;
}

void ELinkDataShaper::addLoc(const ELinkDecoder& decoder, EventType eventType, InteractionRecord ir, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// Adds the local board to the output data vector
  auto firstEntry = data.size();
  data.push_back({decoder.getStatusWord(), decoder.getTriggerWord(), mUniqueId, decoder.getInputs()});
  rofs.emplace_back(ir, eventType, firstEntry, 1);
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
  InteractionRecord ir;
  if (processTrigger(decoder, eventType, ir) && checkLoc(decoder)) {
    addLoc(decoder, eventType, ir, data, rofs);
  }
}

void ELinkDataShaper::onDoneLocDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// This always adds the local board to the output, without performing tests
  EventType eventType;
  InteractionRecord ir;
  processTrigger(decoder, eventType, ir);
  addLoc(decoder, eventType, ir, data, rofs);
}

void ELinkDataShaper::onDoneRegDebug(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  /// Performs action on decoded regional board in debug mode.
  EventType eventType;
  InteractionRecord ir;
  processTrigger(decoder, eventType, ir);
  // If we want to distinguish the two regional e-links, we can use the link ID instead
  auto firstEntry = data.size();
  data.push_back({decoder.getStatusWord(), decoder.getTriggerWord(), mUniqueId, decoder.getInputs()});
  rofs.emplace_back(ir, eventType, firstEntry, 1);
}

} // namespace mid
} // namespace o2
