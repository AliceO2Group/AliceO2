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
#include "CommonConstants/Triggers.h"

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
  mLocalToBCSelfTrig = mElectronicsDelay.localToBC;
  mElectronicsDelay.calibToFET += mElectronicsDelay.localToBC;
  if (!isLoc) {
    mLocalToBCSelfTrig -= electronicsDelay.localToReg;
  }
}

void ELinkDataShaper::set(uint32_t orbit, uint32_t trigger)
{
  /// Sets the orbit and the output data vectors

  if (mIR.isDummy()) {
    // First initialization
    mIR.bc = 0;
    mIR.orbit = orbit - 1;
    mMaxBunches = constants::lhc::LHCMaxBunches;
  } else if ((trigger & o2::trigger::TF || trigger & o2::trigger::SOT) && mRDHOrbit != orbit) {
    // At TF limit, the CRU UL ensures that the first event is the answer of the electronics to a HB trigger,
    // so it is the right time to synch with the orbit in the RDH.
    // This also allows to consistently run on EPNs, since EPN receive packets of TFs
    // and the orbit is not consecutive from one packet to the other.
    // Notice that we subtract 1 to the orbit value, since the TF starts with the answer to a HB trigger,
    // and the value will be therefore correctly incremented.
    // Notice also that we might receive several RDH pages
    // (at least two since there is always an empty stop page),
    // but we want to set the orbit only for the first one.
    // The easiest way to do it is to check that the current orbit differs from the last saved RDH orbit.
    mIR.orbit = orbit - 1;
  }
  mRDHOrbit = orbit;
}

bool ELinkDataShaper::checkLoc(const ELinkDecoder& decoder)
{
  /// Performs checks on the local board
  return (decoder.getId() == (mUniqueId & 0xF));
}

EventType ELinkDataShaper::processSelfTriggered(InteractionRecord& ir)
{
  /// Processes the self-triggered event

  // This is a self-triggered event.
  // The physics data arrives with a delay compared to the BC,
  // which is due to the travel time of muons up to the MID chambers
  // plus the travel time of the signal to the readout electronics.
  // In the case of regional cards, a further delay is expected
  // since this card needs to wait for the tracklet decision of each local card.
  // For simplicity, this delay is added to the localToBC in the constructor.
  // In both cases, we need to correct for the delay in order to go back to the real BC.
  applyElectronicsDelay(ir.orbit, ir.bc, mLocalToBCSelfTrig, mMaxBunches);
  if (ir == mExpectedFET) {
    return EventType::FET;
  }
  return EventType::Standard;
}

EventType ELinkDataShaper::processCalibrationTrigger(const InteractionRecord& ir)
{
  /// Processes the calibration event
  mExpectedFET = ir;
  applyElectronicsDelay(mExpectedFET.orbit, mExpectedFET.bc, mElectronicsDelay.calibToFET, mMaxBunches);
  return EventType::Calib;
}

void ELinkDataShaper::processHBTrigger(uint16_t localClock, uint8_t triggerWord)
{
  /// Processes the HB trigger event

  // The local clock is reset: we are now in synch with the new HB
  ++mIR.orbit;
  // The local clock value in an answer to the orbit trigger corresponds to the number of clocks elapsed since last reset.
  // Since the HB trigger is reset at each orbit, this corresponds to the number of bunches.
  // We need to add one because the local clock starts from 0.
  mMaxBunches = localClock + 1;
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
    eventType = processSelfTriggered(ir);
    return true;
  }

  // From here we treat triggered events
  bool goOn = false;
  eventType = EventType::Standard;

  if (decoder.getTriggerWord() & raw::sORB) {
    // This is the answer to an HB trigger
    processHBTrigger(localClock, decoder.getTriggerWord());
  }

  if (decoder.getTriggerWord() & raw::sCALIBRATE) {
    // This is an answer to a calibration trigger
    eventType = processCalibrationTrigger(ir);
    goOn = true;
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
