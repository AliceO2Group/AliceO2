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

#include "RawInfo.h"

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
      ir.orbit = orbit;
    }
    for (auto& clock : mLastClock) {
      clock = constants::lhc::LHCMaxBunches;
    }
    for (auto& clock : mCalibClocks) {
      clock = constants::lhc::LHCMaxBunches;
    }
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

void GBTOutputHandler::addBoard(size_t ilink, const ELinkDecoder& decoder)
{
  /// Adds the local or regional board to the output data vector
  uint16_t localClock = decoder.getCounter();
  EventType eventType = EventType::Standard;
  if (decoder.getTriggerWord() & raw::sCALIBRATE) {
    mCalibClocks[ilink] = localClock;
    eventType = EventType::Noise;
  } else if (localClock == mCalibClocks[ilink] + sDelayCalibToFET) {
    eventType = EventType::Dead;
  }
  auto firstEntry = mData.size();
  mData.push_back({decoder.getStatusWord(), decoder.getTriggerWord(), crateparams::makeUniqueLocID(crateparams::getCrateIdFromROId(mFeeId), decoder.getId()), decoder.getInputs()});
  InteractionRecord intRec(mIRs[ilink].bc + localClock - sDelayBCToLocal, mIRs[ilink].orbit);
  mROFRecords.emplace_back(intRec, eventType, firstEntry, 1);
}

void GBTOutputHandler::addLoc(size_t ilink, const ELinkDecoder& decoder)
{
  /// Adds the local board to the output data vector
  addBoard(ilink, decoder);
  for (int ich = 0; ich < 4; ++ich) {
    if ((mData.back().firedChambers & (1 << ich))) {
      mData.back().patternsBP[ich] = decoder.getPattern(0, ich);
      mData.back().patternsNBP[ich] = decoder.getPattern(1, ich);
    }
  }
  if (mROFRecords.back().eventType == EventType::Dead) {
    if (invertPattern(mData.back())) {
      mData.pop_back();
      mROFRecords.pop_back();
    }
  }
}

bool GBTOutputHandler::updateIR(size_t ilink, const ELinkDecoder& decoder)
{
  /// Updates the interaction record for the link
  if (decoder.getTriggerWord() & raw::sORB) {
    // This is the answer to an orbit trigger
    // The local clock is reset: we are now in synch with the new HB
    mIRs[ilink] = mIRFirstPage;
    if (!(decoder.getTriggerWord() & (raw::sSOX | raw::sEOX))) {
      mLastClock[ilink] = decoder.getCounter();
    }
    return true;
  }
  return false;
}

void GBTOutputHandler::onDoneLoc(size_t ilink, const ELinkDecoder& decoder)
{
  /// Performs action on decoded local board
  if (updateIR(ilink, decoder)) {
    return;
  }
  if (checkLoc(ilink, decoder)) {
    addLoc(ilink, decoder);
  }
}

void GBTOutputHandler::onDoneLocDebug(size_t ilink, const ELinkDecoder& decoder)
{

  /// This always adds the local board to the output, without performing tests
  updateIR(ilink, decoder);
  addLoc(ilink, decoder);
}

void GBTOutputHandler::onDoneRegDebug(size_t ilink, const ELinkDecoder& decoder)
{
  /// Performs action on decoded regional board in debug mode.
  updateIR(ilink, decoder);
  addBoard(ilink, decoder);
  // The board creation is optimized for the local boards, not the regional
  // (which are transmitted only in debug mode).
  // So, at this point, for the regional board, the local Id is actually the crate ID.
  // If we want to distinguish the two regional e-links, we can use the link ID instead
  mData.back().boardId = crateparams::makeUniqueLocID(crateparams::getCrateIdFromROId(mFeeId), ilink + 8 * (crateparams::getGBTIdInCrate(mFeeId) - 1));
  if (mData.back().triggerWord == 0) {
    if (mROFRecords.back().interactionRecord.bc < sDelayRegToLocal) {
      // In the tests, the HB does not really correspond to a change of orbit
      // So we need to keep track of the last clock at which the HB was received
      // and come back to that value
      // FIXME: Remove this part as well as mLastClock when tests are no more needed
      mROFRecords.back().interactionRecord -= (constants::lhc::LHCMaxBunches - mLastClock[ilink] - 1);
    }
    // This is a self-triggered event.
    // In this case the regional card needs to wait to receive the tracklet decision of each local
    // which result in a delay that needs to be subtracted if we want to be able to synchronize
    // local and regional cards for the checks
    mROFRecords.back().interactionRecord -= sDelayRegToLocal;
  }
}

bool GBTOutputHandler::invertPattern(LocalBoardRO& loc)
{
  /// Gets the proper pattern
  for (int ich = 0; ich < 4; ++ich) {
    loc.patternsBP[ich] = ~loc.patternsBP[ich];
    loc.patternsNBP[ich] = ~loc.patternsNBP[ich];
    if (loc.patternsBP[ich] == 0 && loc.patternsNBP[ich] == 0) {
      loc.firedChambers &= ~(1 << ich);
    }
  }
  return (loc.firedChambers == 0);
}

} // namespace mid
} // namespace o2
