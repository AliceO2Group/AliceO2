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

#include <FOCALBase/EventReader.h>

using namespace o2::focal;

EventReader::EventReader(TTree* eventTree)
{
  init(eventTree);
}

void EventReader::init(TTree* eventTree)
{
  mTreeReader = std::make_unique<TTreeReader>(eventTree);
  mPadBranch = std::make_unique<TTreeReaderValue<std::vector<PadLayerEvent>>>(*mTreeReader, "FOCALPadLayer");
  mPixelChipBranch = std::make_unique<TTreeReaderValue<std::vector<PixelChipRecord>>>(*mTreeReader, "FOCALPixelChip");
  mPixelHitBranch = std::make_unique<TTreeReaderValue<std::vector<PixelHit>>>(*mTreeReader, "FOCALPixelHit");
  mTriggerBranch = std::make_unique<TTreeReaderValue<std::vector<TriggerRecord>>>(*mTreeReader, "FOCALTrigger");

  mHasEntryLoaded = false;
  mTreeReaderHasNext = true;
  mEntryInTF = 0;
}

void EventReader::nextTimeframe()
{
  mTreeReaderHasNext = mTreeReader->Next();
  mHasEntryLoaded = true;
  mEntryInTF = 0;
}

bool EventReader::hasNext() const
{
  if (mHasEntryLoaded) {
    if (mEntryInTF + 1 < (*mTriggerBranch)->size()) {
      // more enties in current timeframe
      return true;
    } else {
      // check whether we have another timeframe to read
      return mTreeReaderHasNext;
    }
  } else {
    return mTreeReaderHasNext;
  }
}

Event EventReader::readNextEvent()
{
  if (!mHasEntryLoaded) {
    nextTimeframe();
  }
  // 2 conditions
  // - end of timeframe
  // - skip empty timeframes
  while (mEntryInTF == (*mTriggerBranch)->size() && mTreeReaderHasNext) {
    nextTimeframe();
  }

  Event nextevent;
  if (mEntryInTF < (*mTriggerBranch)->size()) { // Additional protection if we are closing with empty timeframe
    auto& triggerrecord = (*mTriggerBranch)->at(mEntryInTF);
    gsl::span<const PadLayerEvent> eventPadData;
    if (triggerrecord.getNumberOfPadObjects()) {
      eventPadData = gsl::span<const PadLayerEvent>((*mPadBranch)->data() + triggerrecord.getFirstPadEntry(), triggerrecord.getNumberOfPadObjects());
    }
    gsl::span<const PixelHit> eventPixelHits;
    if (triggerrecord.getNumberOfPixelHitObjects()) {
      eventPixelHits = gsl::span<const PixelHit>((*mPixelHitBranch)->data() + triggerrecord.getFirstPixelHitEntry(), triggerrecord.getNumberOfPixelHitObjects());
    }
    gsl::span<const PixelChipRecord> eventPixelChip;
    if (triggerrecord.getNumberOfPixelChipObjects()) {
      eventPixelChip = gsl::span<const PixelChipRecord>((*mPixelChipBranch)->data() + triggerrecord.getFirstPixelChipEntry(), triggerrecord.getNumberOfPixelChipObjects());
    }

    nextevent.construct(triggerrecord.getBCData(), eventPadData, eventPixelChip, eventPixelHits);
  }
  mEntryInTF++;
  return nextevent;
}