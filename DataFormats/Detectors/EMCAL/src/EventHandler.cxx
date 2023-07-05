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

#include "DataFormatsEMCAL/EventHandler.h"
#include <optional>

using namespace o2::emcal;

template <class CellInputType>
EventHandler<CellInputType>::EventHandler(CellRange cells, TriggerRange triggers) : mTriggerRecordsCells(triggers),
                                                                                    mCells(cells)
{
}

template <class CellInputType>
EventHandler<CellInputType>::EventHandler(ClusterRange clusters, CellIndexRange cellIndices, TriggerRange triggersCluster, TriggerRange triggersCells) : mTriggerRecordsClusters(triggersCluster),
                                                                                                                                                         mTriggerRecordsCellIndices(triggersCells),
                                                                                                                                                         mClusters(clusters),
                                                                                                                                                         mClusterCellIndices(cellIndices)
{
}

template <class CellInputType>
EventHandler<CellInputType>::EventHandler(ClusterRange clusters, CellIndexRange cellIndices, CellRange cells, TriggerRange triggersCluster, TriggerRange triggersCellIndex, TriggerRange triggersCell) : mTriggerRecordsClusters(triggersCluster),
                                                                                                                                                                                                         mTriggerRecordsCellIndices(triggersCellIndex),
                                                                                                                                                                                                         mTriggerRecordsCells(triggersCell),
                                                                                                                                                                                                         mClusters(clusters),
                                                                                                                                                                                                         mClusterCellIndices(cellIndices),
                                                                                                                                                                                                         mCells(cells)
{
}

template <class CellInputType>
int EventHandler<CellInputType>::getNumberOfEvents() const
{
  int neventsClusters = mTriggerRecordsClusters.size(),
      neventsCells = mTriggerRecordsCells.size();
  if (neventsClusters) {
    return neventsClusters;
  } else if (neventsCells) {
    return neventsCells;
  } else {
    return 0;
  }
}

template <class CellInputType>
o2::InteractionRecord EventHandler<CellInputType>::getInteractionRecordForEvent(int eventID) const
{
  std::optional<o2::InteractionRecord> irClusters, irCells;
  if (mTriggerRecordsClusters.size()) {
    if (eventID >= mTriggerRecordsClusters.size()) {
      throw RangeException(eventID, mTriggerRecordsClusters.size());
    }
    irClusters = mTriggerRecordsClusters[eventID].getBCData();
  }
  if (mTriggerRecordsCells.size()) {
    if (eventID >= mTriggerRecordsCells.size()) {
      throw RangeException(eventID, mTriggerRecordsCells.size());
    }
    irCells = mTriggerRecordsCells[eventID].getBCData();
  }
  if (irClusters && irCells) {
    if (compareInteractionRecords(irClusters.value(), irCells.value())) {
      return irClusters.value();
    } else {
      throw InteractionRecordInvalidException(irClusters.value(), irCells.value());
    }
  } else if (irClusters) {
    return irClusters.value();
  } else if (irCells) {
    return irCells.value();
  }
  throw NotInitializedException();
}

template <class CellInputType>
uint64_t EventHandler<CellInputType>::getTriggerBitsForEvent(int eventID) const
{
  std::optional<uint64_t> triggerBitsClusters, triggerBitsCells;
  if (mTriggerRecordsClusters.size()) {
    if (eventID >= mTriggerRecordsClusters.size()) {
      throw RangeException(eventID, mTriggerRecordsClusters.size());
    }
    triggerBitsClusters = mTriggerRecordsClusters[eventID].getTriggerBits();
  }
  if (mTriggerRecordsCells.size()) {
    if (eventID >= mTriggerRecordsCells.size()) {
      throw RangeException(eventID, mTriggerRecordsCells.size());
    }
    triggerBitsClusters = mTriggerRecordsCells[eventID].getTriggerBits();
  }
  if (triggerBitsClusters && triggerBitsCells) {
    if (triggerBitsClusters == triggerBitsCells) {
      return triggerBitsClusters.value();
    } else {
      throw TriggerBitsInvalidException(triggerBitsClusters.value(), triggerBitsCells.value());
    }
  } else if (triggerBitsClusters) {
    return triggerBitsClusters.value();
  } else if (triggerBitsCells) {
    return triggerBitsCells.value();
  }
  throw NotInitializedException();
}

template <class CellInputType>
const typename EventHandler<CellInputType>::ClusterRange EventHandler<CellInputType>::getClustersForEvent(int eventID) const
{
  if (mTriggerRecordsClusters.size()) {
    if (eventID >= mTriggerRecordsClusters.size()) {
      throw RangeException(eventID, mTriggerRecordsClusters.size());
    }
    auto& trgrecord = mTriggerRecordsClusters[eventID];
    return ClusterRange(mClusters.data() + trgrecord.getFirstEntry(), trgrecord.getNumberOfObjects());
  }
  throw NotInitializedException();
}

template <class CellInputType>
const typename EventHandler<CellInputType>::CellRange EventHandler<CellInputType>::getCellsForEvent(int eventID) const
{
  if (mTriggerRecordsCells.size()) {
    if (eventID >= mTriggerRecordsCells.size()) {
      throw RangeException(eventID, mTriggerRecordsCells.size());
    }
    auto& trgrecord = mTriggerRecordsCells[eventID];
    return CellRange(mCells.data() + trgrecord.getFirstEntry(), trgrecord.getNumberOfObjects());
  }
  throw NotInitializedException();
}

template <class CellInputType>
std::vector<gsl::span<const o2::emcal::MCLabel>> EventHandler<CellInputType>::getCellMCLabelForEvent(int eventID) const
{
  if (mCellLabels && mTriggerRecordsCells.size()) {
    if (eventID >= mTriggerRecordsCells.size()) {
      throw RangeException(eventID, mTriggerRecordsCells.size());
    }
    auto& trgrecord = mTriggerRecordsCells[eventID];
    std::vector<gsl::span<const o2::emcal::MCLabel>> eventlabels(trgrecord.getNumberOfObjects());
    for (int index = 0; index < trgrecord.getNumberOfObjects(); index++) {
      eventlabels[index] = mCellLabels->getLabels(trgrecord.getFirstEntry() + index);
    }
    return eventlabels;
  }
  throw NotInitializedException();
}

template <class CellInputType>
const typename EventHandler<CellInputType>::CellIndexRange EventHandler<CellInputType>::getClusterCellIndicesForEvent(int eventID) const
{
  if (mTriggerRecordsCellIndices.size()) {
    if (eventID >= mTriggerRecordsCellIndices.size()) {
      throw RangeException(eventID, mTriggerRecordsCellIndices.size());
    }
    auto& trgrecord = mTriggerRecordsCellIndices[eventID];
    return CellIndexRange(mClusterCellIndices.data() + trgrecord.getFirstEntry(), trgrecord.getNumberOfObjects());
  }
  throw NotInitializedException();
}

template <class CellInputType>
void EventHandler<CellInputType>::reset()
{
  mTriggerRecordsClusters = TriggerRange();
  mTriggerRecordsCellIndices = TriggerRange();
  mTriggerRecordsCells = TriggerRange();
  mClusters = ClusterRange();
  mClusterCellIndices = CellIndexRange();
  mCells = CellRange();
  mCellLabels = nullptr;
}

template <class CellInputType>
EventData<CellInputType> EventHandler<CellInputType>::buildEvent(int eventID) const
{
  EventData<CellInputType> outputEvent;
  outputEvent.mInteractionRecord = getInteractionRecordForEvent(eventID);
  outputEvent.mTriggerBits = getTriggerBitsForEvent(eventID);
  if (hasClusters()) {
    outputEvent.mClusters = getClustersForEvent(eventID);
  }
  if (hasClusterIndices()) {
    outputEvent.mCellIndices = getClusterCellIndicesForEvent(eventID);
  }
  if (hasCells()) {
    outputEvent.mCells = getCellsForEvent(eventID);
  }
  if (mCellLabels) {
    outputEvent.mMCCellLabels = getCellMCLabelForEvent(eventID);
  }

  return outputEvent;
}

template <class CellInputType>
bool EventHandler<CellInputType>::compareInteractionRecords(const InteractionRecord& lhs, const InteractionRecord& rhs) const
{
  return lhs.bc == rhs.bc && lhs.orbit == rhs.orbit;
}

template <class CellInputType>
EventHandler<CellInputType>::EventIterator::EventIterator(const EventHandler<CellInputType>& handler, int eventID, bool forward) : mEventHandler(handler),
                                                                                                                                   mCurrentEvent(),
                                                                                                                                   mEventID(eventID),
                                                                                                                                   mForward(forward)
{
  mCurrentEvent = mEventHandler.buildEvent(mEventID);
}

template <class CellInputType>
bool EventHandler<CellInputType>::EventIterator::operator==(const EventHandler<CellInputType>::EventIterator& rhs) const
{
  return &mEventHandler == &rhs.mEventHandler && mEventID == rhs.mEventID && mForward == rhs.mForward;
}

template <class CellInputType>
typename EventHandler<CellInputType>::EventIterator& EventHandler<CellInputType>::EventIterator::operator++()
{
  if (mForward) {
    mEventID++;
  } else {
    mEventID--;
  }
  mCurrentEvent = mEventHandler.buildEvent(mEventID);
  return *this;
}

template <class CellInputType>
typename EventHandler<CellInputType>::EventIterator EventHandler<CellInputType>::EventIterator::operator++(int)
{
  auto tmp = *this;
  ++(*this);
  return tmp;
}

template <class CellInputType>
typename EventHandler<CellInputType>::EventIterator& EventHandler<CellInputType>::EventIterator::operator--()
{
  if (mForward) {
    mEventID--;
  } else {
    mEventID++;
  }
  mCurrentEvent = mEventHandler.buildEvent(mEventID);
  return *this;
}

template <class CellInputType>
typename EventHandler<CellInputType>::EventIterator EventHandler<CellInputType>::EventIterator::operator--(int)
{
  auto tmp = *this;
  --(*this);
  return tmp;
}

template class o2::emcal::EventHandler<o2::emcal::Cell>;
template class o2::emcal::EventHandler<o2::emcal::Digit>;
