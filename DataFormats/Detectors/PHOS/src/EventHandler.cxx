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

#include "DataFormatsPHOS/EventHandler.h"
#include <optional>

using namespace o2::phos;

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
    if (static_cast<std::size_t>(eventID) >= mTriggerRecordsClusters.size()) {
      throw RangeException(eventID, mTriggerRecordsClusters.size());
    }
    irClusters = mTriggerRecordsClusters[eventID].getBCData();
  }
  if (mTriggerRecordsCells.size()) {
    if (static_cast<std::size_t>(eventID) >= mTriggerRecordsCells.size()) {
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
const typename EventHandler<CellInputType>::ClusterRange EventHandler<CellInputType>::getClustersForEvent(int eventID) const
{
  if (mTriggerRecordsClusters.size()) {
    if (static_cast<std::size_t>(eventID) >= mTriggerRecordsClusters.size()) {
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
    if (static_cast<std::size_t>(eventID) >= mTriggerRecordsCells.size()) {
      throw RangeException(eventID, mTriggerRecordsCells.size());
    }
    auto& trgrecord = mTriggerRecordsCells[eventID];
    return CellRange(mCells.data() + trgrecord.getFirstEntry(), trgrecord.getNumberOfObjects());
  }
  throw NotInitializedException();
}

template <class CellInputType>
const typename EventHandler<CellInputType>::CellIndexRange EventHandler<CellInputType>::getClusterCellIndicesForEvent(int eventID) const
{
  if (mTriggerRecordsCellIndices.size()) {
    if (static_cast<std::size_t>(eventID) >= mTriggerRecordsCellIndices.size()) {
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
}

template <class CellInputType>
EventData<CellInputType> EventHandler<CellInputType>::buildEvent(int eventID) const
{
  EventData<CellInputType> outputEvent;
  outputEvent.mInteractionRecord = getInteractionRecordForEvent(eventID);
  if (hasClusters()) {
    outputEvent.mClusters = getClustersForEvent(eventID);
  }
  if (hasClusterIndices()) {
    outputEvent.mCellIndices = getClusterCellIndicesForEvent(eventID);
  }
  if (hasCells()) {
    outputEvent.mCells = getCellsForEvent(eventID);
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

template class o2::phos::EventHandler<o2::phos::Cell>;
template class o2::phos::EventHandler<o2::phos::Digit>;
