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

#include <algorithm>
#include <iostream>
#include <DataFormatsEMCAL/Constants.h>
#include <EMCALReconstruction/RecoContainer.h>

using namespace o2::emcal;

EventContainer::EventContainer(const o2::InteractionRecord& currentIR) : mInteractionRecord(currentIR) {}

void EventContainer::setCellCommon(int tower, double energy, double time, ChannelType_t celltype, bool isLEDmon, int hwaddress, int fecID, int ddlID, bool doMergeHGLG)
{
  auto updateCell = [energy, time, celltype](Cell& cell) {
    cell.setEnergy(energy);
    cell.setTimeStamp(time);
    cell.setType(celltype);
  };
  auto& datacontainer = isLEDmon ? mLEDMons : mCells;
  if (doMergeHGLG) {
    auto found = std::find_if(datacontainer.begin(), datacontainer.end(), [&tower](const o2::emcal::RecCellInfo& testcell) -> bool { return testcell.mCellData.getTower() == tower; });
    if (found != datacontainer.end()) {
      // Cell already existing, store LG if HG is larger then the overflow cut
      if (celltype == o2::emcal::ChannelType_t::LOW_GAIN) {
        found->mHWAddressLG = hwaddress;
        found->mHGOutOfRange = false; // LG is found so it can replace the HG if the HG is out of range
        if (found->mCellData.getHighGain()) {
          if (isCellSaturated(found->mCellData.getEnergy())) {
            // High gain digit has energy above overflow cut, use low gain instead
            updateCell(found->mCellData);
          }
          found->mIsLGnoHG = false;
        }
      } else {
        // new channel would be HG: use that if it is below overflow cut
        // as the channel existed before it must have been a LG channel,
        // which would be used in case the HG is out-of-range
        found->mIsLGnoHG = false;
        found->mHGOutOfRange = false;
        found->mHWAddressHG = hwaddress;
        if (!isCellSaturated(energy)) {
          updateCell(found->mCellData);
        }
      }
    } else {
      // New cell
      bool lgNoHG = false;       // Flag for filter of cells which have only low gain but no high gain
      bool hgOutOfRange = false; // Flag if only a HG is present which is out-of-range
      int hwAddressLG = -1,      // Hardware address of the LG of the tower (for monitoring)
        hwAddressHG = -1;        // Hardware address of the HG of the tower (for monitoring)
      if (celltype == o2::emcal::ChannelType_t::LOW_GAIN) {
        lgNoHG = true;
        hwAddressLG = hwaddress;
      } else {
        // High gain cell: Flag as low gain if above threshold
        hgOutOfRange = isCellSaturated(energy);
        hwAddressHG = hwaddress;
      }
      datacontainer.push_back({o2::emcal::Cell(tower, energy, time, celltype),
                               lgNoHG,
                               hgOutOfRange,
                               fecID, ddlID, hwAddressLG, hwAddressHG});
    }
  } else {
    // No merge of HG/LG cells (usually MC where either
    // of the two is simulated)
    int hwAddressLG = celltype == ChannelType_t::LOW_GAIN ? hwaddress : -1,
        hwAddressHG = celltype == ChannelType_t::HIGH_GAIN ? hwaddress : -1;
    // New cell
    datacontainer.push_back({o2::emcal::Cell(tower, energy, time, celltype),
                             false,
                             false,
                             fecID, ddlID, hwAddressLG, hwAddressHG});
  }
}

void EventContainer::sortCells(bool isLEDmon)
{
  auto& dataContainer = isLEDmon ? mLEDMons : mCells;
  std::sort(dataContainer.begin(), dataContainer.end(), [](const RecCellInfo& lhs, const RecCellInfo& rhs) -> bool { return lhs.mCellData.getTower() < rhs.mCellData.getTower(); });
}

bool EventContainer::isCellSaturated(double energy) const
{
  return energy / o2::emcal::constants::EMCAL_ADCENERGY > o2::emcal::constants::OVERFLOWCUT;
}

EventContainer& RecoContainer::getEventContainer(const o2::InteractionRecord& currentIR)
{
  auto found = mEvents.find(currentIR);
  if (found != mEvents.end()) {
    return found->second;
  }
  // interaction not found, create new container
  auto res = mEvents.insert({currentIR, EventContainer(currentIR)});
  return res.first->second;
}

const EventContainer& RecoContainer::getEventContainer(const o2::InteractionRecord& currentIR) const
{
  auto found = mEvents.find(currentIR);
  if (found == mEvents.end()) {
    throw InteractionNotFoundException(currentIR);
  }
  return found->second;
}

std::vector<o2::InteractionRecord> RecoContainer::getOrderedInteractions() const
{
  std::vector<o2::InteractionRecord> result;
  for (auto& entry : mEvents) {
    result.emplace_back(entry.first);
  }
  std::sort(result.begin(), result.end(), std::less<>());
  return result;
}

RecoContainerReader::RecoContainerReader(RecoContainer& container) : mDataContainer(container)
{
  mOrderedInteractions = mDataContainer.getOrderedInteractions();
}

EventContainer& RecoContainerReader::nextEvent()
{
  if (!hasNext()) {
    throw InvalidAccessException();
  }
  int currentID = mCurrentEvent;
  mCurrentEvent++;
  try {
    return mDataContainer.getEventContainer(mOrderedInteractions[currentID]);
  } catch (RecoContainer::InteractionNotFoundException& e) {
    throw InvalidAccessException();
  }
}