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

/// \file EventFinder.cxx
/// \brief Implementation of a class to group MCH digits based on MID information
///
/// \author Philippe Pillot, Subatech

#include "MCHTriggering/EventFinder.h"

#include <algorithm>
#include <iterator>
#include <utility>

#include "CommonDataFormat/InteractionRecord.h"
#include "MCHTriggering/EventFinderParam.h"

namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
/// run the event finder algorithm
void EventFinder::run(const gsl::span<const mch::ROFRecord>& mchROFs,
                      const gsl::span<const mch::Digit>& digits,
                      const dataformats::MCLabelContainer* labels,
                      const gsl::span<const mid::ROFRecord>& midROFs)
{
  mEvents.clear();
  mROFs.clear();
  mDigits.clear();
  mLabels.clear();

  if (mchROFs.empty() || midROFs.empty()) {
    return;
  }

  // create empty events ordered in increasing trigger time
  const auto& param = EventFinderParam::Instance();
  for (const auto& midROF : midROFs) {
    if (midROF.nEntries == 0) {
      continue;
    }
    auto midBC = midROF.interactionRecord.toLong();
    mEvents.emplace(std::make_pair(midBC, Event(midBC + param.triggerRange[0], midBC + param.triggerRange[1])));
  }

  // associate each MCH ROF to the first compatible trigger, if any
  for (int i = 0; i < mchROFs.size(); ++i) {
    auto mchBC = mchROFs[i].getBCData().toLong();
    auto itEvent = mEvents.lower_bound(mchBC - param.triggerRange[1] + 1);
    if (itEvent != mEvents.end() && itEvent->second.trgRange[0] < mchBC + mchROFs[i].getBCWidth()) {
      itEvent->second.iMCHROFs.push_back(i);
      itEvent->second.maxRORange = std::max(itEvent->second.maxRORange, mchBC + mchROFs[i].getBCWidth());
    }
  }

#pragma GCC diagnostic push                          // TODO: Remove once this is fixed in GCC
#pragma GCC diagnostic ignored "-Wstringop-overflow" // TODO: Remove once this is fixed in GCC
  // merge overlapping events (when a MCH ROF is compatible with multiple trigger) and cleanup
  for (auto itEvent = mEvents.begin(); itEvent != mEvents.end();) {
    if (itEvent->second.iMCHROFs.empty()) {
      itEvent = mEvents.erase(itEvent);
    } else {
      auto itNextEvent = std::next(itEvent);
      if (itNextEvent != mEvents.end() && itNextEvent->second.trgRange[0] < itEvent->second.maxRORange) {
        itEvent->second.trgRange[1] = itNextEvent->second.trgRange[1];
        itEvent->second.maxRORange = std::max(itEvent->second.maxRORange, itNextEvent->second.maxRORange);
        itEvent->second.iMCHROFs.insert(itEvent->second.iMCHROFs.end(),
                                        itNextEvent->second.iMCHROFs.begin(), itNextEvent->second.iMCHROFs.end());
        mEvents.erase(itNextEvent);
      } else {
        itEvent = itNextEvent;
      }
    }
  }
#pragma GCC diagnostic pop // TODO: Remove once this is fixed in GCC

  // merge digits associated to each event and produce the output ROFs pointing to them
  // the BC range of each ROF is set to contain only the MID IR(s) associated to the event
  InteractionRecord ir{};
  for (const auto& event : mEvents) {
    mDigitLoc.clear();
    int digitOffset = mDigits.size();
    for (auto iMCHROF : event.second.iMCHROFs) {
      for (int iDigit = mchROFs[iMCHROF].getFirstIdx(); iDigit <= mchROFs[iMCHROF].getLastIdx(); ++iDigit) {
        const auto& digit = digits[iDigit];
        auto digitLoc = mDigitLoc.emplace(((digit.getDetID() << 16) | digit.getPadID()), mDigits.size());
        if (digitLoc.second) {
          mDigits.emplace_back(digit);
          if (labels != nullptr) {
            mLabels.addElements(digitLoc.first->second, labels->getLabels(iDigit));
          }
        } else {
          auto& digit0 = mDigits[digitLoc.first->second];
          digit0.setADC(digit0.getADC() + digit.getADC());
          auto nofSamples = digit0.getNofSamples() + digit.getNofSamples();
          digit0.setNofSamples((nofSamples > 0x3FF) ? 0x3FF : nofSamples);
          digit0.setSaturated(digit0.isSaturated() || digit.isSaturated());
          if (labels != nullptr) {
            for (const auto& label : labels->getLabels(iDigit)) {
              mLabels.addElementRandomAccess(digitLoc.first->second, label);
            }
          }
        }
      }
    }
    ir.setFromLong(event.first);
    mROFs.emplace_back(ir, digitOffset, mDigits.size() - digitOffset,
                       event.second.trgRange[1] - param.triggerRange[1] - event.first + 1);
  }
}

} // namespace mch
} // namespace o2
