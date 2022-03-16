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

/// \file EventFinder.h
/// \brief Definition of a class to group MCH digits based on MID information
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_EVENTFINDER_H_
#define O2_MCH_EVENTFINDER_H_

#include <map>
#include <unordered_map>
#include <vector>

#include <gsl/span>

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMID/ROFRecord.h"

namespace o2
{
namespace mch
{

/// Class to group MCH digits based on MID information
class EventFinder
{
 public:
  EventFinder() = default;
  ~EventFinder() = default;

  EventFinder(const EventFinder&) = delete;
  EventFinder& operator=(const EventFinder&) = delete;
  EventFinder(EventFinder&&) = delete;
  EventFinder& operator=(EventFinder&&) = delete;

  void run(const gsl::span<const mch::ROFRecord>& mchROFs, const gsl::span<const mch::Digit>& digits,
           const dataformats::MCLabelContainer* labels, const gsl::span<const mid::ROFRecord>& midROFs);

  /// get the output MCH ROFs
  const std::vector<mch::ROFRecord>& getOutputROFs() const { return mROFs; }
  /// get the output MCH digits
  const std::vector<mch::Digit>& getOutputDigits() const { return mDigits; }
  /// get the output MCH labels
  const dataformats::MCLabelContainer& getOutputLabels() const { return mLabels; }

 private:
  /// internal event structure
  struct Event {
    /// contruct an empty event with trigger window = [bcMin, bcMax[
    Event(int64_t bcMin, int64_t bcMax) : maxRORange(bcMin)
    {
      trgRange[0] = bcMin;
      trgRange[1] = bcMax;
    }

    int64_t trgRange[2]{};       ///< BC range of the MID trigger window
    int64_t maxRORange = 0;      ///< upper limit of the MCH RO window
    std::vector<int> iMCHROFs{}; ///< list of associated MCH ROF indices
  };

  std::map<int64_t, Event> mEvents{};       ///< sorted list of events found
  std::vector<mch::ROFRecord> mROFs{};      ///< list of output MCH ROFs
  std::vector<mch::Digit> mDigits{};        ///< list of output MCH digits
  dataformats::MCLabelContainer mLabels{};  ///< container of output MCH digit labels
  std::unordered_map<int, int> mDigitLoc{}; ///< map the digit indices of a particular event
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_EVENTFINDER_H_
