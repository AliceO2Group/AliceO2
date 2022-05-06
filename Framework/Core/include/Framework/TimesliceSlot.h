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
#ifndef O2_FRAMEWORK_TIMESLICESLOT_H_
#define O2_FRAMEWORK_TIMESLICESLOT_H_

#include <cstdint>
#include <cstddef>

namespace o2::framework
{

struct TimesliceId {
  static constexpr uint64_t INVALID = -1;
  size_t value;
  static bool isValid(TimesliceId const& timeslice);
};

struct TimesliceSlot {
  static constexpr uint64_t INVALID = -1;
  static constexpr uint64_t ANY = -2;
  size_t index;
  static bool isValid(TimesliceSlot const& slot);
  bool operator==(const TimesliceSlot that) const;
  bool operator!=(const TimesliceSlot that) const;
};

inline bool TimesliceId::isValid(TimesliceId const& timeslice) { return timeslice.value != INVALID; }
inline bool TimesliceSlot::isValid(TimesliceSlot const& slot) { return slot.index != INVALID; }

inline bool TimesliceSlot::operator==(const TimesliceSlot that) const
{
  return index == that.index ||
         (index == TimesliceSlot::ANY && TimesliceSlot::INVALID != that.index) ||
         (index != TimesliceSlot::INVALID && TimesliceSlot::ANY == that.index);
}

inline bool TimesliceSlot::operator!=(const TimesliceSlot that) const
{
  return !(*this == that);
}

} // namespace o2::framework
#endif // O2_FRAMEWORK_TIMESLICESLOT_H_
