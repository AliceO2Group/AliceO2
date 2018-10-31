// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_TIMESLICEINDEX_H
#define FRAMEWORK_TIMESLICEINDEX_H

#include <vector>
#include <cstdint>

/// This class holds the information on how to map a given timeslice to a
/// record row in the cache.  It also holds whether or not a given timeslice is
/// dirty and can be used to allocate a new timeslice.  This is also exposed as
/// a service for those who need to create timeslices without data being
/// associated to it.
///
/// In the future this could also come handy to the same timeslice to multiple
/// entries in the cache.

namespace o2
{
namespace framework
{

struct TimesliceId {
  static constexpr uint64_t INVALID = -1;
  size_t value;
  static bool isValid(TimesliceId const& timeslice);
};

struct TimesliceSlot {
  size_t index;
};

class TimesliceIndex
{
 public:
  inline void resize(size_t s);
  inline size_t size() const;
  inline bool isValid(TimesliceSlot const& slot) const;
  inline bool isDirty(TimesliceSlot const& slot) const;
  inline bool isObsolete(TimesliceId newTimestamp) const;
  inline void markAsDirty(TimesliceSlot slot, bool value);
  inline void markAsObsolete(TimesliceSlot slot);
  inline void markAsInvalid(TimesliceSlot slot);
  inline TimesliceSlot bookTimeslice(TimesliceId timestamp);
  inline TimesliceSlot getSlotForTimeslice(TimesliceId timestamp) const;
  inline TimesliceId getTimesliceForSlot(TimesliceSlot slot) const;

 private:
  /// This is the timeslices for all the in flight parts.
  std::vector<TimesliceId> mTimeslices;
  /// This keeps track whether or not something was relayed
  /// since last time we called getReadyToProcess()
  std::vector<bool> mDirty;
};

} // namespace framework
} // namespace o2

#include "TimesliceIndex.inl"
#endif // FRAMEWORK_TIMESLICEINDEX_H
