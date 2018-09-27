// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
namespace o2
{
namespace framework
{

inline bool TimesliceId::isValid(TimesliceId const& timeslice) { return timeslice.value != INVALID; }

inline void TimesliceIndex::resize(size_t s)
{
  mTimeslices.resize(s, { TimesliceId::INVALID });
  mDirty.resize(s, false);
}

inline size_t TimesliceIndex::size() const
{
  assert(mTimeslices.size() == mDirty.size());
  return mTimeslices.size();
}

inline bool TimesliceIndex::isValid(TimesliceSlot const& slot) const
{
  return TimesliceId::isValid(mTimeslices[slot.index]);
}

inline bool TimesliceIndex::isDirty(TimesliceSlot const& slot) const
{
  return mDirty[slot.index];
}

inline bool TimesliceIndex::isObsolete(TimesliceId newTimestamp) const
{
  auto slot = getSlotForTimeslice(newTimestamp);
  auto oldTimestamp = mTimeslices[slot.index];
  return TimesliceId::isValid(oldTimestamp) && oldTimestamp.value > newTimestamp.value;
}

inline void TimesliceIndex::markAsDirty(TimesliceSlot slot, bool value)
{
  mDirty[slot.index] = value;
}

inline void TimesliceIndex::markAsObsolete(TimesliceSlot slot)
{
  // Invalid slots remain invalid.
  if (TimesliceId::isValid(mTimeslices[slot.index])) {
    mTimeslices[slot.index].value += 1;
  }
}

inline void TimesliceIndex::markAsInvalid(TimesliceSlot slot)
{
  mTimeslices[slot.index] = { TimesliceId::INVALID };
}

inline TimesliceSlot TimesliceIndex::bookTimeslice(TimesliceId timestamp)
{
  auto slot = getSlotForTimeslice(timestamp);
  mTimeslices[slot.index] = timestamp;
  mDirty[slot.index] = true;
  return slot;
}

inline TimesliceSlot TimesliceIndex::getSlotForTimeslice(TimesliceId timestamp) const
{
  return TimesliceSlot{ timestamp.value % mTimeslices.size() };
}

inline TimesliceId TimesliceIndex::getTimesliceForSlot(TimesliceSlot slot) const
{
  return mTimeslices[slot.index];
}

} // namespace framework
} // namespace o2
