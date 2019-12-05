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

inline void TimesliceIndex::resize(size_t s)
{
  mVariables.resize(s);
  mPublishedVariables.resize(s);
  mDirty.resize(s, false);
}

inline size_t TimesliceIndex::size() const
{
  assert(mVariables.size() == mDirty.size());
  return mVariables.size();
}

inline bool TimesliceIndex::isValid(TimesliceSlot const& slot) const
{
  // The first element is always time, which should always be filled after a
  // successful query, therefore if this is not the case, we return false.
  assert(mVariables.size() > slot.index);
  if (std::get_if<uint64_t>(&mVariables[slot.index].get(0))) {
    return true;
  }
  return false;
}

inline bool TimesliceIndex::isDirty(TimesliceSlot const& slot) const
{
  assert(mDirty.size() > slot.index);
  return mDirty[slot.index];
}

inline void TimesliceIndex::markAsDirty(TimesliceSlot slot, bool value)
{
  assert(mDirty.size() > slot.index);
  mDirty[slot.index] = value;
}

inline void TimesliceIndex::markAsInvalid(TimesliceSlot slot)
{
  assert(mVariables.size() > slot.index);
  mVariables[slot.index].reset();
}

inline void TimesliceIndex::publishSlot(TimesliceSlot slot)
{
  assert(mVariables.size() > slot.index);
  mPublishedVariables[slot.index] = mVariables[slot.index];
}

inline void TimesliceIndex::associate(TimesliceId timestamp, TimesliceSlot slot)
{
  assert(mVariables.size() > slot.index);
  mVariables[slot.index].put({0, static_cast<uint64_t>(timestamp.value)});
  mVariables[slot.index].commit();
  mDirty[slot.index] = true;
}

inline TimesliceSlot TimesliceIndex::findOldestSlot() const
{
  TimesliceSlot oldest{0};
  auto oldPVal = std::get_if<uint64_t>(&mVariables[oldest.index].get(0));
  if (oldPVal == nullptr) {
    return oldest;
  }
  uint64_t oldTimestamp = *oldPVal;

  for (size_t i = 1; i < mVariables.size(); ++i) {
    auto newPVal = std::get_if<uint64_t>(&mVariables[i].get(0));
    if (newPVal == nullptr) {
      return TimesliceSlot{i};
    }
    uint64_t newTimestamp = *newPVal;

    if (oldTimestamp > newTimestamp) {
      oldest = TimesliceSlot{i};
      oldTimestamp = newTimestamp;
    }
  }
  return oldest;
}

inline TimesliceId TimesliceIndex::getTimesliceForSlot(TimesliceSlot slot) const
{
  assert(mVariables.size() > slot.index);
  auto pval = std::get_if<uint64_t>(&mVariables[slot.index].get(0));
  if (pval == nullptr) {
    return TimesliceId{TimesliceId::INVALID};
  }
  return TimesliceId{*pval};
}

inline data_matcher::VariableContext &TimesliceIndex::getVariablesForSlot(TimesliceSlot slot)
{
  assert(mVariables.size() > slot.index);
  return mVariables[slot.index];
}

inline data_matcher::VariableContext &TimesliceIndex::getPublishedVariablesForSlot(TimesliceSlot slot)
{
  assert(mPublishedVariables.size() > slot.index);
  return mPublishedVariables[slot.index];
}

inline std::tuple<TimesliceIndex::ActionTaken, TimesliceSlot> TimesliceIndex::replaceLRUWith(data_matcher::VariableContext& newContext)
{
  auto oldestSlot = findOldestSlot();
  if (TimesliceIndex::isValid(oldestSlot) == false) {
    mVariables[oldestSlot.index] = newContext;
    return std::make_tuple(ActionTaken::ReplaceUnused, oldestSlot);
  }
  auto oldTimestamp = std::get_if<uint64_t>(&mVariables[oldestSlot.index].get(0));
  if (oldTimestamp == nullptr) {
    mVariables[oldestSlot.index] = newContext;
    return std::make_tuple(ActionTaken::ReplaceUnused, oldestSlot);
  }

  auto newTimestamp = std::get_if<uint64_t>(&newContext.get(0));
  if (newTimestamp == nullptr) {
    return std::make_tuple(ActionTaken::DropInvalid, TimesliceSlot{TimesliceSlot::INVALID});
  }

  if (*newTimestamp > *oldTimestamp) {
    mVariables[oldestSlot.index] = newContext;
    return std::make_tuple(ActionTaken::ReplaceObsolete, oldestSlot);
  }
  return std::make_tuple(ActionTaken::DropObsolete, TimesliceSlot{TimesliceSlot::INVALID});
}

} // namespace framework
} // namespace o2
