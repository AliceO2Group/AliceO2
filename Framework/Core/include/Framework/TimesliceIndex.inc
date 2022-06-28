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
namespace o2::framework
{

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

inline void TimesliceIndex::rescan()
{
  for (size_t i = 0; i < mDirty.size(); i++) {
    mDirty[i] = true;
  }
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

inline data_matcher::VariableContext& TimesliceIndex::getVariablesForSlot(TimesliceSlot slot)
{
  assert(mVariables.size() > slot.index);
  return mVariables[slot.index];
}

inline data_matcher::VariableContext& TimesliceIndex::getPublishedVariablesForSlot(TimesliceSlot slot)
{
  assert(mPublishedVariables.size() > slot.index);
  return mPublishedVariables[slot.index];
}

inline TimesliceIndex::OldestInputInfo TimesliceIndex::getOldestPossibleInput() const
{
  return mOldestPossibleInput;
}

inline TimesliceIndex::OldestOutputInfo TimesliceIndex::getOldestPossibleOutput() const
{
  return mOldestPossibleOutput;
}

} // namespace o2::framework
