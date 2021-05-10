// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTOR_CALIB_TIMESLOTCALIB_H_
#define DETECTOR_CALIB_TIMESLOTCALIB_H_

/// @brief Processor for the multiple time slots calibration

#include "DetectorsCalibration/TimeSlot.h"
#include <deque>
#include <gsl/gsl>
#include <limits>

namespace o2
{
namespace calibration
{

template <typename Input, typename Container>
class TimeSlotCalibration
{
  using Slot = TimeSlot<Container>;

 public:
  TimeSlotCalibration() = default;
  virtual ~TimeSlotCalibration() = default;
  uint64_t getMaxSlotsDelay() const { return mMaxSlotsDelay; }
  void setMaxSlotsDelay(uint64_t v) { mMaxSlotsDelay = v; }

  uint64_t getSlotLength() const { return mSlotLength; }
  void setSlotLength(uint64_t v) { mSlotLength = v < 1 ? 1 : v; }

  uint64_t getCheckIntervalInfiniteSlot() const { return mCheckIntervalInfiniteSlot; }
  void setCheckIntervalInfiniteSlot(uint64_t v) { mCheckIntervalInfiniteSlot = v; }

  uint64_t getCheckDeltaIntervalInfiniteSlot() const { return mCheckDeltaIntervalInfiniteSlot; }
  void setCheckDeltaIntervalInfiniteSlot(uint64_t v) { mCheckDeltaIntervalInfiniteSlot = v < 1 ? mCheckIntervalInfiniteSlot : v; } // if the delta is 0, we ignore it

  TFType getFirstTF() const { return mFirstTF; }
  void setFirstTF(TFType v) { mFirstTF = v; }

  void setUpdateAtTheEndOfRunOnly() { mUpdateAtTheEndOfRunOnly = kTRUE; }

  int getNSlots() const { return mSlots.size(); }
  Slot& getSlotForTF(TFType tf);
  Slot& getSlot(int i) { return (Slot&)mSlots.at(i); }
  const Slot& getSlot(int i) const { return (Slot&)mSlots.at(i); }
  const Slot& getLastSlot() const { return (Slot&)mSlots.back(); }
  const Slot& getFirstSlot() const { return (Slot&)mSlots.front(); }

  virtual bool process(TFType tf, const gsl::span<const Input> data);
  virtual void checkSlotsToFinalize(TFType tf, int maxDelay = 0);
  virtual void finalizeOldestSlot();

  // Methods to be implemented by the derived user class

  // implement and call this method te reset the output slots once they are not needed
  virtual void initOutput() = 0;
  // process the time slot container and add results to the output
  virtual void finalizeSlot(Slot& slot) = 0;
  // create new time slot in the beginning or the end of the slots pool
  virtual Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) = 0;
  // check if the slot has enough data to be finalized
  virtual bool hasEnoughData(const Slot& slot) const = 0;

  virtual void print() const;

 protected:
  auto& getSlots() { return mSlots; }

 private:
  TFType tf2SlotMin(TFType tf) const;

  std::deque<Slot> mSlots;

  TFType mLastClosedTF = 0;
  TFType mFirstTF = 0;
  TFType mMaxSeenTF = 0; // largest TF processed
  uint64_t mSlotLength = 1;
  uint64_t mMaxSlotsDelay = 3;
  bool mUpdateAtTheEndOfRunOnly = false;
  uint64_t mCheckIntervalInfiniteSlot = 1;      // will be used if the TF length is INFINITE_TF_int64 to decide
                                                // when to check if to call the finalize; otherwise it is called
                                                // at every new TF; note that this is an approximation,
                                                // since TFs come in async order
  TFType mLastCheckedTFInfiniteSlot = 0;        // will be used if the TF length is INFINITE_TF_int64 to book-keep
                                                // the last TF at which we tried to calibrate
  uint64_t mCheckDeltaIntervalInfiniteSlot = 1; // will be used if the TF length is INFINITE_TF_int64 when
                                                // the check on the statistics returned false, to determine
                                                // after how many TF to check again.
  bool mWasCheckedInfiniteSlot = false;         // flag to know whether the statistics of the infinite slot was already checked

  ClassDef(TimeSlotCalibration, 1);
};

//_________________________________________________
template <typename Input, typename Container>
bool TimeSlotCalibration<Input, Container>::process(TFType tf, const gsl::span<const Input> data)
{

  // process current TF

  int maxDelay = mMaxSlotsDelay * mSlotLength;
  if (!mUpdateAtTheEndOfRunOnly) {                                                               // if you update at the end of run only, then you accept everything
    if (tf < mLastClosedTF || (!mSlots.empty() && getLastSlot().getTFStart() > tf + maxDelay)) { // ignore TF; note that if you have only 1 timeslot
                                                                                                 // which is INFINITE_TF wide, then maxDelay
                                                                                                 // does not matter: you won't accept TFs from the past,
                                                                                                 // so the first condition will be used
      LOG(INFO) << "Ignoring TF " << tf << ", mLastClosedTF = " << mLastClosedTF;
      return false;
    }
  }

  auto& slotTF = getSlotForTF(tf);
  slotTF.getContainer()->fill(data);
  if (tf > mMaxSeenTF) {
    mMaxSeenTF = tf; // keep track of the most recent TF processed
  }
  if (!mUpdateAtTheEndOfRunOnly) { // if you update at the end of run only, you don't check at every TF which slots can be closed
    // check if some slots are done
    checkSlotsToFinalize(tf, maxDelay);
  }

  return true;
}

//_________________________________________________
template <typename Input, typename Container>
void TimeSlotCalibration<Input, Container>::checkSlotsToFinalize(TFType tf, int maxDelay)
{
  // Check which slots can be finalized, provided the newly arrived TF is tf

  constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
  constexpr int64_t INFINITE_TF_int64 = std::numeric_limits<long>::max() - 1; // this is used to define the end
                                                                              // of the slot in case it is "std::numeric_limits<long>::max()"
                                                                              // long (so we need to subtract 1)

  // if we have one slot only which is INFINITE_TF_int64 long, and we are not at the end of run (tf != INFINITE_TF),
  // we need to check if we got enough statistics, and if so, redefine the slot
  if (mSlots.size() == 1 && mSlots[0].getTFEnd() == INFINITE_TF_int64) {
    uint64_t checkInterval = mCheckIntervalInfiniteSlot + mLastCheckedTFInfiniteSlot;
    if (mWasCheckedInfiniteSlot) {
      checkInterval = mCheckDeltaIntervalInfiniteSlot + mLastCheckedTFInfiniteSlot;
    }
    if (tf >= checkInterval || tf == INFINITE_TF) {
      LOG(DEBUG) << "mMaxSeenTF = " << mMaxSeenTF << ", mLastCheckedTFInfiniteSlot = " << mLastCheckedTFInfiniteSlot << ", checkInterval = " << checkInterval << ", mSlots[0].getTFStart() = " << mSlots[0].getTFStart();
      if (tf == INFINITE_TF) {
        LOG(INFO) << "End of run reached, trying to calibrate what we have, if we have enough statistics";
      } else {
        LOG(INFO) << "Calibrating as soon as we have enough statistics:";
        LOG(INFO) << "Update interval passed (" << checkInterval << "), checking slot for " << mSlots[0].getTFStart() << " <= TF <= " << mSlots[0].getTFEnd();
      }
      mLastCheckedTFInfiniteSlot = tf;
      if (hasEnoughData(mSlots[0])) {
        mWasCheckedInfiniteSlot = false;
        mSlots[0].setTFStart(mLastClosedTF);
        mSlots[0].setTFEnd(mMaxSeenTF);
        LOG(INFO) << "Finalizing slot for " << mSlots[0].getTFStart() << " <= TF <= " << mSlots[0].getTFEnd();
        finalizeSlot(mSlots[0]);                  // will be removed after finalization
        mLastClosedTF = mSlots[0].getTFEnd() + 1; // will not accept any TF below this
        mSlots.erase(mSlots.begin());
        // creating a new slot if we are not at the end of run
        if (tf != INFINITE_TF) {
          LOG(INFO) << "Creating new slot for " << mLastClosedTF << " <= TF <= " << INFINITE_TF_int64;
          emplaceNewSlot(true, mLastClosedTF, INFINITE_TF_int64);
        }
      } else {
        LOG(INFO) << "Not enough data to calibrate";
        mWasCheckedInfiniteSlot = true;
      }
    } else {
      LOG(DEBUG) << "Not trying to calibrate: either not at EoS, or update interval not passed";
    }
  } else {
    // check if some slots are done
    for (auto slot = mSlots.begin(); slot != mSlots.end(); slot++) {
      //if (maxDelay == 0 || (slot->getTFEnd() + maxDelay) < tf) {
      if ((slot->getTFEnd() + maxDelay) < tf) {
        if (hasEnoughData(*slot)) {
          LOG(DEBUG) << "Finalizing slot for " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd();
          finalizeSlot(*slot); // will be removed after finalization
        } else if ((slot + 1) != mSlots.end()) {
          LOG(INFO) << "Merging underpopulated slot " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd()
                    << " to slot " << (slot + 1)->getTFStart() << " <= TF <= " << (slot + 1)->getTFEnd();
          (slot + 1)->mergeToPrevious(*slot);
        } else {
          LOG(INFO) << "Discard underpopulated slot " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd();
          break; // slot has no enough stat. and there is no other slot to merge it to
        }
        mLastClosedTF = slot->getTFEnd() + 1; // will not accept any TF below this
        LOG(INFO) << "closing slot " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd();
        mSlots.erase(slot);
      } else {
        break;
      }
      if (mSlots.empty()) { // since erasing the very last entry may invalidate mSlots.end()
        break;
      }
    }
  }
}

//_________________________________________________
template <typename Input, typename Container>
void TimeSlotCalibration<Input, Container>::finalizeOldestSlot()
{
  // Enforce finalization and removal of the oldest slot
  if (mSlots.empty()) {
    LOG(WARNING) << "There are no slots defined";
    return;
  }
  finalizeSlot(mSlots.front());
  mLastClosedTF = mSlots.front().getTFEnd() + 1; // do not accept any TF below this
  mSlots.erase(mSlots.begin());
}

//________________________________________
template <typename Input, typename Container>
inline TFType TimeSlotCalibration<Input, Container>::tf2SlotMin(TFType tf) const
{

  // returns the min TF of the slot to which "tf" belongs

  if (tf < mFirstTF) {
    throw std::runtime_error("invalide TF");
  }
  if (mUpdateAtTheEndOfRunOnly) {
    return mFirstTF;
  }
  return TFType((tf - mFirstTF) / mSlotLength) * mSlotLength + mFirstTF;
}

//_________________________________________________
template <typename Input, typename Container>
TimeSlot<Container>& TimeSlotCalibration<Input, Container>::getSlotForTF(TFType tf)
{

  LOG(DEBUG) << "Getting slot for TF " << tf;

  if (mUpdateAtTheEndOfRunOnly) {
    if (!mSlots.empty() && mSlots.back().getTFEnd() < tf) {
      mSlots.back().setTFEnd(tf);
    } else if (mSlots.empty()) {
      emplaceNewSlot(true, mFirstTF, tf);
    }
    return mSlots.back();
  }

  if (!mSlots.empty() && mSlots.front().getTFStart() > tf) { // we need to add a slot to the beginning
    auto tfmn = tf2SlotMin(mSlots.front().getTFStart() - 1); // min TF of the slot corresponding to a TF smaller than the first seen
    auto tftgt = tf2SlotMin(tf);                             // min TF of the slot to which the TF "tf" would belong
    while (tfmn >= tftgt) {
      LOG(INFO) << "Adding new slot for " << tfmn << " <= TF <= " << tfmn + mSlotLength - 1;
      emplaceNewSlot(true, tfmn, tfmn + mSlotLength - 1);
      if (!tfmn) {
        break;
      }
      tfmn = tf2SlotMin(mSlots.front().getTFStart() - 1);
    }
    return mSlots[0];
  }
  for (auto it = mSlots.begin(); it != mSlots.end(); it++) {
    auto rel = (*it).relateToTF(tf);
    if (rel == 0) {
      return (*it);
    }
  }
  // need to add in the end
  auto tfmn = mSlots.empty() ? tf2SlotMin(tf) : tf2SlotMin(mSlots.back().getTFEnd() + 1);
  do {
    LOG(INFO) << "Adding new slot for " << tfmn << " <= TF <= " << tfmn + mSlotLength - 1;
    emplaceNewSlot(false, tfmn, tfmn + mSlotLength - 1);
    tfmn = tf2SlotMin(mSlots.back().getTFEnd() + 1);
  } while (tf > mSlots.back().getTFEnd());

  return mSlots.back();
}

//_________________________________________________
template <typename Input, typename Container>
void TimeSlotCalibration<Input, Container>::print() const
{
  for (int i = 0; i < getNSlots(); i++) {
    LOG(INFO) << "Slot #" << i << " of " << getNSlots();
    getSlot(i).print();
  }
}

} // namespace calibration
} // namespace o2

#endif
