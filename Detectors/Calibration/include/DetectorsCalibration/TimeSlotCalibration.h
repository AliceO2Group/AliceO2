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

#ifndef DETECTOR_CALIB_TIMESLOTCALIB_H_
#define DETECTOR_CALIB_TIMESLOTCALIB_H_

/// @brief Processor for the multiple time slots calibration

#include "DetectorsCalibration/TimeSlot.h"
#include "DetectorsBase/TFIDInfoHelper.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonDataFormat/TFIDInfo.h"
#include <deque>
#include <gsl/gsl>
#include <limits>
#include <type_traits>

namespace o2
{
namespace framework
{
class ProcessingContext;
}
namespace calibration
{

template <typename Input, typename Container>
class TimeSlotCalibration
{
 public:
  using Slot = TimeSlot<Container>;
  using TFType = o2::calibration::TFType;

  static constexpr TFType INFINITE_TF = o2::calibration::INFINITE_TF;

  TimeSlotCalibration() = default;
  virtual ~TimeSlotCalibration() = default;
  float getMaxSlotsDelay() const { return mMaxSlotsDelay; }
  void setMaxSlotsDelay(float v) { mMaxSlotsDelay = v > 0. ? v : 0.; }

  TFType getSlotLength() const { return mSlotLength; }
  void setSlotLength(TFType v)
  {
    if (v == 0) {
      setFinalizeWhenReady();
    } else {
      mSlotLength = v;
    }
  }

  TFType getCheckIntervalInfiniteSlot() const { return mCheckIntervalInfiniteSlot; }
  void setCheckIntervalInfiniteSlot(TFType v) { mCheckIntervalInfiniteSlot = v; }

  TFType getCheckDeltaIntervalInfiniteSlot() const { return mCheckDeltaIntervalInfiniteSlot; }
  void setCheckDeltaIntervalInfiniteSlot(TFType v) { mCheckDeltaIntervalInfiniteSlot = v < 1 ? mCheckIntervalInfiniteSlot : v; } // if the delta is 0, we ignore it

  TFType getFirstTF() const { return mFirstTF; }
  void setFirstTF(TFType v) { mFirstTF = v; }

  void setSlotLengthInSeconds(int s) { mSlotLengthInSeconds = s > 0 ? s : 1; }
  void setSlotLengthInOrbits(int n) { mSlotLengthInOrbits = n > 0 ? n : 1; }
  void checkSlotLength()
  {
    if (mSlotLengthInSeconds > 0) {
      TFType ntf = mSlotLengthInSeconds / (o2::base::GRPGeomHelper::getNHBFPerTF() * o2::constants::lhc::LHCOrbitMUS * 1e-6);
      LOGP(info, "Redefining slot duration from {} s. to {} TFs", mSlotLengthInSeconds, ntf);
      setSlotLength(ntf);
      mSlotLengthInSeconds = 0;
    } else if (mSlotLengthInOrbits > 0) {
      TFType ntf = mSlotLengthInOrbits / o2::base::GRPGeomHelper::getNHBFPerTF();
      if (ntf < 1) {
        ntf = 1;
      }
      LOGP(info, "Redefining slot duration from {} orbits to {} TFs", mSlotLengthInOrbits, ntf);
      setSlotLength(ntf);
      mSlotLengthInOrbits = 0;
    }
  }

  void setFinalizeWhenReady()
  {
    mFinalizeWhenReady = true;
    setSlotLength(INFINITE_TF);
  }

  void setUpdateAtTheEndOfRunOnly() { mUpdateAtTheEndOfRunOnly = kTRUE; }

  int getNSlots() const { return mSlots.size(); }
  Slot& getSlotForTF(TFType tf);
  Slot& getSlot(int i) { return (Slot&)mSlots.at(i); }
  const Slot& getSlot(int i) const { return (Slot&)mSlots.at(i); }
  const Slot& getLastSlot() const { return (Slot&)mSlots.back(); }
  const Slot& getFirstSlot() const { return (Slot&)mSlots.front(); }

  template <typename DATA>
  bool process(const DATA& data);
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

  const o2::dataformats::TFIDInfo& getCurrentTFInfo() const { return mCurrentTFInfo; }
  o2::dataformats::TFIDInfo& getCurrentTFInfo() { return mCurrentTFInfo; }

  // from  https://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature
  // Primary template with a static assertion
  // for a meaningful error message
  // if it ever gets instantiated.
  // We could leave it undefined if we didn't care.
  template <typename, typename T>
  struct has_fill_method {
    static_assert(
      std::integral_constant<T, false>::value,
      "Second template parameter needs to be of function type.");
  };

  // specialization that does the checking

  template <typename C, typename Ret, typename... Args>
  struct has_fill_method<C, Ret(Args...)> {
   private:
    template <typename T>
    static constexpr auto check(T*)
      -> typename std::is_same<decltype(std::declval<T>().fill(std::declval<Args>()...)), Ret>::type; // attempt to call it and see if the return type is correct
    template <typename>
    static constexpr std::false_type check(...);
    typedef decltype(check<C>(nullptr)) type;

   public:
    static constexpr bool value = type::value;
  };

 protected:
  auto& getSlots() { return mSlots; }
  uint32_t getRunStartOrbit() const
  {
    long orb = long(mCurrentTFInfo.firstTForbit) - long(o2::base::GRPGeomHelper::getNHBFPerTF() * mCurrentTFInfo.tfCounter);
    if (orb < 0) {
      LOGP(alarm, "Negative runStartOrbit = {} deduced from tfCounter={} and firstTForbit={}, enforcing runStartOrbit to 0", orb, mCurrentTFInfo.tfCounter, mCurrentTFInfo.firstTForbit);
      orb = 0;
    }
    return uint32_t(orb);
  }

  TFType tf2SlotMin(TFType tf) const;

  std::deque<Slot> mSlots;

  o2::dataformats::TFIDInfo mCurrentTFInfo{};
  int mSlotLengthInSeconds = -1; // optionally provided slot length in seconds
  int mSlotLengthInOrbits = -1;  // optionally provided slot length in orbits
  TFType mLastClosedTF = 0;
  TFType mFirstTF = 0;
  TFType mMaxSeenTF = 0;                        // largest TF processed
  TFType mSlotLength = 1;                       // slot length in TFs
  TFType mCheckIntervalInfiniteSlot = 1;        // will be used if the TF length is INFINITE_TF_int64 to decide
                                                // when to check if to call the finalize; otherwise it is called
                                                // at every new TF; note that this is an approximation,
                                                // since TFs come in async order
  TFType mLastCheckedTFInfiniteSlot = 0;        // will be used if the TF length is INFINITE_TF_int64 to book-keep
                                                // the last TF at which we tried to calibrate
  TFType mCheckDeltaIntervalInfiniteSlot = 1;   // will be used if the TF length is INFINITE_TF_int64 when
                                                // the check on the statistics returned false, to determine
                                                // after how many TF to check again.
  float mMaxSlotsDelay = 3.0;                   // difference in slot units between the current TF and oldest slot (end TF) to account for the TF

  bool mWasCheckedInfiniteSlot = false;         // flag to know whether the statistics of the infinite slot was already checked
  bool mUpdateAtTheEndOfRunOnly = false;
  bool mFinalizeWhenReady = false; // if true: single bin is filled until ready, then closed and new one is added
  ClassDef(TimeSlotCalibration, 1);
};

//_________________________________________________
template <typename Input, typename Container>
template <typename DATA>
bool TimeSlotCalibration<Input, Container>::process(const DATA& data)
{
  static bool firstCall = true;
  if (firstCall) {
    firstCall = false;
    checkSlotLength();
  }

  // process current TF
  TFType tf = mCurrentTFInfo.tfCounter;
  uint64_t maxDelay64 = uint64_t(mSlotLength * mMaxSlotsDelay);
  TFType maxDelay = maxDelay64 > o2::calibration::INFINITE_TF ? o2::calibration::INFINITE_TF : TFType(maxDelay64);
  if (!mUpdateAtTheEndOfRunOnly) {                                                               // if you update at the end of run only, then you accept everything
    if (tf < mLastClosedTF || (!mSlots.empty() && getLastSlot().getTFStart() > tf + maxDelay64)) { // ignore TF; note that if you have only 1 timeslot
                                                                                                   // which is INFINITE_TF wide, then maxDelay
                                                                                                   // does not matter: you won't accept TFs from the past,
                                                                                                   // so the first condition will be used
      LOG(info) << "Ignoring TF " << tf << ", mLastClosedTF = " << mLastClosedTF;
      return false;
    }
  }
  auto& slotTF = getSlotForTF(tf);
  using Cont_t = typename std::remove_pointer<decltype(slotTF.getContainer())>::type;
  if constexpr (has_fill_method<Cont_t, void(const o2::dataformats::TFIDInfo&, const DATA&)>::value) {
    slotTF.getContainer()->fill(mCurrentTFInfo, data);
  } else {
    slotTF.getContainer()->fill(data);
  }
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

  // if slot finalization is asked as soon as the slot is ready, we need to check if we got enough statistics, and if so, redefine the slot
  if (mSlots.size() == 1 && mFinalizeWhenReady) {
    TFType checkInterval = mCheckIntervalInfiniteSlot + mLastCheckedTFInfiniteSlot;
    if (mWasCheckedInfiniteSlot) {
      checkInterval = mCheckDeltaIntervalInfiniteSlot + mLastCheckedTFInfiniteSlot;
    }
    if (tf >= checkInterval || tf == INFINITE_TF) {
      LOG(debug) << "mMaxSeenTF = " << mMaxSeenTF << ", mLastCheckedTFInfiniteSlot = " << mLastCheckedTFInfiniteSlot << ", checkInterval = " << checkInterval << ", mSlots[0].getTFStart() = " << mSlots[0].getTFStart();
      if (tf == INFINITE_TF) {
        LOG(info) << "End of run reached, trying to calibrate what we have, if we have enough statistics";
      } else {
        LOG(info) << "Calibrating as soon as we have enough statistics:";
        LOG(info) << "Update interval passed (" << checkInterval << "), checking slot for " << mSlots[0].getTFStart() << " <= TF <= " << INFINITE_TF;
      }
      mLastCheckedTFInfiniteSlot = tf;
      if (hasEnoughData(mSlots[0])) {
        mWasCheckedInfiniteSlot = false;
        mSlots[0].setTFStart(mLastClosedTF);
        mSlots[0].setTFEnd(mMaxSeenTF);
        LOG(info) << "Finalizing slot for " << mSlots[0].getTFStart() << " <= TF <= " << mSlots[0].getTFEnd();
        finalizeSlot(mSlots[0]);                  // will be removed after finalization
        mLastClosedTF = mSlots[0].getTFEnd() < INFINITE_TF ? (mSlots[0].getTFEnd() + 1) : mSlots[0].getTFEnd() < INFINITE_TF; // will not accept any TF below this
        mSlots.erase(mSlots.begin());
        // creating a new slot if we are not at the end of run
        if (tf != INFINITE_TF) {
          LOG(info) << "Creating new slot for " << mLastClosedTF << " <= TF <= " << INFINITE_TF;
          auto& sl = emplaceNewSlot(true, mLastClosedTF, INFINITE_TF);
          sl.setRunStartOrbit(getRunStartOrbit());
        }
      } else {
        LOG(info) << "Not enough data to calibrate";
        mWasCheckedInfiniteSlot = true;
      }
    } else {
      LOG(debug) << "Not trying to calibrate: either not at EoS, or update interval not passed";
    }
  } else {
    // check if some slots are done
    for (auto slot = mSlots.begin(); slot != mSlots.end();) {
      uint64_t lim64 = uint64_t(maxDelay) + slot->getTFEnd();
      TFType tfLim = lim64 < INFINITE_TF ? TFType(lim64) : INFINITE_TF;
      if (tfLim < tf) {
        if (hasEnoughData(*slot)) {
          LOG(debug) << "Finalizing slot for " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd();
          finalizeSlot(*slot); // will be removed after finalization
        } else if ((slot + 1) != mSlots.end()) {
          LOG(info) << "Merging underpopulated slot " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd()
                    << " to slot " << (slot + 1)->getTFStart() << " <= TF <= " << (slot + 1)->getTFEnd();
          (slot + 1)->mergeToPrevious(*slot);
        } else {
          LOG(info) << "Discard underpopulated slot " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd();
          break; // slot has no enough stat. and there is no other slot to merge it to
        }
        mLastClosedTF = slot->getTFEnd() + 1; // will not accept any TF below this
        LOG(info) << "closing slot " << slot->getTFStart() << " <= TF <= " << slot->getTFEnd();
        slot = mSlots.erase(slot);
      } else {
        break; // all following slots will be even closer to the new TF
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
    LOG(warning) << "There are no slots defined";
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
    throw std::runtime_error("invalid TF");
  }
  if (mUpdateAtTheEndOfRunOnly) {
    return mFirstTF;
  }
  uint64_t tft = int64_t(((tf - mFirstTF) / mSlotLength) * mSlotLength) + mFirstTF;
  return tft < o2::calibration::INFINITE_TF ? TFType(tft) : INFINITE_TF;
}

//_________________________________________________
template <typename Input, typename Container>
TimeSlot<Container>& TimeSlotCalibration<Input, Container>::getSlotForTF(TFType tf)
{

  LOG(debug) << "Getting slot for TF " << tf;
  if (mUpdateAtTheEndOfRunOnly) {
    if (!mSlots.empty() && mSlots.back().getTFEnd() < tf) {
      mSlots.back().setTFEnd(tf);
    } else if (mSlots.empty()) {
      auto& sl = emplaceNewSlot(true, mFirstTF, tf);
      sl.setRunStartOrbit(getRunStartOrbit());
      sl.setStaticStartTimeMS(sl.getStartTimeMS());
    }
    return mSlots.back();
  }

  if (!mSlots.empty() && mSlots.front().getTFStart() > tf) { // we need to add a slot to the beginning
    auto tfmn = tf2SlotMin(mSlots.front().getTFStart() - 1); // min TF of the slot corresponding to a TF smaller than the first seen
    auto tftgt = tf2SlotMin(tf);                             // min TF of the slot to which the TF "tf" would belong
    while (tfmn >= tftgt) {
      uint64_t tft = uint64_t(tfmn) + mSlotLength - 1;
      TFType tfmx = tft < o2::calibration::INFINITE_TF ? TFType(tft) : o2::calibration::INFINITE_TF;
      LOG(info) << "Adding new slot for " << tfmn << " <= TF <= " << tfmx;
      auto& sl = emplaceNewSlot(true, tfmn, tfmx);
      sl.setRunStartOrbit(getRunStartOrbit());
      sl.setStaticStartTimeMS(sl.getStartTimeMS());
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
    uint64_t tft = uint64_t(tfmn) + mSlotLength - 1;
    TFType tfmx = tft < o2::calibration::INFINITE_TF ? TFType(tft) : o2::calibration::INFINITE_TF;
    LOG(info) << "Adding new slot for " << tfmn << " <= TF <= " << tfmx;
    auto& sl = emplaceNewSlot(false, tfmn, tfmx);
    sl.setRunStartOrbit(getRunStartOrbit());
    sl.setStaticStartTimeMS(sl.getStartTimeMS());
    tfmn = tf2SlotMin(mSlots.back().getTFEnd() + 1);
  } while (tf > mSlots.back().getTFEnd());

  return mSlots.back();
}

//_________________________________________________
template <typename Input, typename Container>
void TimeSlotCalibration<Input, Container>::print() const
{
  for (int i = 0; i < getNSlots(); i++) {
    LOG(info) << "Slot #" << i << " of " << getNSlots();
    getSlot(i).print();
  }
}

} // namespace calibration
} // namespace o2

#endif
