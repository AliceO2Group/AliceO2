// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawDecoder/ROFFinder.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <fmt/format.h>

#include <fairmq/Tools.h>
#include <FairMQLogger.h>

#include "MCHMappingInterface/Segmentation.h"
#include "CommonConstants/LHCConstants.h"

//#define ROFDEBUG 1

namespace o2
{
namespace mch
{
namespace raw
{

using namespace std;

//_________________________________________________________________________________________________
ROFFinder::ROFFinder(const DataDecoder::RawDigitVector& digits, uint32_t firstTForbit) : mInputDigits(digits), mFirstTForbit(firstTForbit)
{
}

//_________________________________________________________________________________________________
ROFFinder::~ROFFinder() = default;

//_________________________________________________________________________________________________
void ROFFinder::process(bool dummyROFs)
{
  if (dummyROFs) {
    if (!mInputDigits.empty()) {
      mOrderedDigits.resize(mInputDigits.size());
      // fill the ordered vector with indexes in ascending order (no time sorting)
      std::iota(mOrderedDigits.begin(), mOrderedDigits.end(), 0);
      mOutputROFs.emplace_back(digitTime2IR(mInputDigits[0]), 0, mInputDigits.size());
    }
    return;
  }

  // helper function to check if a given digit index is within the limits of the input vector
  auto checkDigitId = [&](RawDigitId id) -> bool {
    bool ok = id < mInputDigits.size();
    if (!ok) {
      LOG(ERROR) << "Invalid digit ID " << id << " (digits vector size is " << mInputDigits.size() << ")\n";
    }
    return ok;
  };

  // helper function to retrieve the digit at a given index
  auto getDigit = [&](RawDigitId id) -> const RawDigit& {
    return mInputDigits[id];
  };

  // helper function to initialize the parameters of the next ROF
  auto startNewROF = [&](const RawDigit& digit, int id) {
    mFirstIdx = id;
    mEntries = 1;
    mIR = digitTime2IR(digit);
  };

  // fill the time-ordered digit vector
  sortDigits();

  // loop on time-ordered digits, grouping together in one ROF all those whose time stamp
  // differs by less than 4 bunch crossings (1 ADC clock cycle)
  for (size_t id = 0; id < mOrderedDigits.size(); id++) {

    // index of the current digit in the input vector
    RawDigitId inputId = mOrderedDigits[id];

    // mak sure the index is valid
    if (!checkDigitId(inputId)) {
      break;
    }
    // get a reference to the current digit, which marks the beginning of a new ROF
    auto& rofSeed = getDigit(inputId);
    startNewROF(rofSeed, id);

#ifdef ROFDEBUG
    std::cout << fmt::format("starting new ROF from {} -> {}\n", id, inputId);
#endif

    for (size_t id2 = id + 1; id2 < mOrderedDigits.size(); id2++) {

      // index of the current digit in the input vector
      RawDigitId inputId2 = mOrderedDigits[id2];

      if (!checkDigitId(inputId2)) {
        break;
      }
      auto& digit = getDigit(inputId2);

#ifdef ROFDEBUG
      std::cout << fmt::format("  checking digit {} -> {}\n", id2, inputId2);
#endif

      constexpr int oneADCclock = 4;
      auto tdiff = digit.getTime() - rofSeed.getTime();
      if (std::abs(tdiff) < oneADCclock) {
        mEntries += 1;
#ifdef ROFDEBUG
        std::cout << fmt::format("  digit {} -> {} added to current ROF\n", id2, inputId2);
#endif
      } else {
        // terminate and store the current ROF and stop the inner loop
        storeROF();
        break;
      }
    }

    // increment the outer loop index by the number of digits added to the ROF in the inner loop
    id += mEntries - 1;
  }

  // terminate and store the last ROF
  storeROF();
}

//_________________________________________________________________________________________________
void ROFFinder::sortDigits()
{
  mOrderedDigits.reserve(mInputDigits.size());
  for (size_t i = 0; i < mInputDigits.size(); i++) {
    auto& digit = mInputDigits[i];
    if (!digit.timeValid()) {
      LOG(ERROR) << "Digit with invalid time, DS " << digit.info.solar << "," << digit.info.ds << "," << digit.info.chip
                 << "  pad " << digit.getDetID() << "," << digit.getPadID() << "  "
                 << digit.getOrbit() << " (" << mFirstTForbit << ") "
                 << digit.getTime() << digit.getBXTime();
      continue;
    }

    auto orbit = digit.getOrbit();
    if (orbit < mFirstTForbit) {
      LOG(ERROR) << "[ROFFinder::fillDigitsArray] orbit smaller than first TF orbit: " << orbit << ", " << mFirstTForbit;
      continue;
    }

#ifdef ROFDEBUG
    std::cout << "Inserting digit: "
              << "pad " << digit.getDetID() << "," << digit.getPadID() << " "
              << digit.getOrbit() << " " << digit.getTime() << " " << digit.getBXTime() << std::endl;
#endif

    mOrderedDigits.emplace_back(i);
  }

  auto rawDigitIdComp = [&](const RawDigitId& id1, const RawDigitId& id2) -> bool {
    const RawDigit& d1 = mInputDigits[id1];
    const RawDigit& d2 = mInputDigits[id2];
    return (d1 < d2);
  };
  std::sort(mOrderedDigits.begin(), mOrderedDigits.end(), rawDigitIdComp);
}

//_________________________________________________________________________________________________
o2::InteractionRecord ROFFinder::digitTime2IR(const RawDigit& digit)
{
  constexpr int BCINORBIT = o2::constants::lhc::LHCMaxBunches;
  auto time = digit.getTime();
  auto firstOrbit = mFirstTForbit;

  // make sure the interaction record is not initialized with negative BC values
  while (time < 0) {
    time += BCINORBIT;
    firstOrbit -= 1;
  }

  uint32_t orbit = digit.getTime() / BCINORBIT + firstOrbit;
  int32_t bc = time % BCINORBIT;
  return o2::InteractionRecord(bc, orbit);
}

//_________________________________________________________________________________________________
void ROFFinder::storeROF()
{
  if (mEntries > 0) {
    mOutputROFs.emplace_back(mIR, mFirstIdx, mEntries);
  }
}

//_________________________________________________________________________________________________
char* ROFFinder::saveDigitsToBuffer(size_t& bufSize)
{
  static constexpr size_t sizeOfDigit = sizeof(o2::mch::Digit);

#ifdef ROFDEBUG
  dumpOutputDigits();
#endif

  bufSize = mOrderedDigits.size() * sizeOfDigit;
  o2::mch::Digit* buf = reinterpret_cast<o2::mch::Digit*>(malloc(bufSize));
  if (!buf) {
    bufSize = 0;
    return nullptr;
  }

  o2::mch::Digit* p = buf;
  for (auto& id : mOrderedDigits) {
    const auto& d = mInputDigits[id];
    memcpy(p, &(d.digit), sizeOfDigit);
    p += 1;
  }

  return reinterpret_cast<char*>(buf);
}

//_________________________________________________________________________________________________
char* ROFFinder::saveROFRsToBuffer(size_t& bufSize)
{
  static constexpr size_t sizeOfROFRecord = sizeof(o2::mch::ROFRecord);

#ifdef ROFDEBUG
  dumpOutputROFs();
#endif

  bufSize = mOutputROFs.size() * sizeOfROFRecord;
  o2::mch::ROFRecord* buf = reinterpret_cast<o2::mch::ROFRecord*>(malloc(bufSize));
  if (!buf) {
    bufSize = 0;
    return nullptr;
  }

  o2::mch::ROFRecord* p = buf;
  for (size_t i = 0; i < mOutputROFs.size(); i++) {
    auto& rof = mOutputROFs[i];
    memcpy(p, &(rof), sizeOfROFRecord);
    p += 1;
  }

  return reinterpret_cast<char*>(buf);
}

//_________________________________________________________________________________________________
bool ROFFinder::isRofTimeMonotonic()
{
  // number of bunch crossings in one orbit
  static const int32_t BCINORBIT = o2::constants::lhc::LHCMaxBunches;

  bool result = true;
  for (size_t i = 1; i < mOutputROFs.size(); i++) {
    const auto& rof = mOutputROFs[i];
    const auto& rofPrev = mOutputROFs[i - 1];
    int64_t delta = rof.getBCData().differenceInBC(rofPrev.getBCData());
    if (rof.getBCData() < rofPrev.getBCData()) {
      LOG(ERROR) << "Non-monotonic ROFs encountered:";
      LOG(ERROR) << fmt::format("ROF1 {}-{} {},{}  ", rofPrev.getFirstIdx(), rofPrev.getLastIdx(),
                                rofPrev.getBCData().orbit, rofPrev.getBCData().bc)
                 << fmt::format("ROF2 {}-{} {},{}", rof.getFirstIdx(), rof.getLastIdx(),
                                rof.getBCData().orbit, rof.getBCData().bc);
      result = false;
    }
    if ((delta % 4) != 0) {
      LOG(ERROR) << "Mis-aligned ROFs encountered:";
      LOG(ERROR) << fmt::format("ROF1 {}-{} {},{}  ", rofPrev.getFirstIdx(), rofPrev.getLastIdx(),
                                rofPrev.getBCData().orbit, rofPrev.getBCData().bc)
                 << fmt::format("ROF2 {}-{} {},{}", rof.getFirstIdx(), rof.getLastIdx(),
                                rof.getBCData().orbit, rof.getBCData().bc);
      result = false;
    }
  }
  return result;
}

//_________________________________________________________________________________________________
bool ROFFinder::isDigitsTimeAligned()
{
  for (size_t i = 0; i < mOutputROFs.size(); i++) {
    auto& rof = mOutputROFs[i];
    for (int j = rof.getFirstIdx() + 1; j <= rof.getLastIdx(); j++) {
      auto id = mOrderedDigits[j];
      auto idPrev = mOrderedDigits[j - 1];
      const auto& digit = mInputDigits[id];
      const auto& digitPrev = mInputDigits[idPrev];
      if (digit.getTime() != digitPrev.getTime()) {
        LOG(ERROR) << "Mis-aligned digits encountered:";
        LOG(ERROR) << fmt::format("TIME1 {}  ", digitPrev.getTime()) << fmt::format("TIME2 {}", digit.getTime());
        return false;
      }
    }
  }
  return true;
}

//_________________________________________________________________________________________________
std::optional<DataDecoder::RawDigit> ROFFinder::getOrderedDigit(int i)
{
  if (i < 0 || i >= mOrderedDigits.size()) {
    return std::nullopt;
  }

  auto id = mOrderedDigits[i];
  if (id >= mInputDigits.size()) {
    return std::nullopt;
  }

  return mInputDigits[id];
}

//_________________________________________________________________________________________________
void ROFFinder::dumpOutputDigits()
{
  std::cout << "OUTPUT DIGITS:\n";
  for (size_t i = 0; i < mOrderedDigits.size(); i++) {
    const auto id = mOrderedDigits[i];
    const auto& digit = mInputDigits[id];
    const auto& d = digit.digit;
    const auto& t = digit.info;

    if (d.getPadID() < 0) {
      continue;
    }
    const o2::mch::mapping::Segmentation& segment = o2::mch::mapping::segmentation(d.getDetID());
    bool bending = segment.isBendingPad(d.getPadID());
    float X = segment.padPositionX(d.getPadID());
    float Y = segment.padPositionY(d.getPadID());
    uint32_t orbit = t.orbit;
    uint32_t bunchCrossing = t.bunchCrossing;
    uint32_t sampaTime = t.sampaTime;
    auto tfTime = digit.getTime();

    int iROF = -1;
    for (size_t j = 0; j < mOutputROFs.size(); j++) {
      const auto& rof = mOutputROFs[j];
      if (rof.getFirstIdx() <= i && rof.getLastIdx() >= i) {
        iROF = j;
      }
    }
    std::cout << fmt::format("    DIGIT [{} -> {}]  ROF {}  DE {:4d}  PAD {:5d}  ADC {:6d}  TIME {} ({} {} {:4d} {})",
                             i, id, iROF, d.getDetID(), d.getPadID(), d.getADC(), tfTime, orbit, bunchCrossing, sampaTime, t.getBXTime());
    std::cout << fmt::format("\tC {}  PAD_XY {:+2.2f} , {:+2.2f}\n", (bending ? (int)0 : (int)1), X, Y);
  }
}

//_________________________________________________________________________________________________
void ROFFinder::dumpOutputROFs()
{
  std::cout << "OUTPUT ROFs:\n";
  for (size_t i = 0; i < mOutputROFs.size(); i++) {
    auto& rof = mOutputROFs[i];
    std::cout << fmt::format("    ROF {} {}-{} {},{}\n", i, rof.getFirstIdx(), rof.getLastIdx(),
                             rof.getBCData().orbit, rof.getBCData().bc);
  }
}

} // namespace raw
} // namespace mch
} // namespace o2
