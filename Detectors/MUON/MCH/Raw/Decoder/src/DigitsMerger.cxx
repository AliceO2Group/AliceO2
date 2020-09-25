// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DataDecoder.cxx
/// \author  Andrea Ferrero
///
/// \brief Implementation of a data processor to run the raw decoding
///

#include "DigitsMerger.h"

#include <iostream>

static bool mPrint = false;

namespace o2
{
namespace mch
{
namespace raw
{

void FeeIdMerger::setOrbit(uint32_t orbit, bool stop)
{
  // perform the merging and send digits of previous orbit if either the stop RDH is received
  // or a new orbit is started
  if ((orbit == currentBuffer.orbit) && (!stop)) {
    return;
  }

  // do the merging
  mergeDigits();

  // send the merged digits
  int nSent = 0;
  for (auto& d : previousBuffer.digits) {
    if (!d.second && (d.first.getPadID() >= 0)) {
      sendDigit(d.first);
      nSent += 1;
    }
  }
  if (mPrint) {
    std::cout << "[FeeIdMerger] sent " << nSent << " digits for orbit " << previousBuffer.orbit << "  current orbit is " << orbit << std::endl;
  }

  // clear the contents of the buffer from the previous orbit, and swap the vectors
  previousBuffer.digits.clear();
  std::swap(previousBuffer.digits, currentBuffer.digits);
  currentBuffer.orbit = orbit;
}

// helper function to check if two digits correspond to the same pad;
static bool areSamePad(const Digit& d1, const Digit& d2)
{
  if (d1.getDetID() != d2.getDetID())
    return false;
  if (d1.getPadID() != d2.getPadID())
    return false;

  return true;
}


static void mergeTwoDigits(const Digit& d1, Digit& d2)
{
  d2.setADC(d1.getADC() + d2.getADC());
  d2.setNofSamples(d1.nofSamples() + d2.nofSamples());
}


void FeeIdMerger::mergeDigits()
{
  const uint32_t bxCounterRollover = 0x100000;
  const uint32_t oneADCclockCycle = 4;

  auto updateDigits = [](Digit& d1, Digit& d2) -> bool {
    // skip all digits not matching the detId/padId
    if (!areSamePad(d1, d2)) {
      return false;
    }

    // compute time difference
    Digit::Time startTime = d1.getTime();
    uint32_t bxStart = startTime.bunchCrossing;
    Digit::Time stopTime = d2.getTime();
    stopTime.sampaTime += d2.nofSamples() - 1;
    uint32_t bxStop = stopTime.bunchCrossing;
    // correct for value rollover
    if (bxStart < bxStop) {
      bxStart += bxCounterRollover;
    }

    uint32_t stopTimeFull = bxStop + (stopTime.sampaTime << 2);
    uint32_t startTimeFull = bxStart + (startTime.sampaTime << 2);
    uint32_t timeDiff = startTimeFull - stopTimeFull;

    if (mPrint) {
      std::cout << "updateDigits: " << d1.getDetID() << "  " << d1.getPadID() << "  " << bxStop << "  " << stopTime.sampaTime
                << "  " << bxStart << "  " << startTime.sampaTime << "  " << timeDiff << std::endl;
    }

    // skip if the time difference is not equal to 1 ADC clock cycle
    if (timeDiff != oneADCclockCycle) {
      return false;
    }

    // merge digits
    mergeTwoDigits(d1, d2);

    return true;
  };

  auto& currentBuffer = getCurrentBuffer();
  auto& previousBuffer = getPreviousBuffer();

  if (mPrint) {
    std::cout << "Merging digits in " << previousBufId << " (orbit=" << previousBuffer.orbit << ")\n";
  }

  for (size_t i = 0; i < previousBuffer.digits.size(); i++) {
    auto& d1 = previousBuffer.digits[i];

    // skip already merged digits
    if (d1.second) {
      continue;
    }

    // skip digits that do not start at the beginning of the time window
    Digit::Time startTime = d1.first.getTime();
    if (startTime.sampaTime != 0) {
      continue;
    }

    for (size_t j = 0; j < previousBuffer.digits.size(); j++) {
      if (i == j) {
        continue;
      }

      auto& d2 = previousBuffer.digits[j];

      // skip already merged digits
      if (d2.second) {
        continue;
      }

      if (updateDigits(d1.first, d2.first)) {
        // mark d1 as merged
        d1.second = true;
        break;
      }
    }
  }

  // only merge digits from consecutive orbits
  uint32_t orbit_p = previousBuffer.orbit;
  uint32_t orbit_c = currentBuffer.orbit;
  if (mPrint) {
    std::cout << "orbit_c: " << orbit_c << "  orbit_p: " << orbit_p << std::endl;
  }
  if ((orbit_c >= orbit_p) && ((orbit_c - orbit_p) > 1)) {
    return;
  }

  if (mPrint) {
    std::cout << "Merging digits from current buffer (orbit=" << currentBuffer.orbit
              << ") into previous buffer (orbit=" << previousBuffer.orbit << ")\n";
  }

  for (size_t i = 0; i < currentBuffer.digits.size(); i++) {
    auto& d1 = currentBuffer.digits[i];

    // skip already merged digits
    if (d1.second) {
      continue;
    }

    // skip digits that do not start at the beginning of the time window
    Digit::Time startTime = d1.first.getTime();
    if (startTime.sampaTime != 0) {
      continue;
    }

    for (size_t j = 0; j < previousBuffer.digits.size(); j++) {
      auto& d2 = previousBuffer.digits[j];

      // skip already merged digits
      if (d2.second) {
        continue;
      }

      if (updateDigits(d1.first, d2.first)) {
        // mark d1 as merged
        d1.second = true;
        break;
      }
    }
  }
}

void Merger::setOrbit(int feeId, uint32_t orbit, bool stop)
{
  if (feeId < 0 || feeId > MCH_MERGER_FEEID_MAX) {
    return;
  }

  mergers[feeId].setOrbit(orbit, stop);
}

void Merger::setDigitHandler(std::function<void(const Digit&)> h)
{
  for (int feeId = 0; feeId <= MCH_MERGER_FEEID_MAX; feeId++) {
    mergers[feeId].setDigitHandler(h);
  }
}

void Merger::addDigit(int feeId, int solarId, int dsAddr, int chAddr,
                      int deId, int padId, int adc, Digit::Time time, uint16_t nSamples)
{
  if (feeId < 0 || feeId > MCH_MERGER_FEEID_MAX) {
    return;
  }

  mergers[feeId].getCurrentBuffer().digits.emplace_back(std::make_pair(o2::mch::Digit{deId, padId, adc, time, nSamples}, false));
}

void Merger::mergeDigits(int feeId)
{
  if (feeId < 0 || feeId > MCH_MERGER_FEEID_MAX) {
    return;
  }

  mergers[feeId].mergeDigits();
}

} // namespace raw
} // namespace mch
} // end namespace o2
